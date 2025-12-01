# SPDX-License-Identifier: Apache-2.0

import time
from typing import Callable, List, Dict, Optional, Union

import msgspec
import torch

from vllm.model_executor.layers.spec_decode_base_sampler import (
    SpecDecodeBaseSampler)
from vllm.utils import is_pin_memory_available


class SpecDecodeWorkerMetrics(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        array_like=True):  # type: ignore[call-arg]
    """Dataclass holding metrics emitted from the spec decode worker.
    """

    # The empirical acceptance rate of the proposal method on a per-token basis.
    # This is useful for evaluating how well the proposal method aligns with the
    # scoring method.
    draft_acceptance_rate: float

    # The empirical efficiency, measured as the number of tokens emitted by the
    # system divided by the number of tokens that could be emitted by the system
    # if the proposal method were perfect.
    system_efficiency: float

    # The number of speculative tokens produced by the proposal method.
    draft_tokens: int

    # The number of tokens emitted by the entire system.
    emitted_tokens: int

    # The number of tokens accepted by the scoring model and verification
    # routine, e.g. Llama2-70B and lossless rejection sampling.
    #
    # NOTE: Any token accepted by the verification routine is considered
    # accepted (regardless of if the speculative prefix is also accepted). The
    # user will usually see less accepted tokens. This metric is helpful when
    # evaluating alignment of the proposal method with the scoring model.
    accepted_tokens: int

    # The number of speculative tokens per sequence.
    num_spec_tokens: int


Timer = Callable[[], float]


class AsyncMetricsCollector:
    """Class which copies rejection/typical-acceptance sampler metrics
    from the device to CPU on a non-default Torch stream.
    """

    def __init__(self,
                 spec_decode_sampler: SpecDecodeBaseSampler,
                 timer: Optional[Timer] = None,
                 collect_interval_s: float = 5.0):
        self.spec_decode_sampler = spec_decode_sampler
        self._timer = time.time if timer is None else timer

        self._rank: Optional[int] = None

        # These counters now aggregate metrics calculated on the CPU.
        self._aggregate_num_accepted_tokens = 0
        self._aggregate_num_emitted_tokens = 0
        self._aggregate_num_draft_tokens = 0

        # For interval-based metrics (e.g. instantaneous acceptance rate)
        self._prev_accepted_tokens = 0
        self._prev_emitted_tokens = 0
        self._prev_draft_tokens = 0

        self._rejsample_metrics_collect_interval_s = collect_interval_s
        self._last_metrics_collect_time = self._timer()

    def init_gpu_tensors(self, rank: int) -> None:
        self._rank = rank
        self._copy_stream = torch.cuda.Stream()

    def init_tensors(self,
                     rank: int,
                     device_type: Union[torch.device, str] = 'cuda') -> None:
        self._rank = rank

    def update_batch_summary_metrics(
        self,
        current_batch_total_accepted_tokens_for_rate: int,
        current_batch_total_draft_tokens: int,
        current_batch_total_emitted_tokens: int,
    ) -> None:
        """
        This method is the entry point for per-sequence metrics. It is
        called by the worker with CPU-calculated stats for the latest batch.
        """
        self._aggregate_num_accepted_tokens += current_batch_total_accepted_tokens_for_rate
        self._aggregate_num_emitted_tokens += current_batch_total_emitted_tokens
        self._aggregate_num_draft_tokens += current_batch_total_draft_tokens

    # --- Start of Bug Fix ---
    # Renamed 'maybe_collect_rejsample_metrics' back to
    # 'maybe_collect_metrics_for_reporting' to match the calling code.
    def maybe_collect_metrics_for_reporting(
            self, k: int) -> Optional[SpecDecodeWorkerMetrics]:
    # --- End of Bug Fix ---
        """
        Periodically collects and calculates metrics. This method maintains the
        original interface but uses CPU-aggregated data instead of async copies.
        """
        if not self._should_collect_rejsample_metrics(self._timer()):
            return None

        # The internal logic is now a direct call to the collection function.
        return self._collect_rejsample_metrics(k)
    
    def maybe_collect_rejsample_metrics(self, k: int):
        """Alias kept for backward-compat with upstream 0.8.4."""
        return self.maybe_collect_metrics_for_reporting(k)

    def _should_collect_rejsample_metrics(self, now: float) -> bool:
        """Return whether or not this iteration should print sampling
        metrics.
        """
        if self._rank != 0:
            return False

        return now - self._last_metrics_collect_time >= self._rejsample_metrics_collect_interval_s  # noqa: E501

    def get_per_sequence_acceptance_rate(
        self,
        metric_history: List[Dict[str, float]],
    ) -> Optional[float]:
        """
        Calculates the acceptance rate for a specific sequence for dynamic K.
        """
        if not metric_history:
            return None

        latest_metrics = metric_history[-1]
        accepted_len = latest_metrics.get("accepted_len_for_rate", 0.0)
        draft_len = latest_metrics.get("proposed_len", 0.0)
        
        if draft_len > 0:
            return accepted_len / draft_len
        
        return None

    def _collect_rejsample_metrics(
            self, k: int) -> SpecDecodeWorkerMetrics:
        """Create metrics object from the aggregated CPU counters."""
        self._last_metrics_collect_time = self._timer()

        # Calculate metrics over the last collection interval.
        draft_tokens_in_interval = self._aggregate_num_draft_tokens - self._prev_draft_tokens
        accepted_tokens_in_interval = self._aggregate_num_accepted_tokens - self._prev_accepted_tokens
        emitted_tokens_in_interval = self._aggregate_num_emitted_tokens - self._prev_emitted_tokens

        self._prev_draft_tokens = self._aggregate_num_draft_tokens
        self._prev_accepted_tokens = self._aggregate_num_accepted_tokens
        self._prev_emitted_tokens = self._aggregate_num_emitted_tokens
        
        if draft_tokens_in_interval > 0:
            draft_acceptance_rate = accepted_tokens_in_interval / draft_tokens_in_interval
        else:
            draft_acceptance_rate = float("nan")
        
        max_num_emitted_tokens = self.get_max_num_emitted_tokens(
            draft_tokens_in_interval, k)
        
        if max_num_emitted_tokens > 0:
            system_efficiency = emitted_tokens_in_interval / max_num_emitted_tokens
        else:
            system_efficiency = float("nan")

        return SpecDecodeWorkerMetrics(
            num_spec_tokens=k,
            draft_acceptance_rate=draft_acceptance_rate,
            system_efficiency=system_efficiency,
            accepted_tokens=accepted_tokens_in_interval,
            draft_tokens=draft_tokens_in_interval,
            emitted_tokens=emitted_tokens_in_interval,
        )

    @staticmethod
    def get_max_num_emitted_tokens(draft_tokens: int, k: int) -> int:
        """
        Calculate the number of emitted tokens, assuming all tokens are
        accepted. The +1 comes from the bonus token.
        This version fixes a bug in the original code for per-sequence-K.
        """
        if k <= 0:
            return draft_tokens
        
        if draft_tokens == 0:
            return 0

        # BUG FIX: With per-sequence K, draft_tokens is a sum of varied k_i,
        # so it's not guaranteed to be a multiple of the max-k used for
        # reporting. The original `assert` is removed. We now estimate.
        total_num_spec_seqs = round(draft_tokens / k) if k > 0 else 0
        num_emitted_per_seq_if_all_accepted = k + 1

        return int(total_num_spec_seqs * num_emitted_per_seq_if_all_accepted)
