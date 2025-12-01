# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Set, Tuple

import torch

from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest, SequenceGroupMetadata
from vllm.spec_decode.interfaces import (SpeculativeProposals,
                                         SpeculativeProposer)
from vllm.spec_decode.proposer_worker_base import ProposerWorkerBase
from vllm.spec_decode.util import sampler_output_to_torch


class Top1Proposer(SpeculativeProposer):
    """Helper class which separates out sequences which would exceed the max
    model length when speculated upon.

    This allows combinations of models such as JackFram/llama-68m draft with
    meta-llama/Llama2-13b-chat-hf, as llama-68m has max_position_embeddings of
    2048 while Llama2-13b has max_position_embeddings of 4096.

    We treat the sequences which exceed the proposal draft model length as
    "non-spec sequences". Essentially they skip the draft model and go through
    normal decoding in the target model.

    Currently, only proposal_lens of 0 and k are supported, where k is a global
    batch proposal length. In the future vLLM should support per-sequence
    proposal lengths.
    """

    def __init__(
        self,
        worker: ProposerWorkerBase,
        device: str,
        vocab_size: int,
        max_proposal_len: Optional[int] = None,
    ):
        self._worker = worker
        self._device = device
        self.max_proposal_len = max_proposal_len
        self._vocab_size = vocab_size

    def get_spec_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> SpeculativeProposals:
        """Get speculative proposals given the input batch.

        Sequences which would exceed the max model length are skipped during
        speculation.
        """
        seq_group_metadata_list = execute_model_req.seq_group_metadata_list

        # Split speculative- and non-speculative- sequences.
        (
            proposal_lens,
            nonzero_proposal_len_seqs,
            nonzero_proposal_len_indices,
        ) = self._split_by_proposal_len(seq_group_metadata_list)

        if nonzero_proposal_len_seqs:
            # The sample_len for the worker call is the maximum of the
            # per-sequence k values in the current batch.

            max_k_for_batch = max(proposal_lens)
            hidden_states = execute_model_req.previous_hidden_states
            if hidden_states is not None:
                hidden_states.prune(nonzero_proposal_len_seqs)
            nonzero_execute_model_req = ExecuteModelRequest(
                seq_group_metadata_list=nonzero_proposal_len_seqs,
                num_lookahead_slots=max_k_for_batch,
                previous_hidden_states=hidden_states,
            )
            maybe_sampler_output, transposed = self._worker.sampler_output(
                execute_model_req=nonzero_execute_model_req,
                sample_len=max_k_for_batch,
                seq_ids_with_bonus_token_in_last_step=\
                    seq_ids_with_bonus_token_in_last_step,
            )
            (
                proposal_lens,
                maybe_sampler_output,
                nonzero_proposal_len_indices,
            ) = self._remove_no_proposal_seqs(proposal_lens,
                                              maybe_sampler_output,
                                              nonzero_proposal_len_indices,
                                              transposed)
        else:
            # If no sequences can be speculated, set sampler output to None.
            maybe_sampler_output = None
            transposed = False

        # Combine speculative- and non-speculative sequences into the same
        # representation.
        proposal_tokens, proposal_probs, proposal_lens_tensor = self._merge_outputs(
            batch_size=len(seq_group_metadata_list),
            proposal_len=execute_model_req.num_lookahead_slots,
            maybe_sampler_output=maybe_sampler_output,
            proposal_lens=proposal_lens,
            nonzero_proposal_len_indices=nonzero_proposal_len_indices,
            sampler_transposed=transposed,
        )

        proposals = SpeculativeProposals(proposal_token_ids=proposal_tokens,
                                         proposal_probs=proposal_probs,
                                         proposal_lens=proposal_lens_tensor,
                                         no_proposals=maybe_sampler_output
                                         is None)
        return proposals

    def _split_by_proposal_len(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[List[int], List[SequenceGroupMetadata], List[int]]:
        """Split sequences by two groups:
        1. Sequences with non-zero proposal length.
        2. Sequences with zero proposal length (due to disabled speculation
        or exceed the maximum model length).
        """

        proposal_lens: List[int] = []
        nonzero_proposal_len_seqs: List[SequenceGroupMetadata] = []
        nonzero_proposal_len_indices: List[int] = []
        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            # The speculative decoding for this request has either been disabled
            # (e.g. due to high traffic) or this is a prompt request.
            if (seq_group_metadata.is_prompt
                    or seq_group_metadata.num_speculative_tokens == 0):
                proposal_lens.append(0)
                continue

            seq_data = next(iter(seq_group_metadata.seq_data.values()))
            seq_len = seq_data.get_len()

            # Currently only proposal lens of 0 or the global batch proposal len
            # are supported.
            # If max_proposal_len is defined, then we shall not exceed this
            # quota for nonzero_proposal

            per_sequence_k = seq_group_metadata.num_speculative_tokens
            new_k = 0
            if (self.max_proposal_len is None
                    or seq_len + per_sequence_k < self.max_proposal_len):
                new_k = per_sequence_k
                nonzero_proposal_len_seqs.append(seq_group_metadata)
                nonzero_proposal_len_indices.append(i)
            proposal_lens.append(new_k)
            seq_group_metadata.num_speculative_tokens = new_k

        return (
            proposal_lens,
            nonzero_proposal_len_seqs,
            nonzero_proposal_len_indices,
        )

    @staticmethod
    def _remove_no_proposal_seqs(proposal_lens, maybe_sampler_output,
                                 nonzero_proposal_len_indices, transposed):
        """Remove sequences from nonzero_proposal_len_indices and reset
        their proposal_len to 0 the draft worker does not provide a proposal
        (maybe_sampler_output=None). This can avoid scoring overheads.
        """

        # If maybe_sampler_output is None, then the draft worker did not
        # provide a proposal for any sequence and thus no action needed.
        # Also we do not support transposed maybe_sampler_output for now
        # because it seems not straightforward for draft workers outputting
        # transposed sampler outputs to handle the case of no proposal.
        if maybe_sampler_output is None or transposed:
            return (proposal_lens, maybe_sampler_output,
                    nonzero_proposal_len_indices)

        new_proposal_lens: List[int] = []
        new_nonzero_proposal_len_indices: List[int] = []
        new_maybe_sampler_output: List[SamplerOutput] = []
        nonzero_proposal_len_idx_ptr = 0
        seq_idx = 0
        while seq_idx < len(
                proposal_lens) and nonzero_proposal_len_idx_ptr < len(
                    nonzero_proposal_len_indices):
            if seq_idx < nonzero_proposal_len_indices[
                    nonzero_proposal_len_idx_ptr]:
                # Sequence is not in the original nonzero_proposal_len_indices,
                # meaning that it has a proposal length of 0 before sending to
                # the draft worker.
                assert proposal_lens[seq_idx] == 0
                new_proposal_lens.append(0)
            else:
                # Sequence is in the original nonzero_proposal_len_indices
                if maybe_sampler_output[nonzero_proposal_len_idx_ptr] is None:
                    # but does not have a proposal from the draft worker.
                    new_proposal_lens.append(0)
                else:
                    # and has a proposal from the draft worker. Add it to the
                    # new nonzero proposal list and keep the sampler output.
                    new_proposal_lens.append(proposal_lens[seq_idx])
                    new_nonzero_proposal_len_indices.append(seq_idx)
                    new_maybe_sampler_output.append(
                        maybe_sampler_output[nonzero_proposal_len_idx_ptr])
                nonzero_proposal_len_idx_ptr += 1
            seq_idx += 1

        # The remaining sequences should have proposal length of 0.
        new_proposal_lens.extend(proposal_lens[seq_idx:])

        # We assume sampler_output will not be a list of all Nones.
        # In this case this function should not be called.
        assert new_maybe_sampler_output
        return (new_proposal_lens, new_maybe_sampler_output,
                new_nonzero_proposal_len_indices)

    def _merge_outputs(
        self,
        batch_size: int,
        proposal_len: int,
        maybe_sampler_output: Optional[List[SamplerOutput]],
        proposal_lens: List[int],
        nonzero_proposal_len_indices: List[int],
        sampler_transposed: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """After speculations are produced, merge the speculation results with
        the skipped sequences.
        """
        device = self._device

        # 1) Handle no proposals: treat both None and [] equally
        if not maybe_sampler_output:  # Previously: if maybe_sampler_output is None
            # Return empty proposals with fixed length (proposal_len)
            proposal_tokens = torch.full(
                (batch_size, proposal_len),
                fill_value=-1,
                dtype=torch.long,
                device=device,
            )
            proposal_probs = torch.zeros(
                (batch_size, proposal_len, self._vocab_size),
                dtype=torch.float32,
                device=device,
            )
            proposal_lens_tensor = torch.zeros(
                (len(proposal_lens),),
                dtype=torch.long,
                device=device,
            )
            return proposal_tokens, proposal_probs, proposal_lens_tensor

        # 2) Merge SamplerOutput -> tensors (actual_k, the generated length, is determined here)
        sampler_output = maybe_sampler_output
        proposal_tokens_part, proposal_probs_part, *_ = sampler_output_to_torch(
            sampler_output, sampler_transposed
        )
        # Expected shape: proposal_tokens_part: [N_nonzero, actual_k] or [N_nonzero, actual_k, 1]
        #                 proposal_probs_part : [N_nonzero, actual_k, V]
        # sampler_output_to_torch reshapes to this form using sampler_transposed.

        # Extract step dimension (actual_k)
        # (whether tokens is [N, actual_k] or [N, actual_k, 1], dimension 1 is the step)
        actual_k = int(proposal_tokens_part.shape[1])

        # 3) Final tensors are always padded to proposal_len
        #    (prevents OOB when downstream code indexes with fixed length)
        # Token tensor
        # Preserve trailing dimension (for token id with extra dim) via shape[2:]
        tail_tok = proposal_tokens_part.shape[2:]  # () or (1,)
        entire_proposal_tokens = proposal_tokens_part.new_full(
            (batch_size, proposal_len, *tail_tok),
            fill_value=-1,
        )
        # Probability tensor
        tail_prob = proposal_probs_part.shape[2:]  # (V,)
        entire_proposal_probs = proposal_probs_part.new_zeros(
            (batch_size, proposal_len, *tail_prob),
        )

        # Copy only the first actual_k positions for nonzero rows
        # (Python list indexing works; can convert to LongTensor if needed)
        entire_proposal_tokens[nonzero_proposal_len_indices, :actual_k] = \
            proposal_tokens_part[:, :actual_k]
        entire_proposal_probs[nonzero_proposal_len_indices, :actual_k] = \
            proposal_probs_part[:, :actual_k]

        proposal_tokens, proposal_probs = (
            entire_proposal_tokens,
            entire_proposal_probs,
        )

        # 4) Clamp per-sequence lengths: set each row length to min(original, actual_k)
        #    - rows with zero proposals keep their original 0
        pl = torch.tensor(proposal_lens, dtype=torch.long, device=device)
        if actual_k < proposal_len:
            # Only clamp nonzero rows
            pl_nonzero = torch.minimum(
                pl[torch.as_tensor(nonzero_proposal_len_indices, dtype=torch.long, device=device)],
                torch.full((), actual_k, dtype=torch.long, device=device),
            )
            pl[torch.as_tensor(nonzero_proposal_len_indices, dtype=torch.long, device=device)] = pl_nonzero
        # (if actual_k >= proposal_len, keep original values)

        proposal_lens_tensor = pl

        return proposal_tokens, proposal_probs, proposal_lens_tensor
