from __future__ import annotations  

import math     
from collections import deque   
from typing import Optional     

import numpy as np  
import torch        
from vllm.logger import init_logger

logger = init_logger(__name__)

# ──────────────────────── helper: WVIR & DSL ─────────────────────────
def _weighted_variance(kl_divergences: np.ndarray, weights: np.ndarray) -> float:      
    """Calculates weighted variance of kl divergence over the speculative iterations

    Args: 
        kl_divergences : np.ndarray
            kl divergneces for a given range
        shape : (maxlen=activation_threshold)
        
        weights : np.ndarray
            Weights on kl divergence. Designed to attenuate the magnitude of kl divergence as it is far from current instance   
        shape : (maxlen=activation_threshold)

    
    Returns:
        weighted_variance : float
    """
    weighted_mean_kl_divergence = (weights * kl_divergences).sum() / weights.sum()
    return ((weights * (kl_divergences - weighted_mean_kl_divergence) ** 2).sum() / weights.sum())

def compute_kl_divergence( 
    draft_probs: torch.Tensor,  
    target_probs: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor: 
    """Compute kl divergence of probability distributions between target model and draft model 
    
    Args: 
        draft_probs: torch.Tensor
            The probability distribution over token ids given context according to the draft model.
        shape = [batch_size, num_speculative_tokens, vocab_size]

        target_probs: torch.Tensor
            The probability distribution over token ids given context according to the target model.
        shape = [batch_size, num_speculative_tokens + 1, vocab_size]

    
    Returns:
        kl_divergence : torch.Tensor
    """
    draft_probs = draft_probs.clamp(min=eps)    
    target_probs = target_probs.clamp(min=eps)  
    return (target_probs *
            (target_probs.log() - draft_probs.log())).sum(dim=-1)

def _weighted_intensity_ratio(kl_divergences: np.ndarray, weight: float = 0.9) -> float:
    """Calcualte the ratio of weighted variances with difference window sizes

    Args:
        kl_divergences : np.ndarray
            kl divergneces for a given range
        shape : (maxlen=activation_threshold)

        weights : np.ndarray
            Weights on kl divergence. Designed to attenuate the magnitude of kl divergence as it is far from current instance   
        shape : (maxlen=activation_threshold)

    Returns:
        weighted_variance_intensity_ratio : float
    """

    num_kl_divergences = len(kl_divergences) 
    if num_kl_divergences < 2:
        return 0.0
    short_window_size = max(1, num_kl_divergences // 3)
    weights_short_window = weight ** np.arange(short_window_size - 1, -1, -1) 
    weights_long_window = weight ** np.arange(num_kl_divergences - 1, -1, -1)
    weighted_variance_short_window = _weighted_variance(kl_divergences[-short_window_size:], weights_short_window)
    weighted_variance_long_window = _weighted_variance(kl_divergences[-num_kl_divergences:], weights_long_window)
    weighted_variance_intensity_ratio = 20.0 if weighted_variance_long_window < 1e-6 else weighted_variance_short_window / weighted_variance_long_window
    return weighted_variance_intensity_ratio

def coordinate_batch_k(
    sgm_list: list,
    schedulers: dict[str, DynamicScheduler],
    kld_histories: dict[str, deque],
    step_klds: dict[str, list[float]],
    props: dict[str, int],
    acc_for_rate: dict[str, int],
    sl_cap_activation: bool,
) -> None:
    """
    Adjust k-values across the entire batch. All calculations use integers and ceiling.
    SL_cap policy is only applied to KLScheduler instances.
    """
    
    # --- Step 1: Collect integer k-values from each KLScheduler ---
    initial_k_list = []
    for sgm in sgm_list:
        rid = sgm.request_id
        sched = schedulers.get(rid)

        if not isinstance(sched, KLScheduler) or not sched.is_warmup_bounds_calculated:
            continue

        kld_hist = kld_histories.get(rid)
        k_prop = props.get(rid, 0)
        if not kld_hist or k_prop == 0:
            continue

        wvir = _weighted_intensity_ratio(np.array(kld_hist))
        kld_list_for_step = step_klds.get(rid, [])
        last_kld = sum(kld_list_for_step) / k_prop if kld_list_for_step else 0.0
        
        # Call _dsl_step directly to get final integer k-value
        initial_k = _dsl_step(
            wvir,
            last_kld,
            sched.dynamic_k_min,
            sched.dynamic_k_max,
            sgm.sampling_params.temperature
        )
        initial_k_list.append(initial_k)

    # --- Step 2: Calculate SL_cap (integer-based + ceiling) ---
    sl_cap = None
    if initial_k_list:
        # Average the integer list, then apply ceiling to get final SL_cap (integer)
        if sl_cap_activation:
            sl_cap = math.ceil(np.mean(initial_k_list))
            logger.info(f"[SL_Cap] Batch SL_cap calculated: {sl_cap} from values: {initial_k_list}")

    # --- Step 3: Final k-value adjustment ---
    # Call each scheduler's adjust method
    for sgm in sgm_list:
        rid = sgm.request_id
        sched = schedulers.get(rid)
        if not sched:
            continue

        # Prepare common parameters for adjust
        kld_hist = kld_histories.get(rid)
        cumulative_proposed = len(kld_hist) if kld_hist else 0
        kld_list_for_step = step_klds.get(rid, [])
        k_prop = props.get(rid, 0)
        mean_kld_in_step = sum(kld_list_for_step) / k_prop if k_prop > 0 else 0.0

        # Pass sl_cap only to KLScheduler
        if isinstance(sched, KLScheduler):
            sched.adjust(
                cumulative_proposed_tokens=cumulative_proposed,
                accepted_len_in_step=acc_for_rate.get(rid, 0),
                temperature=sgm.sampling_params.temperature,
                kld_history=kld_hist,
                kld_values_in_step=kld_list_for_step,
                accepted_klds_mean_in_step=mean_kld_in_step,
                sl_cap=sl_cap  # Pass sl_cap only to KLScheduler
            )
        else:
            # For other schedulers, call without sl_cap
            sched.adjust(
                proposed_len_in_step=k_prop,  # Argument used by other schedulers
                accepted_len_in_step=acc_for_rate.get(rid, 0),
                accepted_klds_mean_in_step=mean_kld_in_step,
                kld_values_in_step=kld_list_for_step,
                kld_history=kld_hist,
            )

# KLScheduler-original ver
def _dsl_step(weighted_variance_intensity_ratio: float, mean_kl_divergence_last_step: float, k_min: int, k_max: int, temperature: float) -> int:
    """ Calculates new speculation length (new k) for the next speculative step
    
    Args:
        weighted_variance_intensity_ratio
            The ratio of weighted variances with difference window sizes
        shape : float

        mean_kl_divergence_last_step
           Mean kl divergence in the previous speculative step
        shape : float

        k_min, k_max            
            Minimum and maximum speculation length for proposal
        shape : int, int

    Returns:
        _dsl_step
        shape : int
    """
    
    α = 2.0
    eps = 1e-8
    if abs(temperature) < eps:
        kl_divergence_adjust = mean_kl_divergence_last_step
    else:
        kl_divergence_adjust = mean_kl_divergence_last_step / (temperature + eps)

    Scale_Factor = math.exp(α * kl_divergence_adjust) - 1
    if weighted_variance_intensity_ratio * Scale_Factor <= 1:
        sl_hat = (k_max - k_min) * (1 - weighted_variance_intensity_ratio * Scale_Factor) + k_min
        k_new = max(k_min, math.ceil(sl_hat))
        return k_new
    return k_min

# KLScheduler-log version
# def _dsl_step(weighted_variance_intensity_ratio: float, mean_kl_divergence_last_step: float, k_min: int, k_max: int, temperature: float) -> int:
#     # --- Changed SF calculation to use log function ---
#     c = 1.0  # Constant to control KLD influence (tunable)
#     eps = 1e-8
#     kl_divergence_adjust = mean_kl_divergence_last_step / (temperature + eps) if temperature != 0.0 else mean_kl_divergence_last_step
    
#     # Use log() instead of exp()
#     Scale_Factor = c * math.log(1 + kl_divergence_adjust)
#     # ------------------------------------

#     if weighted_variance_intensity_ratio * Scale_Factor <= 1:
#         sl_hat = (k_max - k_min) * (1 - weighted_variance_intensity_ratio * Scale_Factor) + k_min
#         k_new = max(k_min, math.ceil(sl_hat))
#         return k_new
#     return k_min
# ────────────────────────────────────────────────────────────────────

class DynamicScheduler:
    """Abstract base class for dynamic schedulers"""
    def __init__(self, init_k: int, k_min: int, k_max: int):
        self.k = init_k
        self.k_min = k_min
        self.k_max = k_max
        
    def adjust(self, **kwargs) -> int:
        raise NotImplementedError

class KLScheduler(DynamicScheduler):
    def __init__(
        self,
        init_k: int,
        k_min: int,
        k_max: int,
        target_accept: float = 0.7,
        dsl_activation_tokens: int = 30,
        request_id: Optional[str] = None,
    ):
        super().__init__(init_k, k_min, k_max)
        self.target_accept = target_accept
        self.warmup_tokens = dsl_activation_tokens
        self.initial_k_min, self.initial_k_max = k_min, k_max
        self.dynamic_k_min, self.dynamic_k_max = None, None
        self.is_warmup_bounds_calculated = False
        self.warmup_kld_samples: list[float] = []   
        self.warmup_accepted_token_samples: list[int] = []  
        self.request_id = request_id

    def _calculate_dynamic_bounds(self):
        if not self.warmup_accepted_token_samples:
            self.dynamic_k_max = self.initial_k_max
            self.dynamic_k_min = self.initial_k_min
        else:
            max_num_accepted_tokens = np.max(self.warmup_accepted_token_samples)
            avg_kld = np.mean(self.warmup_kld_samples) if self.warmup_kld_samples else 0.0      
            max_kld = np.max(self.warmup_kld_samples) if self.warmup_kld_samples else 1.0
            if max_kld == 0: max_kld = 1.0
            
            k_max_float = max_num_accepted_tokens * (1 + 0.5 * (avg_kld / max_kld))
            calculated_k_max = round(k_max_float)       
            self.dynamic_k_max = min(self.initial_k_max, max(self.initial_k_min, calculated_k_max))     
            k_min_float = self.dynamic_k_max * 0.1      
            calculated_k_min = math.ceil(k_min_float)
            self.dynamic_k_min = min(self.dynamic_k_max, max(self.initial_k_min, calculated_k_min))

        
        logger.info(f"Request {self.request_id}: Dynamic bounds calculated: k_min={self.dynamic_k_min}, k_max={self.dynamic_k_max}")
        self.warmup_kld_samples.clear()
        self.warmup_accepted_token_samples.clear()

    def adjust(
        self,
        cumulative_proposed_tokens: int,
        accepted_len_in_step: int,
        temperature: float,
        kld_values_in_step: Optional[list[float]] = None,
        kld_history: Optional[deque[float]] = None,
        accepted_klds_mean_in_step: Optional[float] = None,
        sl_cap: Optional[float] = None,
        **kwargs
    ) -> int:
        
        # Phase 1: Handle warmup logic
        self._handle_warmup_phase(
            cumulative_proposed_tokens,
            accepted_len_in_step,
            kld_values_in_step
        )
        
        # Phase 2: Calculate dynamic bounds then determine k-value
        if self._is_dynamic_bounds_ready(cumulative_proposed_tokens, kld_history):
            k_calculated = self._compute_k_value(
                kld_history,
                accepted_klds_mean_in_step,
                temperature
            )
            k_final = self._apply_sl_cap(k_calculated, sl_cap)
            self.k = k_final
            return self.k
        
        # Fallback: during warmup or when kld_history is unavailable
        return self._get_fallback_k()

    def _handle_warmup_phase(
        self,
        cumulative_tokens: int,
        accepted_len: int,
        kld_values: Optional[list[float]]
    ) -> None:
        """Collect samples and adjust initial k during warmup"""
        if self.is_warmup_bounds_calculated:
            return
            
        if cumulative_tokens < self.warmup_tokens:
            self._collect_warmup_samples(accepted_len, kld_values)
            self._adjust_k_during_warmup(accepted_len)

    def _collect_warmup_samples(
        self,
        accepted_len: int,
        kld_values: Optional[list[float]]
    ) -> None:
        """Collect token and KLD samples during warmup"""
        self.warmup_accepted_token_samples.append(accepted_len)
        if kld_values:
            self.warmup_kld_samples.extend(kld_values)

    def _adjust_k_during_warmup(self, accepted_len: int) -> None:
        """Adjust k-value during warmup"""
        if accepted_len >= self.k:
            self.k = min(self.k * 2 if self.k > 0 else 1, self.initial_k_max)
        else:
            self.k = max(self.k, self.initial_k_min)

    def _is_dynamic_bounds_ready(
        self,
        cumulative_tokens: int,
        kld_history: Optional[deque[float]]
    ) -> bool:
        """Check if dynamic bounds calculation is ready"""
        if not self.is_warmup_bounds_calculated:
            if cumulative_tokens >= self.warmup_tokens:
                self._calculate_dynamic_bounds()
                self.is_warmup_bounds_calculated = True
        return self.is_warmup_bounds_calculated and kld_history is not None

    def _compute_k_value(
        self,
        kld_history: deque[float],
        last_kld: Optional[float],
        temperature: float
    ) -> int:
        """Calculate k-value using weighted KLD analysis"""
        wvir = _weighted_intensity_ratio(np.array(kld_history))
        current_kld = last_kld if last_kld is not None else 0.0
        return _dsl_step(
            wvir,
            current_kld,
            self.dynamic_k_min,
            self.dynamic_k_max,
            temperature
        )

    def _apply_sl_cap(self, k_calculated: int, sl_cap: Optional[float]) -> int:
        """Apply SL cap and log if applied"""
        if sl_cap is None or k_calculated <= sl_cap:
            return k_calculated
            
        logger.info(
            f"[SL_Cap] Request {self.request_id}: k capped. "
            f"Initial: {k_calculated}, Cap: {sl_cap}, Final: {sl_cap}"
        )
        return int(sl_cap)

    def _get_fallback_k(self) -> int:
        """Return default k-value for fallback situations"""
        return self.dynamic_k_min if self.dynamic_k_min is not None else self.initial_k_min


class KLDHorizonScheduler(DynamicScheduler):
    """
    Improved KLD-Horizon v3 algorithm
    - Adaptive threshold calculation
    - Confidence-based horizon detection  
    - Progressive rate limiting
    - Multi-window analysis
    """
    def __init__(
        self,
        init_k: int,
        k_min: int,
        k_max: int,
        # --- Improved hyperparameters ---
        ema_alpha: float = 0.4,  # 0.3 -> 0.4 (more responsive)
        kld_history_size: int = 100,  # 150 -> 100 (focus on recent data)
        base_kld_quantile: float = 0.7,  # 0.6 -> 0.7 (more conservative)
        adaptive_quantile_range: float = 0.2,  # Dynamic quantile adjustment range
        rate_limit_delta: int = 2,  # 5 -> 2 (more stable)
        progressive_rate_limit: bool = True,  # Progressive rate limiting
        cooldown_steps: int = 3,  # 2 -> 3 (longer cooldown)
        major_failure_threshold: float = 0.25,  # 0.2 -> 0.25 (less sensitive)
        confidence_weight: float = 0.3,  # Horizon confidence weight
        # -----------------------------------------
        request_id: Optional[str] = None,
        **kwargs
    ):
        super().__init__(init_k, k_min, k_max)
        self.request_id = request_id
        self.warmup_tokens = 0
        
        # Improved hyperparameters
        self.ema_alpha = ema_alpha
        self.base_kld_quantile = base_kld_quantile
        self.adaptive_quantile_range = adaptive_quantile_range
        self.rate_limit_delta = rate_limit_delta
        self.progressive_rate_limit = progressive_rate_limit
        self.cooldown_steps = cooldown_steps
        self.major_failure_threshold = major_failure_threshold
        self.confidence_weight = confidence_weight
        
        # Algorithm state variables
        self.k_ema = float(init_k)
        self.kld_history_pool = deque(maxlen=kld_history_size)
        self.cooldown_counter = 0
        self.recent_acceptance_rates = deque(maxlen=10)  # Track recent acceptance rates
        self.step_count = 0

    def _calculate_adaptive_threshold(self) -> float:
        """Calculate adaptive threshold adjusted based on recent performance"""
        if len(self.kld_history_pool) <= 5:
            return 1.5  # 5.0 -> 1.5 (more realistic initial value)
            
        # Adjust quantile based on recent acceptance rate
        recent_ar = np.mean(list(self.recent_acceptance_rates)) if self.recent_acceptance_rates else 0.5
        
        # Higher acceptance rate -> more lenient threshold (lower quantile)
        # Lower acceptance rate -> stricter threshold (higher quantile)
        quantile_adjustment = (0.5 - recent_ar) * self.adaptive_quantile_range
        adaptive_quantile = max(0.5, min(0.9, self.base_kld_quantile + quantile_adjustment))
        
        return np.quantile(list(self.kld_history_pool), adaptive_quantile)

    def _detect_horizon_with_confidence(self, kld_values: List[float], threshold: float) -> tuple[int, float]:
        """Detect horizon considering confidence"""
        if not kld_values:
            return 0, 0.0
            
        k_horizon = 0
        confidence_scores = []
        
        for i, kld in enumerate(kld_values):
            # Measure how comfortably KLD passes the threshold
            margin = threshold - kld
            confidence = max(0.0, min(1.0, margin / threshold))  # Normalize to 0~1
            confidence_scores.append(confidence)
            
            if kld > threshold:
                break
            k_horizon = i + 1
        
        # Calculate total confidence (average + consistency bonus)
        if confidence_scores:
            avg_confidence = np.mean(confidence_scores)
            consistency_bonus = 1.0 - np.std(confidence_scores) if len(confidence_scores) > 1 else 1.0
            total_confidence = avg_confidence * consistency_bonus
        else:
            total_confidence = 0.0
            
        return k_horizon, total_confidence

    def _progressive_rate_limiting(self, k_prev: int, k_target: int) -> int:
        """Progressive rate limiting - apply large changes over multiple steps"""
        if not self.progressive_rate_limit:
            delta = min(self.rate_limit_delta, abs(k_target - k_prev))
            return k_prev + delta if k_target > k_prev else k_prev - delta
            
        # Dynamic rate limit based on change magnitude
        change_magnitude = abs(k_target - k_prev)
        
        if change_magnitude <= 1:
            effective_delta = 1
        elif change_magnitude <= 3:
            effective_delta = min(2, self.rate_limit_delta)
        else:
            effective_delta = max(1, self.rate_limit_delta // 2)  # More conservative for large changes
            
        if k_target > k_prev:
            return min(k_prev + effective_delta, k_target)
        else:
            return max(k_prev - effective_delta, k_target)

    def adjust(
        self,
        kld_values_in_step: List[float],
        proposed_len_in_step: int,
        accepted_len_in_step: int,
        **kwargs
    ) -> int:
        self.step_count += 1
        self._update_acceptance_rate(proposed_len_in_step, accepted_len_in_step)
        
        is_cooldown_active = self._handle_cooldown_state()
        kld_threshold = self._calculate_adaptive_threshold()
        
        k_horizon, confidence = self._detect_horizon(
            kld_values_in_step, 
            kld_threshold
        )
        self._update_kld_history(kld_values_in_step)
        
        k_horizon = self._apply_horizon_bonus(
            k_horizon, 
            proposed_len_in_step, 
            confidence
        )
        self._update_ema(k_horizon, confidence)
        
        k_candidate = self._calculate_k_candidate()
        k_candidate = self._apply_cooldown_constraint(k_candidate, is_cooldown_active)
        self.k = self._apply_bounds(k_candidate)
        
        self._detect_major_failure(proposed_len_in_step, accepted_len_in_step)
        return self.k

    def _update_acceptance_rate(self, proposed: int, accepted: int) -> None:
        """Track and store acceptance rate"""
        current_ar = accepted / proposed if proposed > 0 else 0.0
        self.recent_acceptance_rates.append(current_ar)

    def _handle_cooldown_state(self) -> bool:
        """Manage cooldown state"""
        is_active = self.cooldown_counter > 0
        if is_active:
            self.cooldown_counter -= 1
        return is_active

    def _detect_horizon(
        self, 
        kld_values: List[float], 
        threshold: float
    ) -> Tuple[int, float]:
        if not kld_values:
            return 0, 0.0
            
        horizon = len(kld_values)
        for idx, kld_val in enumerate(kld_values):
            if kld_val < threshold:
                horizon = idx
                break
                
        under_threshold_count = sum(1 for kld in kld_values if kld < threshold)
        confidence = under_threshold_count / len(kld_values)
        return horizon, confidence

    def _update_kld_history(self, kld_values: List[float]) -> None:
        """Update KLD history"""
        if kld_values:
            self.kld_history_pool.extend(kld_values)

    def _apply_horizon_bonus(
        self, 
        horizon: int, 
        proposed: int, 
        confidence: float
    ) -> int:
        """Apply horizon bonus"""
        if horizon == proposed and proposed < self.k_max and confidence > 0.7:
            return horizon + 1
        return horizon

    def _update_ema(self, horizon: int, confidence: float) -> None:
        """Update confidence-weighted EMA"""
        alpha = self.ema_alpha * (0.5 + 0.5 * confidence)
        self.k_ema = (alpha * horizon) + (1 - alpha) * self.k_ema

    def _calculate_k_candidate(self) -> int:
        """Calculate k candidate value"""
        k_prev = self.k
        k_target = round(self.k_ema)
        return self._progressive_rate_limiting(k_prev, k_target)

    def _apply_cooldown_constraint(
        self, 
        candidate: int, 
        is_cooldown: bool
    ) -> int:
        """Apply cooldown constraint"""
        return min(candidate, self.k) if is_cooldown else candidate

    def _apply_bounds(self, candidate: int) -> int:
        """Apply final bounds"""
        return max(self.k_min, min(self.k_max, candidate))

    def _detect_major_failure(self, proposed: int, accepted: int) -> None:
        """Detect and handle major failure"""
        if proposed > 0 and (accepted / proposed) < self.major_failure_threshold:
            self.cooldown_counter = self.cooldown_steps
            self.k = max(self.k_min, self.k - 1)


class GuardrailScheduler(DynamicScheduler):
    """
    Scheduler that adjusts k based on acceptance rate, using KLD as a 'guardrail' to improve stability.
    """
    def __init__(
        self,
        init_k: int,
        k_min: int,
        k_max: int,
        # --- Hyperparameters for Guardrail algorithm ---
        target_acceptance_rate: float = 0.7,
        history_len: int = 5,              # History length for smoothing acceptance rate
        kld_veto_threshold: float = 3.5,   # KLD threshold to veto k-value increase
        cooldown_steps: int = 3,           # Steps to block k increase after major failure
        major_failure_threshold: float = 0.2, # Acceptance rate threshold for major failure
        # -----------------------------------------
        request_id: Optional[str] = None,
        **kwargs  # Arguments for compatibility with other schedulers
    ):
        super().__init__(init_k, k_min, k_max)
        self.request_id = request_id
        self.warmup_tokens = 0  # Added for compatibility with KLScheduler

        # Algorithm hyperparameters
        self.target_acceptance_rate = target_acceptance_rate
        self.kld_veto_threshold = kld_veto_threshold
        self.cooldown_steps = cooldown_steps
        self.major_failure_threshold = major_failure_threshold
        
        # Algorithm state variables
        self.acceptance_rate_history = deque(maxlen=history_len)
        self.cooldown_counter = 0

    def adjust(
        self,
        proposed_len_in_step: int,
        accepted_len_in_step: int,
        accepted_klds_mean_in_step: float,
        **kwargs
    ) -> int:

        self._update_acceptance_rate_history(proposed_len_in_step, accepted_len_in_step)
        self._update_cooldown_counter()
        
        if self._is_major_failure(proposed_len_in_step, accepted_len_in_step):
            return self._handle_major_failure()
        
        return self._adjust_k_based_on_ar(accepted_klds_mean_in_step)

    def _update_acceptance_rate_history(self, proposed: int, accepted: int) -> None:
        current_ar = accepted / proposed if proposed > 0 else 0.0
        self.acceptance_rate_history.append(current_ar)
        self.avg_ar = np.mean(self.acceptance_rate_history)

    def _update_cooldown_counter(self) -> None:
        self.cooldown_counter = max(0, self.cooldown_counter - 1)

    def _is_major_failure(self, proposed: int, accepted: int) -> bool:
        current_ar = accepted / proposed if proposed > 0 else 0.0
        return current_ar < self.major_failure_threshold

    def _handle_major_failure(self) -> int:
        self.cooldown_counter = self.cooldown_steps
        self.k = max(self.k_min, self.k - 1)
        return self.k

    def _adjust_k_based_on_ar(self, kld: float) -> int:
        if self.avg_ar > self.target_acceptance_rate:
            return self._try_increase_k(kld)
        elif self.avg_ar < self.target_acceptance_rate:
            return self._decrease_k()
        return self.k

    def _try_increase_k(self, kld: float) -> int:
        if self.cooldown_counter == 0 and kld < self.kld_veto_threshold:
            self.k = min(self.k_max, self.k + 1)
        return self.k

    def _decrease_k(self) -> int:
        self.k = max(self.k_min, self.k - 1)
        return self.k


class HybridScheduler(DynamicScheduler):
    """
    Hybrid scheduler that uses static-k by default, temporarily increasing k only under specific conditions.
    """
    def __init__(
        self,
        init_k: int,
        k_min: int,
        k_max: int,
        # --- Hyperparameters for Hybrid algorithm ---
        default_k: int = 4,                   # Default k-value to use
        boost_k: int = 8,                     # K-value to use in boost mode
        boost_lookback_steps: int = 3,        # Consecutive successful steps to check for boost activation
        boost_ar_threshold: float = 0.9,      # Acceptance rate threshold to consider as 'success'
        boost_kld_threshold: float = 1.5,     # KLD threshold to consider as 'success'
        # -----------------------------------------
        request_id: Optional[str] = None,
        **kwargs  # Arguments for compatibility with other schedulers
    ):
        # Set init_k to default_k when calling super().__init__
        super().__init__(default_k, k_min, k_max)
        self.request_id = request_id
        self.warmup_tokens = 0  # Added for compatibility with KLScheduler

        # Algorithm hyperparameters
        self.default_k = default_k
        self.boost_k = boost_k
        self.boost_lookback_steps = boost_lookback_steps
        self.boost_ar_threshold = boost_ar_threshold
        self.boost_kld_threshold = boost_kld_threshold
        
        # Algorithm state variables
        self.recent_performance = deque(maxlen=self.boost_lookback_steps)
        self.is_boost_mode = False

    def adjust(
        self,
        proposed_len_in_step: int,
        accepted_len_in_step: int,
        accepted_klds_mean_in_step: float,
        **kwargs
    ) -> int:
        
        current_ar = self._calculate_current_ar(proposed_len_in_step, accepted_len_in_step)
        
        if self._should_exit_boost_mode(current_ar):
            self._exit_boost_mode()
        
        self._update_performance_history(current_ar, accepted_klds_mean_in_step)
        
        if self._should_enter_boost_mode():
            self._enter_boost_mode()
            
        return self._set_k_based_on_mode()

    def _calculate_current_ar(self, proposed_len_in_step: int, accepted_len_in_step: int) -> float:
        return accepted_len_in_step / proposed_len_in_step if proposed_len_in_step > 0 else 0.0

    def _should_exit_boost_mode(self, current_ar: float) -> bool:
        return self.is_boost_mode and current_ar < self.boost_ar_threshold

    def _exit_boost_mode(self) -> None:
        self.is_boost_mode = False
        self.recent_performance.clear()
        self.k = self.default_k

    def _update_performance_history(self, current_ar: float, accepted_klds_mean_in_step: float) -> None:
        self.recent_performance.append((current_ar, accepted_klds_mean_in_step))

    def _should_enter_boost_mode(self) -> bool:
        if self.is_boost_mode or len(self.recent_performance) != self.boost_lookback_steps:
            return False
            
        for ar, kld in self.recent_performance:
            if ar < self.boost_ar_threshold or kld > self.boost_kld_threshold:
                return False
        return True

    def _enter_boost_mode(self) -> None:
        self.is_boost_mode = True

    def _set_k_based_on_mode(self) -> int:
        self.k = self.boost_k if self.is_boost_mode else self.default_k
        return self.k

class AdaEDLScheduler(DynamicScheduler):
    def __init__(self,
                 init_k: int,
                 k_min: int,
                 k_max: int,
                 # --- New hyperparameters ---
                 beta_1: float = 0.5,
                 beta_2: float = 0.9,
                 alpha: float = 0.9,
                 epsilon: float = 0.01,
                 gamma_risk: float = 0.1,  # Gamma value for Risk Score calculation
                 initial_lambda: float = 0.5,  # Initial Lambda value
                 **kwargs):

        super().__init__(init_k=init_k, k_min=k_min, k_max=k_max)

        # Store new hyperparameters
        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma_risk = float(gamma_risk)

        # Variables to store sequence state
        self.cumulative_ar = 0.0  # Cumulative Acceptance Rate (AR)
        self.lambda_val = float(initial_lambda)  # Current Lambda value
        self.request_id = None

    def adjust(self,
            *,
            proposed_len_in_step: int,
            accepted_len_in_step: int,
            max_k: int,
            **_kwargs) -> None:
        """
        Instead of adjusting k, adjust the Lambda threshold to be used in the next step.
        """
        ar_n = self._calculate_ar(proposed_len_in_step, accepted_len_in_step)
        self._update_cumulative_ar(ar_n)
        lambda_prime_n = self._compute_lambda_prime(ar_n, accepted_len_in_step, max_k)
        self._update_lambda_val(lambda_prime_n)
        
    def _calculate_ar(self, proposed: int, accepted: int) -> float:
        """Calculate current step's Acceptance Rate (AR_n)"""
        return accepted / proposed if proposed > 0 else 0.0

    def _update_cumulative_ar(self, ar_n: float) -> None:
        """Update cumulative AR (Exponential Moving Average)"""
        self.cumulative_ar = self.beta_1 * self.cumulative_ar + (1.0 - self.beta_1) * ar_n

    def _compute_lambda_prime(self, ar_n: float, accepted: int, max_k: int) -> float:
        """Calculate Lambda'_n (includes conditional logic)"""
        if self.cumulative_ar < self.alpha:
            return self.lambda_val + self.epsilon
        elif self.cumulative_ar > self.alpha and accepted == max_k:
            return self.lambda_val - self.epsilon
        return self.lambda_val

    def _update_lambda_val(self, lambda_prime_n: float) -> None:
        """Calculate and update Lambda_{n+1} (EMA)"""
        self.lambda_val = self.beta_2 * self.lambda_val + (1.0 - self.beta_2) * lambda_prime_n
