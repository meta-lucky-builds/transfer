"""Interacting-multiple-model extended Kalman filter math (spec 7.1 to 7.3, eq. 13 to 20).

Pure tensor functions with no neural parts, so every step can be checked against
closed-form Kalman results. Shapes use B for batch, K for active regimes, n for the
compact state (64) and m for the observation (134).

    belief      log_pi [B, K], mean [B, K, n], cov [B, K, n, n]
    log_T       [B, K, K], log probability of moving from regime i (dim 1) to j (dim 2)

Two numerical rules from the spec are enforced here:

* No explicit inverse. Gains and likelihoods use a Cholesky factor of the innovation
  covariance, and a failing factorisation gets the smallest jitter that works, from
  JitterPolicy.initial up to JitterPolicy.maximum, counted in a RepairLog. Beyond the
  maximum the filter raises instead of inventing a belief (spec 18: never silently zero
  a covariance).
* Covariances are updated in Joseph form and symmetrised.

Masked observations are exact, not approximated with a huge variance. For an invalid
coordinate the observation row and innovation are zeroed and its noise set to 1. The
innovation covariance then splits into a valid block and an identity block, so the
gain has zero columns for invalid coordinates, the identity block adds nothing to the
log-determinant, and the dimension count uses valid coordinates only. The result
equals a Kalman filter run on the valid coordinates alone, which a test checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

LOG_2PI = math.log(2.0 * math.pi)


class NumericalBeliefError(RuntimeError):
    """A covariance could not be repaired within the configured jitter bound."""


@dataclass(frozen=True)
class JitterPolicy:
    initial: float = 1e-7
    maximum: float = 1e-3


@dataclass
class RepairLog:
    """Every covariance repair, as the spec requires them to be logged."""

    count: int = 0
    max_jitter: float = 0.0
    events: list = field(default_factory=list)

    def record(self, where: str, jitter: float, failures: int) -> None:
        self.count += failures
        self.max_jitter = max(self.max_jitter, jitter)
        if len(self.events) < 1000:
            self.events.append((where, jitter, failures))


def symmetrize(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.transpose(-1, -2))


def robust_cholesky(
    matrix: torch.Tensor,
    policy: JitterPolicy,
    log: Optional[RepairLog] = None,
    where: str = "",
    diagonal_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Lower Cholesky factor, adding the smallest sufficient jitter to failing matrices.

    diagonal_mask [..., m] limits jitter to selected diagonal entries, so the identity
    block of masked coordinates stays exactly the identity.
    """
    factor, info = torch.linalg.cholesky_ex(matrix)
    if not bool((info > 0).any()):
        return factor
    size = matrix.shape[-1]
    eye = torch.eye(size, dtype=matrix.dtype, device=matrix.device)
    if diagonal_mask is not None:
        eye = torch.diag_embed(diagonal_mask.to(matrix.dtype)).expand_as(matrix)
    failing = (info > 0).to(matrix.dtype)[..., None, None]
    jitter = policy.initial
    while jitter <= policy.maximum * (1 + 1e-9):
        repaired = matrix + failing * jitter * eye
        factor, info = torch.linalg.cholesky_ex(repaired)
        if not bool((info > 0).any()):
            if log is not None:
                log.record(where, jitter, int(failing.sum()))
            return factor
        jitter *= 10.0
    raise NumericalBeliefError(f"{where}: covariance not positive definite even with jitter {policy.maximum}")


# ---------------------------------------------------------------- regimes


def sticky_log_transition(regimes: int, dt_s: torch.Tensor, residence_s: float) -> torch.Tensor:
    """Eq. 14 in log space. dt_s [B] -> [B, K, K].

    epsilon = 1 - exp(-dt / residence), so log(1 - epsilon) = -dt / residence exactly,
    which avoids cancellation for small dt. For K = 1 the matrix is [1].
    """
    batch = dt_s.shape[0]
    if regimes == 1:
        return dt_s.new_zeros(batch, 1, 1)
    rate = dt_s / residence_s
    stay = -rate
    # log(epsilon / (K - 1)); -inf when dt = 0, which correctly forbids switching.
    move = torch.log(-torch.expm1(-rate)) - math.log(regimes - 1)
    eye = torch.eye(regimes, dtype=torch.bool, device=dt_s.device)
    return torch.where(eye, stay[:, None, None], move[:, None, None])


def imm_mix(log_pi: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor, log_transition: torch.Tensor):
    """Regime prediction and moment-matched mixing, spec 7.1 and eq. 15.

    Returns (log prior regime weights [B, K], mixed means, mixed covariances). The
    between-mean outer product is kept: dropping it makes the filter falsely certain
    when the regimes disagree. A regime that no other regime can reach keeps its own
    moments and a prior weight of -inf instead of producing NaN.
    """
    if log_pi.shape[-1] == 1:
        return log_pi + log_transition[..., 0, :], mean, cov
    joint = log_transition + log_pi[..., :, None]  # [B, i, j]
    log_prior = torch.logsumexp(joint, dim=-2)  # [B, j]
    reachable = torch.isfinite(log_prior)
    safe_prior = torch.where(reachable, log_prior, torch.zeros_like(log_prior))
    weights = torch.exp(joint - safe_prior[..., None, :])
    eye = torch.eye(log_pi.shape[-1], dtype=weights.dtype, device=weights.device)
    weights = torch.where(reachable[..., None, :], weights, eye)
    mixed_mean = torch.einsum("bij,bin->bjn", weights, mean)
    diff = mean[:, :, None, :] - mixed_mean[:, None, :, :]  # [B, i, j, n]
    spread = diff[..., :, None] * diff[..., None, :]
    mixed_cov = torch.einsum("bij,bijnk->bjnk", weights, cov[:, :, None] + spread)
    return log_prior, mixed_mean, symmetrize(mixed_cov)


def posterior_log_weights(log_prior: torch.Tensor, log_likelihood: torch.Tensor) -> torch.Tensor:
    """Eq. 19. With no valid coordinates the likelihoods are all zero and this is the prior."""
    joint = log_prior + log_likelihood
    return joint - torch.logsumexp(joint, dim=-1, keepdim=True)


def mixture_moments(log_pi: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor):
    """Eq. 20: one summary mean and covariance for APIs that need a single Gaussian.

    Planning should sample the mixture instead of using this average state.
    """
    weights = torch.exp(log_pi)
    total = torch.einsum("bk,bkn->bn", weights, mean)
    diff = mean - total[:, None]
    spread = diff[..., :, None] * diff[..., None, :]
    return total, symmetrize(torch.einsum("bk,bknm->bnm", weights, cov + spread))


# ------------------------------------------------------------- prediction


def batched_jacobian(function: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, create_graph: bool):
    """Jacobian of a function applied independently to each [..., n] row of x.

    Because row r of the output depends only on row r of the input, the Jacobian of the
    summed output holds every per-row Jacobian. Returns [..., n_out, n_in].
    """
    leading = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])

    def summed(rows):
        return function(rows.reshape(*leading, -1)).reshape(rows.shape[0], -1).sum(0)

    jac = torch.autograd.functional.jacobian(summed, flat, create_graph=create_graph, vectorize=True)
    return jac.permute(1, 0, 2).reshape(*leading, jac.shape[0], x.shape[-1])


def ekf_predict(
    mean: torch.Tensor,
    cov: torch.Tensor,
    transition: Callable[[torch.Tensor], torch.Tensor],
    process_variance: torch.Tensor,
    differentiable_jacobian: bool = True,
):
    """Eq. 16: first-order propagation through the compact transition.

    transition maps [B, K, n] to [B, K, n] and must treat rows independently.
    process_variance [B, K, n] is the diagonal of Q for this interval. With
    differentiable_jacobian=False the Jacobian is a constant for the optimiser: the
    surrogate-gradient ablation of spec 9.3, which must be named as such when used.
    Returns (predicted mean, predicted covariance, Jacobian).
    """
    predicted = transition(mean)
    create_graph = differentiable_jacobian and torch.is_grad_enabled()
    # autograd.functional.jacobian enables grad internally, so this also works under
    # no_grad, where it returns a detached Jacobian.
    jac = batched_jacobian(transition, mean, create_graph=create_graph)
    if not create_graph:
        jac = jac.detach()
    predicted_cov = jac @ cov @ jac.transpose(-1, -2) + torch.diag_embed(process_variance)
    return predicted, symmetrize(predicted_cov), jac


# ----------------------------------------------------------- measurement


@dataclass
class UpdateResult:
    mean: torch.Tensor  # posterior [B, K, n]
    cov: torch.Tensor  # posterior [B, K, n, n]
    log_likelihood: torch.Tensor  # [B, K], over valid coordinates only
    predicted_observation: torch.Tensor  # [B, K, m], H mu^- + b on every coordinate
    innovation: torch.Tensor  # [B, K, m], zero on invalid coordinates
    innovation_cov: torch.Tensor  # [B, K, m, m], identity block on invalid coordinates
    valid_count: torch.Tensor  # [B]


def masked_update(
    mean: torch.Tensor,
    cov: torch.Tensor,
    observation: torch.Tensor,
    mask: torch.Tensor,
    obs_matrix: torch.Tensor,
    obs_bias: torch.Tensor,
    obs_variance: torch.Tensor,
    policy: JitterPolicy = JitterPolicy(),
    log: Optional[RepairLog] = None,
) -> UpdateResult:
    """Eq. 17 to 19 with exact masking. See the module docstring for why it is exact.

    observation [B, m], mask [B, m] (True = valid), obs_matrix [B, K, m, n],
    obs_bias [B, K, m], obs_variance [B, m] diagonal of R, positive.
    Invalid observation values may be anything finite or not; they never enter.
    """
    dtype = mean.dtype
    valid = mask.to(dtype)[:, None, :]  # [B, 1, m]
    predicted = (obs_matrix @ mean[..., None]).squeeze(-1) + obs_bias
    clean = torch.where(mask, observation, torch.zeros_like(observation))  # NaN-safe
    innovation = (clean[:, None, :] - predicted) * valid
    masked_matrix = obs_matrix * valid[..., None]
    noise = torch.where(mask, obs_variance, torch.ones_like(obs_variance))[:, None, :]  # [B, 1, m]
    cross = cov @ masked_matrix.transpose(-1, -2)  # P H^T, [B, K, n, m]
    innovation_cov = symmetrize(masked_matrix @ cross + torch.diag_embed(noise.expand_as(predicted)))
    factor = robust_cholesky(
        innovation_cov, policy, log, "innovation covariance", diagonal_mask=mask[:, None, :].expand_as(predicted)
    )
    gain = torch.cholesky_solve(cross.transpose(-1, -2), factor).transpose(-1, -2)  # [B, K, n, m]
    posterior_mean = mean + (gain @ innovation[..., None]).squeeze(-1)
    eye = torch.eye(mean.shape[-1], dtype=dtype, device=mean.device)
    keep = eye - gain @ masked_matrix
    # noise is [B, 1, m]; the extra axis lines it up with gain [B, K, n, m] column-wise.
    posterior_cov = keep @ cov @ keep.transpose(-1, -2) + (gain * noise[:, :, None, :]) @ gain.transpose(-1, -2)
    whitened = torch.linalg.solve_triangular(factor, innovation[..., None], upper=False).squeeze(-1)
    log_det = 2.0 * torch.log(torch.diagonal(factor, dim1=-2, dim2=-1)).sum(-1)
    count = mask.sum(-1)
    log_likelihood = -0.5 * (whitened.pow(2).sum(-1) + log_det + count[:, None].to(dtype) * LOG_2PI)
    return UpdateResult(
        posterior_mean, symmetrize(posterior_cov), log_likelihood, predicted, innovation, innovation_cov, count
    )


def mixture_predictive(log_prior: torch.Tensor, update: UpdateResult):
    """Moment-matched predictive distribution of the observation over regimes.

    Returns (mean [B, m], covariance [B, m, m]). The mean covers every coordinate, so
    for a missing coordinate it is the prior predicted value, which the recurrent
    history uses as its stand-in. The covariance is meaningful on the valid block only;
    block_nis replaces the rest by an identity block.
    """
    weights = torch.exp(log_prior)
    mean = torch.einsum("bk,bkm->bm", weights, update.predicted_observation)
    diff = update.predicted_observation - mean[:, None]
    spread = diff[..., :, None] * diff[..., None, :]
    cov = torch.einsum("bk,bkij->bij", weights, update.innovation_cov + spread)
    return mean, symmetrize(cov)


def block_nis(
    observation: torch.Tensor,
    mask: torch.Tensor,
    predictive_mean: torch.Tensor,
    predictive_cov: torch.Tensor,
    blocks: dict[str, tuple[int, int]],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Normalised innovation squared per block from the marginal predictive of that block.

    Returns name -> (NIS [B], valid dimension count [B]). A block with no valid
    coordinates has NIS 0 and count 0; callers must skip it rather than read it as a
    perfect prediction. Under a correct Gaussian model E[NIS] equals the count, which
    is a diagnostic, not proof (spec 7.5).
    """
    out = {}
    clean = torch.where(mask, observation, torch.zeros_like(observation))
    for name, (start, end) in blocks.items():
        block_mask = mask[:, start:end]
        e = (clean[:, start:end] - predictive_mean[:, start:end]) * block_mask.to(clean.dtype)
        sub = predictive_cov[:, start:end, start:end]
        # Invalid rows: replace by identity so they decouple, as in masked_update.
        keep = block_mask[:, :, None] & block_mask[:, None, :]
        eye = torch.eye(end - start, dtype=sub.dtype, device=sub.device).expand_as(sub)
        sub = torch.where(keep, sub, eye)
        factor = torch.linalg.cholesky(symmetrize(sub))
        z = torch.linalg.solve_triangular(factor, e[..., None], upper=False).squeeze(-1)
        out[name] = (z.pow(2).sum(-1), block_mask.sum(-1))
    return out
