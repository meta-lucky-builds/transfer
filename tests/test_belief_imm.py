"""IMM-EKF math against closed-form references (spec 21: Kalman closed-form checks,
mixture moment matching, missing-observation behaviour, PSD under random cases).

References here are written independently with explicit inverses in float64, which
the implementation itself never uses.
"""

import math

import pytest
import torch

from robojepa_ttt.belief.imm import (
    JitterPolicy,
    NumericalBeliefError,
    RepairLog,
    block_nis,
    ekf_predict,
    imm_mix,
    masked_update,
    mixture_moments,
    mixture_predictive,
    posterior_log_weights,
    robust_cholesky,
    sticky_log_transition,
)

torch.set_default_dtype(torch.float64)


@pytest.fixture(autouse=True)
def attention_layout():
    """Override the conftest fixture: these tests do not touch the predictor."""
    yield None


def random_spd(*shape, n, scale=1.0, generator=None):
    a = torch.randn(*shape, n, n, generator=generator)
    return scale * (a @ a.transpose(-1, -2) / n + 0.1 * torch.eye(n))


def reference_kf_update(mean, cov, y, H, b, R):
    """Textbook update with an explicit inverse, one Gaussian, no batch."""
    S = H @ cov @ H.T + torch.diag(R)
    K = cov @ H.T @ torch.linalg.inv(S)
    e = y - H @ mean - b
    post_mean = mean + K @ e
    post_cov = (torch.eye(len(mean)) - K @ H) @ cov
    loglik = -0.5 * (e @ torch.linalg.inv(S) @ e + torch.logdet(S) + len(y) * math.log(2 * math.pi))
    return post_mean, post_cov, loglik


def test_update_matches_textbook_kalman():
    g = torch.Generator().manual_seed(0)
    n, m = 7, 5
    mean = torch.randn(1, 1, n, generator=g)
    cov = random_spd(1, 1, n=n, generator=g)
    H = torch.randn(1, 1, m, n, generator=g)
    b = torch.randn(1, 1, m, generator=g)
    R = torch.rand(1, m, generator=g) + 0.1
    y = torch.randn(1, m, generator=g)
    out = masked_update(mean, cov, y, torch.ones(1, m, dtype=torch.bool), H, b, R)
    ref_mean, ref_cov, ref_ll = reference_kf_update(mean[0, 0], cov[0, 0], y[0], H[0, 0], b[0, 0], R[0])
    assert torch.allclose(out.mean[0, 0], ref_mean, atol=1e-10)
    assert torch.allclose(out.cov[0, 0], ref_cov, atol=1e-10)
    assert torch.allclose(out.log_likelihood[0, 0], ref_ll, atol=1e-10)


def test_masked_update_equals_kalman_on_valid_coordinates_only():
    """Exact masking: different masks per batch element, invalid values are NaN."""
    g = torch.Generator().manual_seed(1)
    B, n, m = 4, 6, 8
    mean = torch.randn(B, 1, n, generator=g)
    cov = random_spd(B, 1, n=n, generator=g)
    H = torch.randn(B, 1, m, n, generator=g)
    b = torch.randn(B, 1, m, generator=g)
    R = torch.rand(B, m, generator=g) + 0.1
    y = torch.randn(B, m, generator=g)
    mask = torch.rand(B, m, generator=g) > 0.4
    mask[0] = True
    mask[1, :] = False
    mask[2, :3] = False
    y_nan = torch.where(mask, y, torch.full_like(y, float("nan")))
    out = masked_update(mean, cov, y_nan, mask, H, b, R)
    for i in range(B):
        keep = mask[i]
        if keep.sum() == 0:
            assert torch.allclose(out.mean[i, 0], mean[i, 0]) and torch.allclose(out.cov[i, 0], cov[i, 0])
            assert out.log_likelihood[i, 0] == 0.0 and out.valid_count[i] == 0
            continue
        ref = reference_kf_update(mean[i, 0], cov[i, 0], y[i, keep], H[i, 0][keep], b[i, 0][keep], R[i][keep])
        assert torch.allclose(out.mean[i, 0], ref[0], atol=1e-10)
        assert torch.allclose(out.cov[i, 0], ref[1], atol=1e-10)
        assert torch.allclose(out.log_likelihood[i, 0], ref[2], atol=1e-10)
    assert torch.isfinite(out.mean).all() and torch.isfinite(out.cov).all()


def test_joseph_update_keeps_covariance_psd_over_random_cases():
    g = torch.Generator().manual_seed(2)
    worst = 0.0
    for trial in range(200):
        n, m = 5, 4
        scale = 10.0 ** torch.empty(1).uniform_(-4, 3, generator=g).item()
        cov = random_spd(1, 1, n=n, scale=scale, generator=g)
        H = torch.randn(1, 1, m, n, generator=g) * 10.0 ** torch.empty(1).uniform_(-2, 2, generator=g).item()
        R = (torch.rand(1, m, generator=g) + 1e-6) * 10.0 ** torch.empty(1).uniform_(-6, 1, generator=g).item()
        out = masked_update(torch.zeros(1, 1, n), cov, torch.randn(1, m, generator=g), torch.ones(1, m, dtype=torch.bool), H, torch.zeros(1, 1, m), R)
        eig = torch.linalg.eigvalsh(out.cov[0, 0])
        worst = min(worst, float(eig.min() / eig.max().clamp_min(1e-300)))
        assert torch.equal(out.cov, out.cov.transpose(-1, -2))
    assert worst > -1e-10


def test_no_valid_coordinates_keeps_prior_and_regime_weights():
    g = torch.Generator().manual_seed(3)
    B, K, n, m = 2, 3, 4, 5
    mean = torch.randn(B, K, n, generator=g)
    cov = random_spd(B, K, n=n, generator=g)
    out = masked_update(mean, cov, torch.randn(B, m, generator=g), torch.zeros(B, m, dtype=torch.bool),
                        torch.randn(B, K, m, n, generator=g), torch.randn(B, K, m, generator=g), torch.ones(B, m))
    assert torch.allclose(out.mean, mean) and torch.allclose(out.cov, cov)
    log_prior = torch.log_softmax(torch.randn(B, K, generator=g), -1)
    assert torch.allclose(posterior_log_weights(log_prior, out.log_likelihood), log_prior)


def test_sticky_transition_rows_are_distributions_and_dt_zero_forbids_switching():
    dt = torch.tensor([0.0, 0.2, 60.0])
    log_T = sticky_log_transition(4, dt, residence_s=600.0)
    assert torch.allclose(log_T.exp().sum(-1), torch.ones(3, 4))
    assert torch.equal(log_T[0].exp(), torch.eye(4))
    eps = 1 - math.exp(-0.2 / 600)
    assert log_T[1, 0, 0].exp().item() == pytest.approx(1 - eps, rel=1e-12)
    assert log_T[1, 0, 1].exp().item() == pytest.approx(eps / 3, rel=1e-9)
    assert torch.equal(sticky_log_transition(1, dt, 600.0), torch.zeros(3, 1, 1))


def reference_mix(pi, means, covs, T):
    K = len(pi)
    prior = T.T @ pi
    out_m, out_c = [], []
    for j in range(K):
        w = T[:, j] * pi / prior[j]
        mj = sum(w[i] * means[i] for i in range(K))
        cj = sum(w[i] * (covs[i] + torch.outer(means[i] - mj, means[i] - mj)) for i in range(K))
        out_m.append(mj)
        out_c.append(cj)
    return prior, torch.stack(out_m), torch.stack(out_c)


def test_mixing_matches_reference_and_keeps_between_mean_spread():
    g = torch.Generator().manual_seed(4)
    K, n = 3, 4
    pi = torch.softmax(torch.randn(K, generator=g), 0)
    means = torch.randn(K, n, generator=g) * 3
    covs = random_spd(K, n=n, generator=g)
    T = torch.softmax(torch.randn(K, K, generator=g), 1)
    log_prior, mm, mc = imm_mix(pi.log()[None], means[None], covs[None], T.log()[None])
    ref = reference_mix(pi, means, covs, T)
    assert torch.allclose(log_prior[0].exp(), ref[0], atol=1e-12)
    assert torch.allclose(mm[0], ref[1], atol=1e-12) and torch.allclose(mc[0], ref[2], atol=1e-12)
    # Two equal covariances, opposite means, uniform mixing: spread must appear.
    eye = torch.eye(2)
    _, _, c = imm_mix(torch.full((1, 2), 0.5).log(), torch.tensor([[[-1.0, 0], [1.0, 0]]]), torch.stack([eye, eye])[None],
                      torch.full((1, 2, 2), 0.5).log())
    assert c[0, 0, 0, 0].item() == pytest.approx(2.0)
    assert (torch.linalg.eigvalsh(mc[0]) > 0).all()


def test_unreachable_regime_does_not_produce_nan():
    log_pi = torch.tensor([[0.0, -float("inf")]])
    means = torch.randn(1, 2, 3)
    covs = torch.eye(3).expand(1, 2, 3, 3).clone()
    log_prior, mm, mc = imm_mix(log_pi, means, covs, torch.log(torch.eye(2))[None])
    assert log_prior[0, 1] == -float("inf")
    assert torch.isfinite(mm).all() and torch.isfinite(mc).all()
    assert torch.allclose(mm[0, 1], means[0, 1])


def test_mixture_moments_match_samples():
    g = torch.Generator().manual_seed(5)
    log_pi = torch.log(torch.tensor([[0.3, 0.7]]))
    means = torch.tensor([[[0.0, 1.0], [2.0, -1.0]]])
    covs = random_spd(1, 2, n=2, generator=g)
    mean, cov = mixture_moments(log_pi, means, covs)
    count = 400_000
    regime = (torch.rand(count, generator=g) < 0.7).long()
    chol = torch.linalg.cholesky(covs[0])
    samples = means[0][regime] + (chol[regime] @ torch.randn(count, 2, 1, generator=g)).squeeze(-1)
    assert torch.allclose(mean[0], samples.mean(0), atol=0.01)
    assert torch.allclose(cov[0], torch.cov(samples.T), atol=0.02)


def test_exact_two_regime_posterior_without_switching():
    """With no switching, a bank of linear Kalman filters with Bayesian weights is the
    exact posterior. IMM must reproduce it over many steps, including missing data."""
    g = torch.Generator().manual_seed(6)
    n, m, steps = 3, 2, 25
    A = [torch.eye(n) * 0.95 + 0.05 * torch.randn(n, n, generator=g) for _ in range(2)]
    Q = [torch.rand(n, generator=g) * 0.05 + 0.01 for _ in range(2)]
    H = torch.randn(m, n, generator=g)
    R = torch.rand(m, generator=g) * 0.1 + 0.05
    truth = torch.zeros(n)
    ys, masks = [], []
    for t in range(steps):
        truth = A[1] @ truth + Q[1].sqrt() * torch.randn(n, generator=g)
        ys.append(H @ truth + R.sqrt() * torch.randn(m, generator=g))
        masks.append(torch.rand(m, generator=g) > 0.3)

    log_pi = torch.log(torch.tensor([[0.5, 0.5]]))
    mean = torch.zeros(1, 2, n)
    cov = torch.eye(n).expand(1, 2, n, n).clone()
    ref = [(torch.zeros(n), torch.eye(n), math.log(0.5)) for _ in range(2)]
    for t in range(steps):
        log_prior, mean, cov = imm_mix(log_pi, mean, cov, sticky_log_transition(2, torch.zeros(1), 600.0))
        mats = torch.stack(A)[None]
        mean, cov, _ = ekf_predict(mean, cov, lambda x: (mats @ x[..., None]).squeeze(-1), torch.stack(Q)[None])
        out = masked_update(mean, cov, ys[t][None], masks[t][None], H.expand(1, 2, m, n), torch.zeros(1, 2, m), R[None])
        log_pi = posterior_log_weights(log_prior, out.log_likelihood)
        mean, cov = out.mean, out.cov
        new_ref = []
        for k in range(2):
            mk, ck, lk = ref[k]
            mk, ck = A[k] @ mk, A[k] @ ck @ A[k].T + torch.diag(Q[k])
            keep = masks[t]
            if keep.any():
                mk, ck, ll = reference_kf_update(mk, ck, ys[t][keep], H[keep], torch.zeros(int(keep.sum())), R[keep])
                lk = lk + ll
            new_ref.append((mk, ck, lk))
        ref = new_ref
        weights = torch.softmax(torch.tensor([r[2] for r in ref]), 0)
        for k in range(2):
            assert torch.allclose(mean[0, k], ref[k][0], atol=1e-9)
            assert torch.allclose(cov[0, k], ref[k][1], atol=1e-9)
        assert torch.allclose(log_pi[0].exp(), weights, atol=1e-9)
    assert log_pi[0, 1].exp() > 0.5, "the regime that generated the data should be favoured"


def test_ekf_predict_linear_and_nonlinear_jacobians():
    g = torch.Generator().manual_seed(7)
    n = 4
    A = torch.randn(n, n, generator=g)
    mean = torch.randn(2, 1, n, generator=g)
    cov = random_spd(2, 1, n=n, generator=g)
    q = torch.rand(2, 1, n, generator=g)
    pred, pcov, jac = ekf_predict(mean, cov, lambda x: x @ A.T, q)
    assert torch.allclose(jac, A.expand(2, 1, n, n))
    assert torch.allclose(pcov, A @ cov @ A.T + torch.diag_embed(q))

    W = torch.randn(n, n, generator=g)
    f = lambda x: torch.tanh(x @ W.T) + 0.5 * x
    with torch.no_grad():
        _, _, jac = ekf_predict(mean, cov, f, q)
    eps = 1e-6
    for i in range(n):
        step = torch.zeros(n)
        step[i] = eps
        numeric = (f(mean + step) - f(mean - step)) / (2 * eps)
        assert torch.allclose(jac[..., i], numeric, atol=1e-8)


def test_second_order_gradient_through_the_jacobian():
    """Differentiable filtering needs d(P^-)/d(theta) through the Jacobian (spec 9.3)."""
    g = torch.Generator().manual_seed(8)
    n = 3
    mean = torch.randn(1, 1, n, generator=g)
    cov = random_spd(1, 1, n=n, generator=g)

    def loss(theta):
        f = lambda x: torch.tanh(x * theta)
        _, pcov, _ = ekf_predict(mean, cov, f, torch.full((1, 1, n), 0.1))
        return torch.logdet(pcov[0, 0])

    theta = torch.randn(n, generator=g, requires_grad=True)
    assert torch.autograd.gradcheck(loss, (theta,))

    def surrogate(theta):
        f = lambda x: torch.tanh(x * theta)
        _, pcov, _ = ekf_predict(mean, cov, f, torch.full((1, 1, n), 0.1), differentiable_jacobian=False)
        return torch.logdet(pcov[0, 0])

    assert surrogate(theta).requires_grad is False or surrogate(theta).grad_fn is None


def test_robust_cholesky_repairs_and_logs_then_refuses():
    log = RepairLog()
    singular = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    factor = robust_cholesky(singular, JitterPolicy(1e-7, 1e-3), log, "test")
    assert log.count == 1 and 0 < log.max_jitter <= 1e-3
    assert torch.allclose(factor @ factor.transpose(-1, -2), singular, atol=1e-3)
    with pytest.raises(NumericalBeliefError):
        robust_cholesky(torch.tensor([[[1.0, 0.0], [0.0, -1.0]]]), JitterPolicy(1e-7, 1e-3), log, "test")


def test_block_nis_uses_the_marginal_block():
    g = torch.Generator().manual_seed(9)
    m = 5
    cov = random_spd(1, n=m, generator=g)
    mean = torch.randn(1, m, generator=g)
    y = torch.randn(1, m, generator=g)
    mask = torch.tensor([[True, True, False, True, True]])
    stats = block_nis(y, mask, mean, cov, {"a": (0, 3), "b": (3, 5), "none": (2, 3)})
    keep = [0, 1]
    e = (y - mean)[0, keep]
    expected = e @ torch.linalg.inv(cov[0][keep][:, keep]) @ e
    assert stats["a"][0].item() == pytest.approx(expected.item())
    assert stats["a"][1].item() == 2 and stats["none"][1].item() == 0 and stats["none"][0].item() == 0


def test_mixture_predictive_matches_mode_bank():
    g = torch.Generator().manual_seed(10)
    B, K, n, m = 1, 2, 3, 2
    out = masked_update(torch.randn(B, K, n, generator=g), random_spd(B, K, n=n, generator=g), torch.randn(B, m, generator=g),
                        torch.ones(B, m, dtype=torch.bool), torch.randn(B, K, m, n, generator=g), torch.zeros(B, K, m), torch.ones(B, m))
    log_prior = torch.log(torch.tensor([[0.25, 0.75]]))
    mean, cov = mixture_predictive(log_prior, out)
    w = log_prior.exp()[0]
    ref_mean = (w[:, None] * out.predicted_observation[0]).sum(0)
    ref_cov = sum(w[k] * (out.innovation_cov[0, k] + torch.outer(out.predicted_observation[0, k] - ref_mean, out.predicted_observation[0, k] - ref_mean)) for k in range(K))
    assert torch.allclose(mean[0], ref_mean) and torch.allclose(cov[0], ref_cov)


def test_batch_elements_and_regimes_never_mix():
    """Regression: a broadcast once aligned batch with regimes in the Joseph term."""
    g = torch.Generator().manual_seed(11)
    B, K, n, m = 3, 2, 4, 5
    mean = torch.randn(B, K, n, generator=g)
    cov = random_spd(B, K, n=n, generator=g)
    y = torch.randn(B, m, generator=g)
    mask = torch.rand(B, m, generator=g) > 0.3
    H = torch.randn(B, K, m, n, generator=g)
    b = torch.randn(B, K, m, generator=g)
    R = torch.rand(B, m, generator=g) + 0.1
    whole = masked_update(mean, cov, y, mask, H, b, R)
    for i in range(B):
        for k in range(K):
            one = masked_update(mean[i : i + 1, k : k + 1], cov[i : i + 1, k : k + 1], y[i : i + 1], mask[i : i + 1],
                                H[i : i + 1, k : k + 1], b[i : i + 1, k : k + 1], R[i : i + 1])
            assert torch.allclose(whole.mean[i, k], one.mean[0, 0], atol=1e-12)
            assert torch.allclose(whole.cov[i, k], one.cov[0, 0], atol=1e-12)
            assert torch.allclose(whole.log_likelihood[i, k], one.log_likelihood[0, 0], atol=1e-12)
