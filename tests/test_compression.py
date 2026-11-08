import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov
 
from core import (
    CompreSSM,
    CompressionConfig,
    DiagonalDiscreteSSM,
    ReducedDiagonalSSM,
    ReductionSchedule,
    compress_diagonal_numpy,
    lru_arrays_from_diagonal,
)
 
 
def random_diagonal_system(n=16, m=2, p=3, seed=0):
    rng = np.random.default_rng(seed)
    lam = rng.uniform(0.5, 0.97, n) * np.exp(1j * rng.uniform(0, 2 * np.pi, n))
    strength = np.geomspace(1.0, 1e-4, n)
    B = (rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))) * strength[:, None]
    C = (rng.standard_normal((p, n)) + 1j * rng.standard_normal((p, n))) * strength[None, :]
    D = rng.standard_normal(p)
    return DiagonalDiscreteSSM.from_numpy(lam=lam, B=B, C=C, D=D)
 
 
def frequency_response(A, B, C, omega):
    z = np.exp(1j * omega)
    return C @ np.linalg.solve(z * np.eye(A.shape[0]) - A, B)
 
 
def max_response_error(A1, B1, C1, A2, B2, C2, points=800):
    omegas = np.linspace(0, 2 * np.pi, points, endpoint=False)
    return max(
        np.linalg.norm(frequency_response(A1, B1, C1, w) - frequency_response(A2, B2, C2, w), 2)
        for w in omegas
    )
 
 
def test_diagonal_gramians_match_scipy():
    s = random_diagonal_system()
    P_d, Q_d = CompreSSM.calculate_gramians_diagonal(s.lam, B=s.B, C=s.C)
    P_s, Q_s = CompreSSM.calculate_gramians(np.diag(s.lam), B=s.B, C=s.C)
    assert np.allclose(P_d, P_s, atol=1e-9)
    assert np.allclose(Q_d, Q_s, atol=1e-9)
 
 
def test_gramian_residuals():
    s = random_diagonal_system()
    A = np.diag(s.lam)
    P, Q = CompreSSM.calculate_gramians_diagonal(s.lam, B=s.B, C=s.C)
    assert np.allclose(A @ P @ A.conj().T - P + s.B @ s.B.conj().T, 0, atol=1e-9)
    assert np.allclose(A.conj().T @ Q @ A - Q + s.C.conj().T @ s.C, 0, atol=1e-9)
 
 
def test_projection_is_left_inverse_of_transform():
    s = random_diagonal_system()
    P, Q = CompreSSM.calculate_gramians_diagonal(s.lam, B=s.B, C=s.C)
    bal = CompreSSM.balance(P, Q=Q)
    r = 6
    T, d = CompreSSM.calculate_transform_matrix(bal.Lp, r=r, sigma=bal.sigma, Vh=bal.Vh)
    S = CompreSSM.calculate_projection_matrix(bal.Lq, r=r, diag_entries=d, U=bal.U)
    assert np.allclose(S @ T, np.eye(r), atol=1e-8)
 
 
def test_hsvs_match_eigenvalues_of_PQ():
    s = random_diagonal_system(n=8)
    P, Q = CompreSSM.calculate_gramians_diagonal(s.lam, B=s.B, C=s.C)
    bal = CompreSSM.balance(P, Q=Q)
    eig = np.sort(np.sqrt(np.abs(np.linalg.eigvals(P @ Q))))[::-1]
    assert np.allclose(bal.sigma[:4], eig[:4], rtol=1e-6)
 
 
def test_full_rank_truncation_preserves_response():
    s = random_diagonal_system(n=8)
    A = np.diag(s.lam)
    P, Q = CompreSSM.calculate_gramians_diagonal(s.lam, B=s.B, C=s.C)
    bal = CompreSSM.balance(P, Q=Q)
    Ar, Br, Cr = CompreSSM.truncate(A, s.B, s.C, balancing=bal, r=8)
    assert max_response_error(A, s.B, s.C, Ar, Br, Cr) < 1e-6
 
 
@pytest.mark.parametrize("r", [4, 8, 12])
def test_error_bound_holds(r):
    s = random_diagonal_system()
    A = np.diag(s.lam)
    P, Q = CompreSSM.calculate_gramians_diagonal(s.lam, B=s.B, C=s.C)
    bal = CompreSSM.balance(P, Q=Q)
    Ar, Br, Cr = CompreSSM.truncate(A, s.B, s.C, balancing=bal, r=r)
    bound = 2 * np.sum(bal.sigma[r:])
    assert max_response_error(A, s.B, s.C, Ar, Br, Cr) <= bound + 1e-9
 
 
def test_diagonalize_reduced_preserves_response():
    s = random_diagonal_system()
    A = np.diag(s.lam)
    P, Q = CompreSSM.calculate_gramians_diagonal(s.lam, B=s.B, C=s.C)
    bal = CompreSSM.balance(P, Q=Q)
    Ar, Br, Cr = CompreSSM.truncate(A, s.B, s.C, balancing=bal, r=8)
    lam_r, Bd, Cd = CompreSSM.diagonalize_reduced(Ar, Br, Cr)
    assert max_response_error(Ar, Br, Cr, np.diag(lam_r), Bd, Cd) < 1e-6
 
 
def test_compress_diagonal_numpy_end_to_end():
    s = random_diagonal_system()
    reduced = compress_diagonal_numpy(s, config=CompressionConfig(tol=0.01, max_ratio=0.95))
    assert reduced is not None
    assert reduced.r < s.n
    assert reduced.lam.shape == (reduced.r,)
    assert np.max(np.abs(reduced.lam)) < 1
    error = max_response_error(
        np.diag(s.lam), s.B, s.C, np.diag(reduced.lam), reduced.B, reduced.C
    )
    assert error <= reduced.error_bound + 1e-9
    assert np.array_equal(reduced.D, s.D)
 
 
def test_compress_returns_none_when_not_worth_it():
    s = random_diagonal_system()
    assert compress_diagonal_numpy(s, config=CompressionConfig(tol=0.0, max_ratio=0.2)) is None
 
 
def test_gate_is_strict():
    assert not CompreSSM.should_compress(19, 20, max_ratio=0.95)
    assert CompreSSM.should_compress(18, 20, max_ratio=0.95)
    assert not CompreSSM.should_compress(0, 20, max_ratio=0.95)
 
 
def test_rank_selection_known_values():
    sigma = np.array([5.0, 2.0, 0.5, 0.05, 0.001])
    assert CompreSSM.rank_selection(sigma, threshold=0.99) == 3
    assert CompreSSM.rank_selection(sigma, threshold=1.0) == 5
    assert CompreSSM.rank_selection(sigma, threshold=0.5) == 1
 
 
def test_lru_parameter_round_trip():
    s = random_diagonal_system()
    reduced = ReducedDiagonalSSM(lam=s.lam, B=s.B, C=s.C, D=s.D, r=s.n, error_bound=0.0)
    arrays = lru_arrays_from_diagonal(reduced)
    back = DiagonalDiscreteSSM.from_lru_numpy(D=s.D, **arrays)
    assert np.allclose(back.lam, s.lam, atol=1e-6)
    assert np.allclose(back.B, s.B, atol=1e-6)
    assert np.allclose(back.C, s.C, atol=1e-6)
 
 
def test_cholesky_jitter_handles_near_singular():
    v = np.array([[1.0], [2.0], [3.0]])
    M = v @ v.T
    L = CompreSSM.cholesky_with_jitter(M, name="M")
    assert np.allclose(L @ L.T, M, atol=1e-6)
 
 
def test_schedule_matches_repo_rule():
    sch = ReductionSchedule(red_start=0, red_end=20000, red_interval=5000, eval_interval=1000)
    fired = [s for s in range(0, 25000) if sch.fires_at(s)]
    assert fired == [4999, 9999, 14999, 19999]
 
