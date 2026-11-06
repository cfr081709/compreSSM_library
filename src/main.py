import torch
import numpy as np
from scipy.linalg import (
    solve_discrete_lyapunov,
    cholesky,
    svd
)

def torch_collect_training_matrices(model, dt=1.0):
    state_dict = model.state_dict()

    for key in ('ssm.A_log', 'ssm.B', 'ssm.C', 'ssm.D'):
        if key not in state_dict:
            raise KeyError(f"Training matrix {key} not found")

    a = -np.exp(state_dict['ssm.A_log'].detach().cpu().numpy())
    n = a.shape[0]

    B = state_dict['ssm.B'].detach().cpu().numpy().reshape(n, -1)
    C = state_dict['ssm.C'].detach().cpu().numpy().reshape(-1, n)
    D = state_dict['ssm.D'].detach().cpu().numpy()

    a_bar = np.exp(dt * a)
    B = ((a_bar - 1.0) / a)[:, None] * B
    A = np.diag(a_bar)

    if np.max(np.abs(a_bar)) >= 1:
        raise ValueError("A is not stable (spectral radius >= 1)")

    return A, B, C, D

def calculate_gramians(A, B, C):
    P = solve_discrete_lyapunov(A, B @ B.conj().T)

    Q = solve_discrete_lyapunov(A.conj().T, C.conj().T @ C)

    P = (P + P.conj().T) / 2

    Q = (Q + Q.conj().T) / 2

    return P, Q

def calculate_sqrt(P, Q):
    try:
        Lp = cholesky(P, lower=True)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError("Gramian not positive definite on matrix P (controllability Gramian)") from e
    try:
        Lq = cholesky(Q, lower=True)
    except np.linalg.LinAlgError as e:
        raise np.linalg.LinAlgError("Gramian not positive definite on matrix Q (observability Gramian)") from e
    return Lp, Lq

def HSVs(Lp, Lq):
    M = Lq.conj().T @ Lp

    U, sigma, Vh = svd(M)

    return U, sigma, Vh

def rank_selection(sigma, t=.99):
    total = np.sum(sigma)

    if total <= 0:
        return len(sigma)

    cum = np.cumsum(sigma) / total

    r = int(np.searchsorted(cum, t)) + 1

    return min(max(r, 1), len(sigma))

def should_compress(r, n, max_ratio):
    return r / n <= max_ratio

def calculate_transform_matrix(Lp, r, sigma, Vh):
    Vr = Vh[:r].conj().T
    diag_entries = np.maximum(sigma[:r], 1e-12) ** -0.5
    T = Lp @ Vr @ np.diag(diag_entries)

    return T, diag_entries

def calculate_projection_matrix(Lq, r, diag_entries, U):
    Lq = Lq.conj().T
    diag = np.diag(diag_entries)
    Ur = U[:, :r].conj().T
    S = diag @ Ur @ Lq

    return S

def truncate_A(A, S, T):
    Ar = S @ A @ T

    return Ar

def truncate_B(S, B):
    Br = S @ B

    return Br

def truncate_C(C, T):
    Cr = C @ T

    return Cr