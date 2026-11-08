import argparse
import importlib.util
import sys
import types
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from core import (
    CompreSSM,
    CompressionConfig,
    DiagonalDiscreteSSM,
    DiscreteSSM,
    compress_diagonal_numpy,
    compress_numpy,
)

TOL = 0.05
N_STATES = 16
N_CHANNELS = 3
DEFAULT_REPO_DIRS = [
    "external/compressm/compressm/reduction",
    "compressm/compressm/reduction",
]


def make_lru_params(n=N_STATES, h=N_CHANNELS, seed=0):
    rng = np.random.default_rng(seed)
    r_min, r_max = 0.4, 0.99
    u1, u2 = rng.uniform(size=n), rng.uniform(size=n)
    nu_log = np.log(-0.5 * np.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
    theta_log = np.log(2 * np.pi * u2)
    modulus = np.exp(-np.exp(nu_log))
    gamma_log = np.log(np.sqrt(1.0 - modulus ** 2))
    strength = np.geomspace(1.0, 1e-3, n)[:, None]
    return {
        "nu_log": nu_log,
        "theta_log": theta_log,
        "gamma_log": gamma_log,
        "B_re": rng.standard_normal((n, h)) / np.sqrt(2 * h) * strength,
        "B_im": rng.standard_normal((n, h)) / np.sqrt(2 * h) * strength,
        "C_re": rng.standard_normal((h, n)) / np.sqrt(n),
        "C_im": rng.standard_normal((h, n)) / np.sqrt(n),
        "D": rng.standard_normal(h),
    }


def frequency_error(sys_a, sys_b, points=400):
    (A1, B1, C1), (A2, B2, C2) = sys_a, sys_b
    worst = 0.0
    for w in np.linspace(0, 2 * np.pi, points, endpoint=False):
        z = np.exp(1j * w)
        g1 = C1 @ np.linalg.solve(z * np.eye(A1.shape[0]) - A1, B1)
        g2 = C2 @ np.linalg.solve(z * np.eye(A2.shape[0]) - A2, B2)
        worst = max(worst, np.linalg.norm(g1 - g2, 2))
    return worst


def run_mine(params, tol=TOL):
    system = DiagonalDiscreteSSM.from_lru_numpy(**params)
    P, Q = CompreSSM.calculate_gramians_diagonal(system.lam, B=system.B, C=system.C)
    sigma = CompreSSM.balance(P, Q=Q).sigma
    r = CompreSSM.rank_selection(sigma, threshold=1.0 - tol)
    reduced = compress_diagonal_numpy(system, config=CompressionConfig(tol=tol, max_ratio=1.0))
    return {
        "sigma": sigma,
        "r": r,
        "lam_r": None if reduced is None else reduced.lam,
        "reduced": None if reduced is None else (np.diag(reduced.lam), reduced.B, reduced.C),
        "full": (np.diag(system.lam), system.B, system.C),
    }


def run_dense_reference(params, tol=TOL):
    s = DiagonalDiscreteSSM.from_lru_numpy(**params)
    A = np.diag(s.lam)
    P, Q = CompreSSM.calculate_gramians(A, B=s.B, C=s.C)
    sigma = CompreSSM.balance(P, Q=Q).sigma
    r = CompreSSM.rank_selection(sigma, threshold=1.0 - tol)
    reduced = compress_numpy(
        DiscreteSSM.from_numpy(A=A, B=s.B, C=s.C, D=s.D),
        config=CompressionConfig(tol=tol, max_ratio=1.0),
    )
    return {
        "sigma": sigma,
        "r": r,
        "lam_r": None if reduced is None else np.linalg.eigvals(reduced.A),
        "reduced": None if reduced is None else (reduced.A, reduced.B, reduced.C),
        "full": (A, s.B, s.C),
    }


def load_repo_reduction(reduction_dir):
    """Load the repo's hsv.py and balanced_truncation.py without importing the
    repo's package __init__ (which needs JAX and equinox)."""
    package = types.ModuleType("compressm")
    package.__path__ = []
    subpackage = types.ModuleType("compressm.reduction")
    subpackage.__path__ = []
    sys.modules["compressm"] = package
    sys.modules["compressm.reduction"] = subpackage
    for name in ("hsv", "balanced_truncation"):
        path = Path(reduction_dir) / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"compressm.reduction.{name}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        setattr(subpackage, name, module)
    return sys.modules["compressm.reduction.hsv"], sys.modules["compressm.reduction.balanced_truncation"]


def run_repo(params, repo, *, tol=TOL, method="sqrtm"):
    """The repo's own reduction code on the same weights. The glue after
    lti_to_lru mirrors LRULayer.reduce in lru.py (divide B by gamma, store log gamma)."""
    hsv, balanced_truncation = repo
    system = DiagonalDiscreteSSM.from_lru_numpy(**params)

    P, Q, g = hsv.hankel_singular_values_diagonal(system.lam, system.B, system.C)
    P, Q = np.asarray(P), np.asarray(Q)
    sigma = np.sort(np.nan_to_num(np.asarray(g).real.ravel()))[::-1]

    rank = hsv.reduction_analysis(g, hankel_tol=tol)["recommended_ranks"]["threshold"]
    result = {"sigma": sigma, "r": rank, "lam_r": None, "reduced": None,
              "full": (np.diag(system.lam), system.B, system.C)}
    if rank >= system.n:
        return result

    A_red, B_red, C_red = balanced_truncation.reduce_discrete_lti(
        system.lam, system.B, system.C, P, Q, rank=rank, method=method
    )
    nu_log, theta_log, B_re, B_im, C_re, C_im, gammas = balanced_truncation.lti_to_lru(
        A_red, B_red, C_red
    )
    gammas = np.asarray(gammas).real
    back = DiagonalDiscreteSSM.from_lru_numpy(
        nu_log=np.real(np.asarray(nu_log)),
        theta_log=np.real(np.asarray(theta_log)),
        B_re=np.real(np.asarray(B_re) / gammas[:, None]),
        B_im=np.real(np.asarray(B_im) / gammas[:, None]),
        gamma_log=np.real(np.log(gammas)),
        C_re=np.real(np.asarray(C_re)),
        C_im=np.real(np.asarray(C_im)),
        D=params["D"],
    )
    result["lam_r"] = back.lam
    result["reduced"] = (np.diag(back.lam), back.B, back.C)
    return result


def compare(mine, other, label):
    print(f"\n=== mine vs {label} ===")
    s1 = np.sort(mine["sigma"])[::-1]
    s2 = np.sort(other["sigma"])[::-1]
    k = min(len(s1), len(s2))
    body = s1[:k] > 1e-6 * s1[0]
    rel = np.abs(s1[:k][body] - s2[:k][body]) / s1[:k][body]
    print(f"HSVs: {body.sum()} values above 1e-6 of the largest, max relative difference {rel.max():.2e}")
    print(f"rank: mine {mine['r']}, other {other['r']}")
    if mine["lam_r"] is not None and other["lam_r"] is not None and len(mine["lam_r"]) == len(other["lam_r"]):
        cost = np.abs(mine["lam_r"][:, None] - other["lam_r"][None, :])
        i, j = linear_sum_assignment(cost)
        print(f"reduced eigenvalues (matched by nearest): max mismatch {cost[i, j].max():.2e}")
    if mine["reduced"] is not None and other["reduced"] is not None:
        err = frequency_error(mine["reduced"], other["reduced"])
        print(f"frequency response of the two reduced models: max difference {err:.2e}")
    if mine["reduced"] is not None:
        err = frequency_error(mine["reduced"], mine["full"])
        print(f"(mine reduced vs mine full, for scale: {err:.2e})")
    if other["reduced"] is not None:
        err = frequency_error(other["reduced"], other["full"])
        print(f"(other reduced vs full, for scale: {err:.2e})")


def find_reduction_dir(explicit):
    candidates = [explicit] if explicit else DEFAULT_REPO_DIRS
    for candidate in candidates:
        if candidate and (Path(candidate) / "hsv.py").exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-reduction", help="folder containing the repo's hsv.py and balanced_truncation.py")
    parser.add_argument("--tol", type=float, default=TOL)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    params = make_lru_params(seed=args.seed)
    mine = run_mine(params, tol=args.tol)

    reduction_dir = find_reduction_dir(args.repo_reduction)
    if reduction_dir is None:
        print("Repo reduction folder not found, so this is a dry run against the dense SciPy path.")
        compare(mine, run_dense_reference(params, tol=args.tol), "dense SciPy path (dry run)")
        return

    repo = load_repo_reduction(reduction_dir)
    for method in ("sqrtm", "chol"):
        try:
            other = run_repo(params, repo, tol=args.tol, method=method)
        except Exception as error:
            print(f"\n=== mine vs repo ({method}) ===\nrepo path failed: {type(error).__name__}: {error}")
            continue
        compare(mine, other, f"repo reduction, method={method}")


if __name__ == "__main__":
    main()