import numpy as np
import torch
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from torch import nn
from pydantic import BaseModel, Field
from scipy.linalg import solve_discrete_lyapunov, cholesky, svd
 
 
class CompressionConfig(BaseModel):
    dt: float = Field(default=1.0, gt=0.0)
    tol: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Fraction of the sum of Hankel singular values to discard",
    )
    max_ratio: float = Field(default=0.95, gt=0.0, le=1.0)
 
    @property
    def threshold(self) -> float:
        """Fraction of the sum of Hankel singular values to retain."""
        return 1.0 - self.tol
 
 
class ReductionSchedule(BaseModel):
    red_start: int = Field(ge=0)
    red_end: int = Field(ge=0)
    red_interval: int = Field(gt=0)
 
    def fires_at(self, step: int) -> bool:
        return (
            self.red_start <= step < self.red_end
            and (step + 1) % self.red_interval == 0
        )
 
 
@dataclass(frozen=True)
class DiscreteSSM:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
 
    @property
    def n(self) -> int:
        return self.A.shape[0]
 
    @classmethod
    def from_numpy(cls, *, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray):
        A, B, C, D = (np.asarray(matrix) for matrix in (A, B, C, D))
        if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
            raise ValueError("A must be a non-empty square matrix")
        if B.ndim != 2 or B.shape[0] != A.shape[0]:
            raise ValueError("B must be a matrix with the same number of rows as A")
        if C.ndim != 2 or C.shape[1] != A.shape[0]:
            raise ValueError("C must be a matrix with the same number of columns as A")
        if not all(np.all(np.isfinite(matrix)) for matrix in (A, B, C, D)):
            raise ValueError("State-space matrices must contain only finite values")
        if np.max(np.abs(np.linalg.eigvals(A))) >= 1:
            raise ValueError("A is not stable (spectral radius >= 1)")
        return cls(A=A, B=B, C=C, D=D)
 
    @classmethod
    def from_continuous_diagonal_numpy(cls, *, a: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, dt: float):
        """Discretize a stable diagonal continuous-time system with ZOH input."""
        a = np.asarray(a)
        if a.ndim != 1 or a.size == 0:
            raise ValueError("a must be a non-empty one-dimensional array")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and greater than zero")
        if not np.all(np.isfinite(a)) or np.any(np.real(a) >= 0):
            raise ValueError("Continuous-time diagonal rates must be finite and stable")
 
        a_bar = np.exp(dt * a)
        integration_factor = np.full(a.shape, dt, dtype=np.result_type(a, dt))
        np.divide(np.expm1(dt * a), a, out=integration_factor, where=a != 0)
        B_bar = integration_factor[:, None] * B
        return cls.from_numpy(A=np.diag(a_bar), B=B_bar, C=C, D=D)
 
    @classmethod
    def from_tensors(cls, *, A_log: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor, dt: float):
        a = -np.exp(tensor_to_numpy(A_log))
        n = a.shape[0]
        return cls.from_continuous_diagonal_numpy(
            a=a,
            B=tensor_to_numpy(B).reshape(n, -1),
            C=tensor_to_numpy(C).reshape(-1, n),
            D=tensor_to_numpy(D),
            dt=dt,
        )
 
    @classmethod
    def from_module(cls, module: nn.Module, *, dt: float):
        sd = module.state_dict()
        return cls.from_tensors(
            A_log=sd['ssm.A_log'], B=sd['ssm.B'], C=sd['ssm.C'], D=sd['ssm.D'], dt=dt
        )
 
 
@dataclass(frozen=True)
class DiagonalDiscreteSSM:
    lam: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
 
    @property
    def n(self) -> int:
        return self.lam.shape[0]
 
    @classmethod
    def from_numpy(cls, *, lam: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray):
        lam, B, C, D = (np.asarray(matrix) for matrix in (lam, B, C, D))
        if lam.ndim != 1 or lam.size == 0:
            raise ValueError("lam must be a non-empty one-dimensional array")
        if B.ndim != 2 or B.shape[0] != lam.shape[0]:
            raise ValueError("B must be a matrix with one row per state")
        if C.ndim != 2 or C.shape[1] != lam.shape[0]:
            raise ValueError("C must be a matrix with one column per state")
        if not all(np.all(np.isfinite(matrix)) for matrix in (lam, B, C, D)):
            raise ValueError("State-space matrices must contain only finite values")
        if np.max(np.abs(lam)) >= 1:
            raise ValueError("lam is not stable (spectral radius >= 1)")
        return cls(lam=lam, B=B, C=C, D=D)
 
    @classmethod
    def from_lru_numpy(cls, *, nu_log, theta_log, B_re, B_im, gamma_log, C_re, C_im, D):
        lam = np.exp(-np.exp(nu_log) + 1j * np.exp(theta_log))
        B = (B_re + 1j * B_im) * np.exp(gamma_log)[:, None]
        C = C_re + 1j * C_im
        return cls.from_numpy(lam=lam, B=B, C=C, D=D)
 
    @classmethod
    def from_lru_module(cls, module: nn.Module):
        p = {k: tensor_to_numpy(v) for k, v in module.named_parameters()}
        return cls.from_lru_numpy(
            nu_log=p['nu_log'], theta_log=p['theta_log'],
            B_re=p['B_re'], B_im=p['B_im'], gamma_log=p['gamma_log'],
            C_re=p['C_re'], C_im=p['C_im'], D=p['D'],
        )
 
 
@dataclass(frozen=True)
class ReducedSSM:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    r: int
    error_bound: float
 
 
@dataclass(frozen=True)
class ReducedDiagonalSSM:
    lam: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    r: int
    error_bound: float
 
 
@dataclass(frozen=True)
class Balancing:
    Lp: np.ndarray
    Lq: np.ndarray
    U: np.ndarray
    sigma: np.ndarray
    Vh: np.ndarray
 
 
def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().copy()
 
 
class CompreSSM:
    @staticmethod
    def calculate_gramians_diagonal(lam: np.ndarray, *, B: np.ndarray, C: np.ndarray):
        denom = 1.0 - lam[:, None] * lam.conj()[None, :]
        P = (B @ B.conj().T) / denom
        Q = (C.conj().T @ C) / denom.conj()
        return (P + P.conj().T) / 2, (Q + Q.conj().T) / 2
 
    @staticmethod
    def calculate_gramians(A: np.ndarray, *, B: np.ndarray, C: np.ndarray):
        P = solve_discrete_lyapunov(A, B @ B.conj().T)
        Q = solve_discrete_lyapunov(A.conj().T, C.conj().T @ C)
        return (P + P.conj().T) / 2, (Q + Q.conj().T) / 2
 
    @staticmethod
    def diagonalize_reduced(Ar: np.ndarray, Br: np.ndarray, Cr: np.ndarray):
        lam_r, V = np.linalg.eig(Ar)
        return lam_r, np.linalg.solve(V, Br), Cr @ V
 
    @staticmethod
    def cholesky_with_jitter(M: np.ndarray, *, name: str, jitter: float = 1e-12, max_tries: int = 8):
        n = M.shape[0]
        scale = float(np.real(np.trace(M))) / n
        for k in range(max_tries):
            eps = 0.0 if k == 0 else jitter * scale * 10.0 ** (k - 1)
            try:
                return cholesky(M + eps * np.eye(n), lower=True)
            except np.linalg.LinAlgError:
                continue
        raise np.linalg.LinAlgError(f"Gramian not positive definite on matrix {name}, even with jitter")
 
    @staticmethod
    def calculate_sqrt(P: np.ndarray, *, Q: np.ndarray):
        Lp = CompreSSM.cholesky_with_jitter(P, name="P (controllability Gramian)")
        Lq = CompreSSM.cholesky_with_jitter(Q, name="Q (observability Gramian)")
        return Lp, Lq
 
    @staticmethod
    def HSVs(Lp: np.ndarray, *, Lq: np.ndarray):
        return svd(Lq.conj().T @ Lp)
 
    @staticmethod
    def balance(P: np.ndarray, *, Q: np.ndarray) -> Balancing:
        Lp, Lq = CompreSSM.calculate_sqrt(P, Q=Q)
        U, sigma, Vh = CompreSSM.HSVs(Lp, Lq=Lq)
        return Balancing(Lp=Lp, Lq=Lq, U=U, sigma=sigma, Vh=Vh)
 
    @staticmethod
    def rank_selection(sigma: np.ndarray, *, threshold: float) -> int:
        """
        Finds the smallest rank retaining the requested fraction of the HSV sum.
 
        Args:
            sigma: 1D array of Hankel singular values sorted in descending order.
            threshold: Fraction of the total HSV sum to retain (1.0 - tol).
        """
        sigma = np.asarray(sigma)
        if sigma.ndim != 1 or not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
            raise ValueError("sigma must be a one-dimensional array of finite non-negative values")
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be between zero and one")
        if sigma.size == 0:
            return 0
 
        total = np.sum(sigma)
        if total <= 0:
            return len(sigma)
        cumulative_sum = np.cumsum(sigma)
        ranks = np.flatnonzero(cumulative_sum >= threshold * total)
        return int(ranks[0] + 1) if ranks.size else len(sigma)
 
    @staticmethod
    def should_compress(r: int, n: int, *, max_ratio: float):
        return 0 < r < max_ratio * n
 
    @staticmethod
    def calculate_transform_matrix(Lp: np.ndarray, *, r: int, sigma: np.ndarray, Vh: np.ndarray):
        if np.any(sigma[:r] <= 0):
            raise np.linalg.LinAlgError("Retained Hankel singular values must be positive")
        Vr = Vh[:r].conj().T
        diag_entries = sigma[:r] ** -0.5
        return Lp @ Vr @ np.diag(diag_entries), diag_entries
 
    @staticmethod
    def calculate_projection_matrix(Lq: np.ndarray, *, r: int, diag_entries: np.ndarray, U: np.ndarray):
        return np.diag(diag_entries) @ U[:, :r].conj().T @ Lq.conj().T
 
    @staticmethod
    def truncate_A(A: np.ndarray, *, S: np.ndarray, T: np.ndarray):
        return S @ A @ T
 
    @staticmethod
    def truncate_B(S: np.ndarray, *, B: np.ndarray):
        return S @ B
 
    @staticmethod
    def truncate_C(C: np.ndarray, *, T: np.ndarray):
        return C @ T
 
    @staticmethod
    def truncate(A: np.ndarray, B: np.ndarray, C: np.ndarray, *, balancing: Balancing, r: int):
        T, diag_entries = CompreSSM.calculate_transform_matrix(
            balancing.Lp, r=r, sigma=balancing.sigma, Vh=balancing.Vh
        )
        S = CompreSSM.calculate_projection_matrix(
            balancing.Lq, r=r, diag_entries=diag_entries, U=balancing.U
        )
        return (
            CompreSSM.truncate_A(A, S=S, T=T),
            CompreSSM.truncate_B(S, B=B),
            CompreSSM.truncate_C(C, T=T),
        )
 
 
def compress_from_gramians(A: np.ndarray, B: np.ndarray, C: np.ndarray, *, P: np.ndarray, Q: np.ndarray, config: CompressionConfig):
    balancing = CompreSSM.balance(P, Q=Q)
    r = CompreSSM.rank_selection(balancing.sigma, threshold=config.threshold)
 
    if not CompreSSM.should_compress(r, A.shape[0], max_ratio=config.max_ratio):
        return None
 
    Ar, Br, Cr = CompreSSM.truncate(A, B, C, balancing=balancing, r=r)
    error_bound = 2 * float(np.real(np.sum(balancing.sigma[r:])))
    return Ar, Br, Cr, r, error_bound
 
 
def compress_numpy(system: DiscreteSSM, *, config: CompressionConfig) -> ReducedSSM | None:
    """Apply discrete-time balanced truncation to one stable, minimal LTI system.
 
    The returned error bound is the standard H-infinity bound for balanced
    truncation; it assumes the LTI system is stable and minimal.
    """
    P, Q = CompreSSM.calculate_gramians(system.A, B=system.B, C=system.C)
    result = compress_from_gramians(system.A, system.B, system.C, P=P, Q=Q, config=config)
    if result is None:
        return None
    Ar, Br, Cr, r, error_bound = result
    return ReducedSSM(A=Ar, B=Br, C=Cr, D=system.D, r=r, error_bound=error_bound)
 
 
def compress_diagonal_numpy(system: DiagonalDiscreteSSM, *, config: CompressionConfig) -> ReducedDiagonalSSM | None:
    """Balanced truncation of a diagonal system, returned in diagonal form."""
    P, Q = CompreSSM.calculate_gramians_diagonal(system.lam, B=system.B, C=system.C)
    result = compress_from_gramians(np.diag(system.lam), system.B, system.C, P=P, Q=Q, config=config)
    if result is None:
        return None
    Ar, Br, Cr, r, error_bound = result
    lam_r, Br_diag, Cr_diag = CompreSSM.diagonalize_reduced(Ar, Br, Cr)
    return ReducedDiagonalSSM(lam=lam_r, B=Br_diag, C=Cr_diag, D=system.D, r=r, error_bound=error_bound)
 
 
def compress_module(module: nn.Module, *, config: CompressionConfig) -> ReducedSSM | None:
    """Extract and compress the expected fixed-step diagonal SSM; does not mutate the module."""
    system = DiscreteSSM.from_module(module, dt=config.dt)
    return compress_numpy(system, config=config)
 
 
def compress_lru_module(module: nn.Module, *, config: CompressionConfig) -> ReducedDiagonalSSM | None:
    """Extract and compress one LRU block; does not mutate the module."""
    system = DiagonalDiscreteSSM.from_lru_module(module)
    return compress_diagonal_numpy(system, config=config)
 
 
def lru_arrays_from_diagonal(reduced: ReducedDiagonalSSM) -> dict[str, np.ndarray]:
    modulus = np.clip(np.abs(reduced.lam), 1e-12, 1.0 - 1e-12)
    theta = np.maximum(np.angle(reduced.lam) % (2 * np.pi), 1e-8)
    gamma = np.sqrt(1.0 - modulus ** 2)
    B_stored = reduced.B / gamma[:, None]
    return {
        'nu_log': np.log(-np.log(modulus)),
        'theta_log': np.log(theta),
        'gamma_log': np.log(gamma),
        'B_re': B_stored.real,
        'B_im': B_stored.imag,
        'C_re': reduced.C.real,
        'C_im': reduced.C.imag,
    }
 
 
def replace_parameter(module: nn.Module, name: str, array: np.ndarray) -> None:
    old = getattr(module, name)
    new = torch.tensor(np.ascontiguousarray(array), dtype=old.dtype, device=old.device)
    setattr(module, name, nn.Parameter(new, requires_grad=old.requires_grad))
 
 
def write_back_lru(module: nn.Module, *, reduced: ReducedDiagonalSSM) -> None:
    """Replace the LRU block's parameters with the reduced ones. D is unchanged."""
    for name, array in lru_arrays_from_diagonal(reduced).items():
        replace_parameter(module, name, array)
 
 
def find_blocks(model: nn.Module, *, block_cls: type) -> list[nn.Module]:
    return [m for m in model.modules() if isinstance(m, block_cls)]
 
 
def reduce_lru_blocks(blocks: Iterable[nn.Module], *, config: CompressionConfig) -> list[ReducedDiagonalSSM | None]:
    results = []
    for block in blocks:
        reduced = compress_lru_module(block, config=config)
        if reduced is not None:
            write_back_lru(block, reduced=reduced)
        results.append(reduced)
    return results
 
 
def run_scheduled_reduction(
    *,
    step: int,
    schedule: ReductionSchedule,
    blocks: Iterable[nn.Module],
    config: CompressionConfig,
    make_optimizer: Callable[[], torch.optim.Optimizer],
):
    """Returns a freshly built optimizer if any block was reduced, otherwise None."""
    if not schedule.fires_at(step):
        return None
    results = reduce_lru_blocks(blocks, config=config)
    if all(result is None for result in results):
        return None
    return make_optimizer()