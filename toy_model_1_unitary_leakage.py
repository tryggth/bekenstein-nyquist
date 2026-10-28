#!/usr/bin/env python3
"""
Milestone 1: Minimalist Algebraic Toy Model — The N x N Unitary Block-Leakage Sandbox
(Phase 1 & Phase 5: unitary_bandwidth_depth_tradeoff)

Tests the finite-dimensional kinematic core of the Bekenstein-Nyquist Shock Worldline
Conjecture in H = C^N using IEEE 754 float64 / complex128 precision.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import torch

# Enforce IEEE 754 float64 / complex128 precision globally
torch.set_default_dtype(torch.float64)
COMPLEX_DTYPE = torch.complex128
FLOAT_DTYPE = torch.float64
RANK_TOL = 1e-10
ASSERT_TOL = 1e-13


@dataclass
class BlockPartition:
    """2x2 block partition of an N x N matrix with respect to V_Lambda (dim d) and V_Lambda^perp (dim N-d)."""

    U11: torch.Tensor  # (d, d)
    U12: torch.Tensor  # (d, N - d)
    U21: torch.Tensor  # (N - d, d)
    U22: torch.Tensor  # (N - d, N - d)


@dataclass
class ElementaryGate:
    """Single elementary unitary update W_k in U(N) coupling V_Lambda with a 1D clock wire C_k subset V_Lambda^perp."""

    step_index: int
    W: torch.Tensor  # (N, N) unitary
    clock_wire_vector: torch.Tensor  # (N,) unit vector spanning C_k subset V_Lambda^perp
    P_Ck: torch.Tensor  # (N, N) rank-1 projector onto C_k
    leakage_rank: int  # rank((I - P_Lambda) W_k P_Lambda) <= 1
    commutator_rank: int  # rank([P_Lambda, W_k]) <= 2


@dataclass
class DepthSweepResult:
    """Observables and verification metrics for a depth-D sequential factorization U_D = W_D ... W_1."""

    depth: int
    target_rank_r: int
    U_D: torch.Tensor
    gates: List[ElementaryGate]
    forward_leakage_rank: int
    commutator_rank: int
    telescoping_commutator_error: float
    one_sided_recursion_max_error: float
    isometric_error: float
    theoretical_min_isometric_error: float
    compression_error: float
    theoretical_min_compression_error: float
    exact_isometric_success: bool
    exact_compression_success: bool


def numerical_rank(M: torch.Tensor, tol: float = RANK_TOL) -> int:
    """Compute the numerical rank of a complex matrix via SVD in float64/complex128."""
    if M.numel() == 0:
        return 0
    s = torch.linalg.svdvals(M)
    return int(torch.sum(s > tol).item())


def build_bandlimit_projector(N: int = 10, d: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct the orthogonal bandlimit projector P_Lambda = diag(I_d, 0_{N-d})
    onto the d-dimensional low-frequency Bekenstein subspace V_Lambda subset C^N,
    and its ultraviolet orthogonal complement projector Q_Lambda = I_N - P_Lambda.
    """
    if not (0 < d < N):
        raise ValueError(f"Require 0 < d < N, got N={N}, d={d}.")
    diag = torch.zeros(N, dtype=COMPLEX_DTYPE)
    diag[:d] = 1.0 + 0.0j
    P_Lambda = torch.diag(diag)
    I_N = torch.eye(N, dtype=COMPLEX_DTYPE)
    Q_Lambda = I_N - P_Lambda
    return P_Lambda, Q_Lambda


def partition_blocks(M: torch.Tensor, d: int) -> BlockPartition:
    """Partition an N x N matrix into 2x2 blocks around low-frequency dimension d."""
    return BlockPartition(
        U11=M[:d, :d],
        U12=M[:d, d:],
        U21=M[d:, :d],
        U22=M[d:, d:],
    )


def _haar_unitary(n: int, generator: torch.Generator) -> torch.Tensor:
    """Sample a random unitary matrix in U(n) via QR decomposition with Mezzadri phase normalization."""
    real = torch.randn((n, n), dtype=FLOAT_DTYPE, generator=generator)
    imag = torch.randn((n, n), dtype=FLOAT_DTYPE, generator=generator)
    Z = torch.complex(real, imag) / torch.sqrt(torch.tensor(2.0, dtype=FLOAT_DTYPE))
    Q, R = torch.linalg.qr(Z)
    diag_R = torch.diagonal(R)
    phases = diag_R / torch.abs(diag_R)
    return Q * phases.unsqueeze(0)


def generate_nonlinear_unitary(
    N: int = 10,
    d: int = 4,
    method: Literal["qr", "hamiltonian"] = "qr",
    dt: float = 0.65,
    target_rank: Optional[int] = None,
    seed: int = 42,
) -> torch.Tensor:
    """
    Generate a nonlinear transport surrogate U_nl in U(N) that mixes V_Lambda and V_Lambda^perp.

    Args:
        N: Total Hilbert space dimension (default 10).
        d: Low-frequency Bekenstein subspace dimension (default 4).
        method:
            - "qr": QR decomposition of a random complex matrix (or CS-structured unitary if target_rank is set).
            - "hamiltonian": Matrix exponential U_nl = exp(i * dt * H_nl) of a Hermitian mode-coupling Hamiltonian.
        dt: Macro-step evolution time for the Hamiltonian method.
        target_rank: Optional controlled cross-subspace leakage rank r in {0, ..., min(d, N-d)}.
                     If None, generates a generic full-coupling unitary with r = min(d, N-d).
        seed: Random number generator seed for reproducibility.
    """
    max_r = min(d, N - d)
    if target_rank is not None and not (0 <= target_rank <= max_r):
        raise ValueError(f"target_rank must be in [0, {max_r}], got {target_rank}.")

    gen = torch.Generator().manual_seed(seed)

    if method == "qr":
        if target_rank is None:
            return _haar_unitary(N, gen)
        # Construct a Haar-random unitary within V_Lambda and V_Lambda^perp coupled across exactly `target_rank` channels
        L1 = _haar_unitary(d, gen)
        R1 = _haar_unitary(d, gen)
        L2 = _haar_unitary(N - d, gen)
        R2 = _haar_unitary(N - d, gen)

        # Angles theta_1, ..., theta_r in (0.15*pi, 0.40*pi) so sin(theta_k) and cos(theta_k) are strictly away from 0
        angles = torch.zeros(d, dtype=FLOAT_DTYPE)
        if target_rank > 0:
            rand_u = torch.rand(target_rank, dtype=FLOAT_DTYPE, generator=gen)
            angles[:target_rank] = (0.15 + 0.25 * rand_u) * torch.pi

        c = torch.cos(angles).to(COMPLEX_DTYPE)
        s = torch.sin(angles).to(COMPLEX_DTYPE)

        C_mat = torch.diag(c)
        S_21 = torch.zeros((N - d, d), dtype=COMPLEX_DTYPE)
        S_12 = torch.zeros((d, N - d), dtype=COMPLEX_DTYPE)
        for k in range(target_rank):
            S_21[k, k] = s[k]
            S_12[k, k] = -s[k]

        C_22 = torch.eye(N - d, dtype=COMPLEX_DTYPE)
        for k in range(target_rank):
            C_22[k, k] = c[k]

        U11 = L1 @ C_mat @ R1.mH
        U21 = L2 @ S_21 @ R1.mH
        U12 = L1 @ S_12 @ R2.mH
        U22 = L2 @ C_22 @ R2.mH

        top = torch.cat([U11, U12], dim=1)
        bottom = torch.cat([U21, U22], dim=1)
        return torch.cat([top, bottom], dim=0)

    elif method == "hamiltonian":
        # Build Hermitian mode-coupling Hamiltonian H_nl = H_nl^dagger
        real = torch.randn((N, N), dtype=FLOAT_DTYPE, generator=gen)
        imag = torch.randn((N, N), dtype=FLOAT_DTYPE, generator=gen)
        Z = torch.complex(real, imag) / torch.sqrt(torch.tensor(2.0 * N, dtype=FLOAT_DTYPE))
        H_nl = 0.5 * (Z + Z.mH)

        if target_rank is not None:
            # Project the off-diagonal mode-coupling block H_21 onto a rank-`target_rank` matrix
            H21 = H_nl[d:, :d]
            U_h, S_h, Vh_h = torch.linalg.svd(H21, full_matrices=False)
            S_trunc = torch.zeros_like(S_h)
            S_trunc[:target_rank] = S_h[:target_rank] + 0.5  # ensure non-vanishing singular values
            H21_rank_r = (U_h * S_trunc.to(COMPLEX_DTYPE).unsqueeze(0)) @ Vh_h
            # To ensure exp(i dt H_nl) has exact leakage rank `target_rank`, make H_11 and H_22 preserve the SVD bases
            H11_diag = torch.randn(d, dtype=FLOAT_DTYPE, generator=gen).to(COMPLEX_DTYPE)
            H22_diag = torch.randn(N - d, dtype=FLOAT_DTYPE, generator=gen).to(COMPLEX_DTYPE)
            U_h_full, _, _ = torch.linalg.svd(H21, full_matrices=True)
            H_nl[:d, :d] = Vh_h.mH @ torch.diag(H11_diag) @ Vh_h
            H_nl[d:, d:] = U_h_full @ torch.diag(H22_diag) @ U_h_full.mH
            H_nl[d:, :d] = H21_rank_r
            H_nl[:d, d:] = H21_rank_r.mH

        return torch.linalg.matrix_exp(1j * dt * H_nl)

    else:
        raise ValueError(f"Unknown method: {method!r}. Expected 'qr' or 'hamiltonian'.")


def verify_kinematic_commutator_identities(
    U_nl: torch.Tensor, P_Lambda: torch.Tensor, d: int
) -> Dict[str, object]:
    """
    Compute and verify all Phase 1 kinematic identities:
      1. Unitarity of U_nl: U_nl^dagger U_nl = I_N and U_nl U_nl^dagger = I_N.
      2. Forward leakage A = (I - P_Lambda) U_nl P_Lambda and reverse leakage B = P_Lambda U_nl (I - P_Lambda).
      3. Unitary off-diagonal Gram identities:
           U11 U11^dagger + U12 U12^dagger = I_d
           U11^dagger U11 + U21^dagger U21 = I_d
      4. Rank equality: rank(A) = rank(B) = r and matching singular values sigma(A) = sigma(B).
      5. Commutator decomposition: [P_Lambda, U_nl] = B - A and rank([P_Lambda, U_nl]) = rank(A) + rank(B) = 2r.
    """
    N = U_nl.shape[0]
    I_N = torch.eye(N, dtype=COMPLEX_DTYPE)
    I_d = torch.eye(d, dtype=COMPLEX_DTYPE)
    Q_Lambda = I_N - P_Lambda

    # 1. Unitarity check
    unitary_err_left = float(torch.linalg.norm(U_nl.mH @ U_nl - I_N, ord="fro").item())
    unitary_err_right = float(torch.linalg.norm(U_nl @ U_nl.mH - I_N, ord="fro").item())
    assert unitary_err_left < ASSERT_TOL, f"U_nl not left-unitary: err={unitary_err_left}"
    assert unitary_err_right < ASSERT_TOL, f"U_nl not right-unitary: err={unitary_err_right}"

    # 2. Block partition and leakage matrices
    blocks = partition_blocks(U_nl, d)
    A = Q_Lambda @ U_nl @ P_Lambda  # Forward Nyquist leakage (bottom-left block U21)
    B = P_Lambda @ U_nl @ Q_Lambda  # Reverse Nyquist leakage (top-right block U12)

    assert torch.linalg.norm(A[d:, :d] - blocks.U21, ord="fro").item() < ASSERT_TOL
    assert torch.linalg.norm(B[:d, d:] - blocks.U12, ord="fro").item() < ASSERT_TOL

    # 3. Off-diagonal Gram identities
    gram_row_err = float(
        torch.linalg.norm(
            blocks.U11 @ blocks.U11.mH + blocks.U12 @ blocks.U12.mH - I_d, ord="fro"
        ).item()
    )
    gram_col_err = float(
        torch.linalg.norm(
            blocks.U11.mH @ blocks.U11 + blocks.U21.mH @ blocks.U21 - I_d, ord="fro"
        ).item()
    )
    assert gram_row_err < ASSERT_TOL, f"Row Gram identity failed: err={gram_row_err}"
    assert gram_col_err < ASSERT_TOL, f"Column Gram identity failed: err={gram_col_err}"

    # 4. Singular value & rank identity between A and B
    sv_U11 = torch.linalg.svdvals(blocks.U11)
    sv_U21 = torch.linalg.svdvals(blocks.U21)
    sv_U12 = torch.linalg.svdvals(blocks.U12)
    # Non-zero singular values of U21 and U12 must coincide and satisfy s_i^2 + c_i^2 = 1
    k_min = min(d, N - d)
    sv_diff_AB = float(torch.linalg.norm(sv_U21[:k_min] - sv_U12[:k_min]).item())
    assert sv_diff_AB < ASSERT_TOL, f"Singular values of A and B differ: {sv_diff_AB}"

    rank_A = numerical_rank(A)
    rank_B = numerical_rank(B)
    assert rank_A == rank_B, f"Expected rank(A) == rank(B), got {rank_A} != {rank_B}"
    r = rank_A

    # 5. Commutator [P_Lambda, U_nl] = B - A and rank([P_Lambda, U_nl]) = 2r
    commutator = P_Lambda @ U_nl - U_nl @ P_Lambda
    comm_decomp_err = float(torch.linalg.norm(commutator - (B - A), ord="fro").item())
    assert comm_decomp_err < ASSERT_TOL, f"[P_Lambda, U_nl] != B - A: err={comm_decomp_err}"

    rank_comm = numerical_rank(commutator)
    assert rank_comm == rank_A + rank_B == 2 * r, (
        f"Expected rank([P_Lambda, U_nl]) == 2r == {2 * r}, got {rank_comm}"
    )

    return {
        "N": N,
        "d": d,
        "r": r,
        "rank_A": rank_A,
        "rank_B": rank_B,
        "rank_commutator": rank_comm,
        "unitary_err": max(unitary_err_left, unitary_err_right),
        "gram_row_err": gram_row_err,
        "gram_col_err": gram_col_err,
        "sv_diff_AB": sv_diff_AB,
        "comm_decomp_err": comm_decomp_err,
        "sv_U11": sv_U11,
        "sv_U21": sv_U21,
        "sv_U12": sv_U12,
        "A": A,
        "B": B,
        "commutator": commutator,
        "blocks": blocks,
    }


def factorize_into_elementary_gates(
    U_nl: torch.Tensor,
    P_Lambda: torch.Tensor,
    d: int,
    depth_D: int,
    wire_mode: Literal["aligned_isometric", "coordinate_compression"] = "aligned_isometric",
) -> DepthSweepResult:
    """
    Factor U_nl restricted to V_Lambda into a sequential product of D elementary unitary
    updates U_D = W_D W_{D-1} ... W_1, where each W_k in U(N) acts non-trivially only on
    V_Lambda plus a single 1D auxiliary clock register wire C_k subset V_Lambda^perp (dim C_k = 1),
    acting as the identity on V_Lambda^perp ominus C_k.

    Using the Cosine-Sine (CS) / SVD decomposition of the restriction U_nl P_Lambda:
      - U11 = L diag(c_1, ..., c_d) R^dagger
      - U21 = Q_perp S R^dagger, with s_k = sqrt(1 - c_k^2) ordered so s_1 >= s_2 >= ... >= s_r > 0.
    In `aligned_isometric` mode, C_k = span{q_k} (the k-th column of Q_perp in V_Lambda^perp),
    which simultaneously achieves exact isometric reconstruction U_D P_Lambda = U_nl P_Lambda
    AND exact block compression P_Lambda U_D P_Lambda = P_Lambda U_nl P_Lambda iff D >= r.
    In `coordinate_compression` mode, C_k = span{e_{d + ((k-1) mod (N-d))}} (standard basis wire),
    which achieves exact block compression P_Lambda U_D P_Lambda = P_Lambda U_nl P_Lambda iff D >= r.
    """
    N = U_nl.shape[0]
    if depth_D < 0:
        raise ValueError(f"depth_D must be >= 0, got {depth_D}")

    I_N = torch.eye(N, dtype=COMPLEX_DTYPE)
    Q_Lambda = I_N - P_Lambda
    blocks = partition_blocks(U_nl, d)
    r = numerical_rank(blocks.U21)

    # SVD of U21 = Q_perp @ diag(s_1, ..., s_d) @ R^dagger (with s_1 >= s_2 >= ... >= s_d)
    Q_perp, s_vals_raw, Rh = torch.linalg.svd(blocks.U21, full_matrices=True)
    R = Rh.mH  # (d, d) unitary on V_Lambda
    s_vals = torch.clamp(s_vals_raw, 0.0, 1.0)
    # Pad s_vals to length d if N - d < d
    if s_vals.shape[0] < d:
        s_vals = torch.cat(
            [s_vals, torch.zeros(d - s_vals.shape[0], dtype=FLOAT_DTYPE)], dim=0
        )

    # Extract L in U(d) and cosines c_k from U11 @ R = L @ diag(c_1, ..., c_d)
    U11_R = blocks.U11 @ R  # columns are orthogonal with norms c_k = sqrt(1 - s_k^2)
    c_vals = torch.sqrt(torch.clamp(1.0 - s_vals**2, min=0.0))

    L = torch.zeros((d, d), dtype=COMPLEX_DTYPE)
    for k in range(d):
        col = U11_R[:, k]
        norm_col = torch.linalg.norm(col).item()
        if norm_col > 1e-12:
            L[:, k] = col / norm_col
            c_vals[k] = norm_col
    # If any c_k == 0 (purely leaking channel s_k == 1), complete L to a unitary basis on V_Lambda
    if torch.any(c_vals <= 1e-12):
        # Orthonormalize columns of L via QR for zero-cosine columns
        Q_L, R_L = torch.linalg.qr(L + 1e-15 * torch.eye(d, dtype=COMPLEX_DTYPE))
        for k in range(d):
            if c_vals[k] <= 1e-12:
                L[:, k] = Q_L[:, k]

    # Basis change unitary on V_Lambda: B_V = L @ R^dagger in U(d)
    B_V = L @ R.mH
    W_basis = I_N.clone()
    W_basis[:d, :d] = B_V

    gates: List[ElementaryGate] = []

    if depth_D == 0:
        # Depth 0 (zero clock-wire couplings across V_Lambda^perp): pure in-band unitary W_basis on V_Lambda
        U_D = W_basis.clone()
    else:
        for k in range(1, depth_D + 1):
            idx = k - 1
            # Select 1D clock wire C_k subset V_Lambda^perp
            wire_idx = idx % (N - d)
            if wire_mode == "aligned_isometric":
                q_vec_perp = Q_perp[:, wire_idx]
                clock_vec = torch.zeros(N, dtype=COMPLEX_DTYPE)
                clock_vec[d:] = q_vec_perp
            elif wire_mode == "coordinate_compression":
                clock_vec = torch.zeros(N, dtype=COMPLEX_DTYPE)
                clock_vec[d + wire_idx] = 1.0 + 0.0j
            else:
                raise ValueError(f"Unknown wire_mode: {wire_mode!r}")

            P_Ck = torch.outer(clock_vec, clock_vec.conj())

            # Construct the 2D Givens rotation in span{l_k, clock_vec} for k <= min(d, r)
            G_k = I_N.clone()
            if idx < min(d, r):
                l_vec = torch.zeros(N, dtype=COMPLEX_DTYPE)
                l_vec[:d] = L[:, idx]
                c_k = c_vals[idx].to(COMPLEX_DTYPE)
                s_k = s_vals[idx].to(COMPLEX_DTYPE)

                # On span{l_vec, clock_vec}, replace I_2 with [[c_k, -s_k], [s_k, c_k]]
                P_plane = torch.outer(l_vec, l_vec.conj()) + P_Ck
                rot_plane = (
                    c_k * torch.outer(l_vec, l_vec.conj())
                    - s_k * torch.outer(l_vec, clock_vec.conj())
                    + s_k * torch.outer(clock_vec, l_vec.conj())
                    + c_k * P_Ck
                )
                G_k = I_N - P_plane + rot_plane

            # Absorb the internal V_Lambda basis rotation B_V into W_1 (since W_1 is free to act on V_Lambda)
            W_k = G_k @ W_basis if k == 1 else G_k

            # Verify elementary gate constraints
            assert torch.linalg.norm(W_k.mH @ W_k - I_N, ord="fro").item() < ASSERT_TOL
            # Verify W_k acts as the identity on V_Lambda^perp ominus C_k
            P_perp_compl = Q_Lambda - P_Ck
            id_compl_err_right = float(
                torch.linalg.norm((W_k - I_N) @ P_perp_compl, ord="fro").item()
            )
            id_compl_err_left = float(
                torch.linalg.norm(P_perp_compl @ (W_k - I_N), ord="fro").item()
            )
            assert id_compl_err_right < ASSERT_TOL, (
                f"Gate W_{k} not identity on V_Lambda^perp ominus C_{k}: {id_compl_err_right}"
            )
            assert id_compl_err_left < ASSERT_TOL, (
                f"Gate W_{k} not left-identity on V_Lambda^perp ominus C_{k}: {id_compl_err_left}"
            )

            gate_leakage_rank = numerical_rank(Q_Lambda @ W_k @ P_Lambda)
            gate_comm_rank = numerical_rank(P_Lambda @ W_k - W_k @ P_Lambda)
            assert gate_leakage_rank <= 1, (
                f"Gate W_{k} violated rank((I - P_Lambda) W_k P_Lambda) <= 1: got {gate_leakage_rank}"
            )
            assert gate_comm_rank <= 2, (
                f"Gate W_{k} violated rank([P_Lambda, W_k]) <= 2: got {gate_comm_rank}"
            )

            gates.append(
                ElementaryGate(
                    step_index=k,
                    W=W_k,
                    clock_wire_vector=clock_vec,
                    P_Ck=P_Ck,
                    leakage_rank=gate_leakage_rank,
                    commutator_rank=gate_comm_rank,
                )
            )

        # Sequential product U_D = W_D W_{D-1} ... W_1
        U_D = I_N.clone()
        for gate in gates:
            U_D = gate.W @ U_D

    # Verify the telescoping commutator identity:
    # [P_Lambda, U_D] = sum_{k=1}^D (W_D ... W_{k+1}) [P_Lambda, W_k] (W_{k-1} ... W_1)
    comm_UD = P_Lambda @ U_D - U_D @ P_Lambda
    telescoping_sum = torch.zeros((N, N), dtype=COMPLEX_DTYPE)
    for k_idx in range(depth_D):
        # Prefix W_{k-1} ... W_1
        prefix = I_N.clone()
        for j in range(k_idx):
            prefix = gates[j].W @ prefix
        # Suffix W_D ... W_{k+1}
        suffix = I_N.clone()
        for j in range(k_idx + 1, depth_D):
            suffix = gates[j].W @ suffix
        W_k = gates[k_idx].W
        comm_Wk = P_Lambda @ W_k - W_k @ P_Lambda
        telescoping_sum = telescoping_sum + suffix @ comm_Wk @ prefix

    telescoping_err = float(torch.linalg.norm(comm_UD - telescoping_sum, ord="fro").item())
    assert telescoping_err < ASSERT_TOL, (
        f"Telescoping commutator identity failed at D={depth_D}: err={telescoping_err}"
    )

    # Verify the one-sided leakage decomposition recursion at each step:
    # (I - P)(W_k U_{k-1}) P = ((I - P) W_k P)(P U_{k-1} P) + ((I - P) W_k (I - P))((I - P) U_{k-1} P)
    U_prev = I_N.clone()
    max_recursion_err = 0.0
    for gate in gates:
        W_k = gate.W
        U_curr = W_k @ U_prev
        lhs = Q_Lambda @ U_curr @ P_Lambda
        rhs = (Q_Lambda @ W_k @ P_Lambda) @ (P_Lambda @ U_prev @ P_Lambda) + (
            Q_Lambda @ W_k @ Q_Lambda
        ) @ (Q_Lambda @ U_prev @ P_Lambda)
        rec_err = float(torch.linalg.norm(lhs - rhs, ord="fro").item())
        assert rec_err < ASSERT_TOL, f"One-sided leakage recursion failed: {rec_err}"
        max_recursion_err = max(max_recursion_err, rec_err)
        U_prev = U_curr

    # Compute ranks and subadditivity bounds
    fwd_leak_rank = numerical_rank(Q_Lambda @ U_D @ P_Lambda)
    comm_rank = numerical_rank(comm_UD)
    assert fwd_leak_rank <= depth_D, (
        f"Rank subadditivity violated: rank((I-P) U_D P)={fwd_leak_rank} > D={depth_D}"
    )
    assert comm_rank <= 2 * depth_D, (
        f"Commutator rank subadditivity violated: rank([P, U_D])={comm_rank} > 2D={2 * depth_D}"
    )

    # Reconstruction errors
    isometric_err = float(
        torch.linalg.norm((U_D - U_nl) @ P_Lambda, ord="fro").item()
    )
    compression_err = float(
        torch.linalg.norm(P_Lambda @ (U_D - U_nl) @ P_Lambda, ord="fro").item()
    )

    # Theoretical Eckart-Young lower bounds for any rank-D leakage unitary
    if depth_D == 0:
        # Best depth-0 unitary on V_Lambda is L R^dagger (or I_N if no gates at all)
        theo_min_iso = float(
            torch.sqrt(torch.sum(s_vals[:r] ** 2 + (1.0 - c_vals[:r]) ** 2)).item()
        )
        theo_min_comp = float(torch.sqrt(torch.sum((1.0 - c_vals[:r]) ** 2)).item())
    elif depth_D < r:
        theo_min_iso = float(
            torch.sqrt(
                torch.sum(s_vals[depth_D:r] ** 2 + (1.0 - c_vals[depth_D:r]) ** 2)
            ).item()
        )
        theo_min_comp = float(
            torch.sqrt(torch.sum((1.0 - c_vals[depth_D:r]) ** 2)).item()
        )
    else:
        theo_min_iso = 0.0
        theo_min_comp = 0.0

    return DepthSweepResult(
        depth=depth_D,
        target_rank_r=r,
        U_D=U_D,
        gates=gates,
        forward_leakage_rank=fwd_leak_rank,
        commutator_rank=comm_rank,
        telescoping_commutator_error=telescoping_err,
        one_sided_recursion_max_error=max_recursion_err,
        isometric_error=isometric_err,
        theoretical_min_isometric_error=theo_min_iso,
        compression_error=compression_err,
        theoretical_min_compression_error=theo_min_comp,
        exact_isometric_success=(isometric_err < ASSERT_TOL),
        exact_compression_success=(compression_err < ASSERT_TOL),
    )


def run_depth_tradeoff_sweep(
    U_nl: torch.Tensor,
    P_Lambda: torch.Tensor,
    d: int,
    max_depth: Optional[int] = None,
) -> Tuple[List[DepthSweepResult], List[DepthSweepResult]]:
    """
    Sweep circuit depth D = 0, 1, ..., max_depth for both:
      1. `aligned_isometric` clock wires (testing U_D P_Lambda = U_nl P_Lambda and P_Lambda U_D P_Lambda = P_Lambda U_nl P_Lambda)
      2. `coordinate_compression` clock wires (testing P_Lambda U_D P_Lambda = P_Lambda U_nl P_Lambda on standard basis wires)
    and assert the Phase 1 / Phase 5 Bandwidth-Depth Tradeoff Theorem:
      - Exact unitary reconstruction fails for EVERY D < r.
      - Exact unitary reconstruction succeeds iff D >= r = rank([P_Lambda, U_nl]) / 2.
    """
    N = U_nl.shape[0]
    r = numerical_rank((torch.eye(N, dtype=COMPLEX_DTYPE) - P_Lambda) @ U_nl @ P_Lambda)
    if max_depth is None:
        max_depth = N - d

    aligned_results: List[DepthSweepResult] = []
    coord_results: List[DepthSweepResult] = []

    for D in range(0, max_depth + 1):
        res_aligned = factorize_into_elementary_gates(
            U_nl, P_Lambda, d, depth_D=D, wire_mode="aligned_isometric"
        )
        res_coord = factorize_into_elementary_gates(
            U_nl, P_Lambda, d, depth_D=D, wire_mode="coordinate_compression"
        )

        if D < r:
            # Falsification check: must strictly fail for D < r
            assert not res_aligned.exact_isometric_success, (
                f"Impossible isometric reconstruction succeeded at D={D} < r={r}"
            )
            assert not res_aligned.exact_compression_success, (
                f"Impossible block compression succeeded at D={D} < r={r}"
            )
            assert not res_coord.exact_compression_success, (
                f"Impossible coordinate block compression succeeded at D={D} < r={r}"
            )
            assert res_aligned.commutator_rank == 2 * D < 2 * r
            if D >= 1:
                assert (
                    abs(
                        res_aligned.isometric_error
                        - res_aligned.theoretical_min_isometric_error
                    )
                    < ASSERT_TOL
                )
                assert (
                    abs(
                        res_aligned.compression_error
                        - res_aligned.theoretical_min_compression_error
                    )
                    < ASSERT_TOL
                )
        else:
            # Verification check: must succeed to machine precision for D >= r
            assert res_aligned.exact_isometric_success, (
                f"Aligned isometric reconstruction failed at D={D} >= r={r}: err={res_aligned.isometric_error}"
            )
            assert res_aligned.exact_compression_success, (
                f"Aligned compression reconstruction failed at D={D} >= r={r}: err={res_aligned.compression_error}"
            )
            assert res_coord.exact_compression_success, (
                f"Coordinate compression reconstruction failed at D={D} >= r={r}: err={res_coord.compression_error}"
            )
            assert res_aligned.commutator_rank == 2 * r
            assert res_aligned.forward_leakage_rank == r

        aligned_results.append(res_aligned)
        coord_results.append(res_coord)

    return aligned_results, coord_results


def run_adversarial_falsification_check(
    U_nl: torch.Tensor, P_Lambda: torch.Tensor, d: int, num_random_trials: int = 200, seed: int = 123
) -> Dict[str, float]:
    """
    Falsification test: Generate arbitrary random circuits U_D = W_D ... W_1 of depth D = r - 1
    where each W_k couples V_Lambda to an arbitrary 1D subspace C_k subset V_Lambda^perp,
    and verify that NO depth D < r circuit can ever achieve rank((I - P) U_D P) >= r or
    reduce the block compression / isometric error below the Eckart-Young spectral tail bound.
    """
    N = U_nl.shape[0]
    I_N = torch.eye(N, dtype=COMPLEX_DTYPE)
    Q_Lambda = I_N - P_Lambda
    blocks = partition_blocks(U_nl, d)
    r = numerical_rank(blocks.U21)
    if r == 0:
        return {"min_observed_iso_err": 0.0, "eckart_young_lower_bound": 0.0}

    D_sub = r - 1
    s_vals = torch.linalg.svdvals(blocks.U21)
    # Universal Eckart-Young lower bound on ||(I - P_Lambda)(U_D - U_nl) P_Lambda||_F for any rank <= D_sub operator
    ey_lower_bound = float(torch.linalg.norm(s_vals[D_sub:r]).item())

    gen = torch.Generator().manual_seed(seed)
    min_observed_err = float("inf")
    max_observed_comm_rank = 0

    for _ in range(num_random_trials):
        U_D = I_N.clone()
        for _k in range(D_sub):
            # Random 1D clock wire in V_Lambda^perp
            v_real = torch.randn(N - d, dtype=FLOAT_DTYPE, generator=gen)
            v_imag = torch.randn(N - d, dtype=FLOAT_DTYPE, generator=gen)
            v = torch.complex(v_real, v_imag)
            v = v / torch.linalg.norm(v)

            # Build (d+1)x(d+1) Haar unitary acting on V_Lambda oplus span{v}
            U_sub = _haar_unitary(d + 1, gen)
            # Embed into C^N via basis [e_0, ..., e_{d-1}, (0, v)]
            basis = torch.zeros((N, d + 1), dtype=COMPLEX_DTYPE)
            basis[:d, :d] = torch.eye(d, dtype=COMPLEX_DTYPE)
            basis[d:, d] = v
            W_k = I_N - (basis @ basis.mH) + (basis @ U_sub @ basis.mH)

            assert numerical_rank(Q_Lambda @ W_k @ P_Lambda) <= 1
            assert numerical_rank(P_Lambda @ W_k - W_k @ P_Lambda) <= 2
            U_D = W_k @ U_D

        comm_rank = numerical_rank(P_Lambda @ U_D - U_D @ P_Lambda)
        assert comm_rank <= 2 * D_sub < 2 * r
        max_observed_comm_rank = max(max_observed_comm_rank, comm_rank)

        iso_err = float(torch.linalg.norm((U_D - U_nl) @ P_Lambda, ord="fro").item())
        assert iso_err >= ey_lower_bound - 1e-12, (
            f"Eckart-Young lower bound violated: {iso_err} < {ey_lower_bound}"
        )
        min_observed_err = min(min_observed_err, iso_err)

    return {
        "tested_depth": D_sub,
        "target_rank_r": r,
        "max_observed_comm_rank": max_observed_comm_rank,
        "min_observed_iso_err": min_observed_err,
        "eckart_young_lower_bound": ey_lower_bound,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Milestone 1: N x N Unitary Block-Leakage Sandbox (Phase 1 & Phase 5)"
    )
    parser.add_argument("--N", type=int, default=10, help="Fine-grid dimension N (default: 10)")
    parser.add_argument(
        "--d", type=int, default=4, help="Low-frequency Bekenstein subspace dim d (default: 4)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    N, d = args.N, args.d
    P_Lambda, Q_Lambda = build_bandlimit_projector(N=N, d=d)

    print("=" * 88)
    print("MILESTONE 1: N x N Unitary Block-Leakage Sandbox (Phase 1 & Phase 5 Verification)")
    print(f"Precision: {COMPLEX_DTYPE} / {FLOAT_DTYPE} | State space dim N = {N}, Bekenstein dim d = {d}, UV dim N-d = {N - d}")
    print("=" * 88)

    for method in ("qr", "hamiltonian"):
        print(f"\n--- [Method: {method.upper()}] Full-Coupling Nonlinear Transport Surrogate ---")
        U_nl = generate_nonlinear_unitary(N=N, d=d, method=method, seed=args.seed)
        kin = verify_kinematic_commutator_identities(U_nl, P_Lambda, d=d)
        r = kin["r"]

        print(f"  Unitarity error ||U_nl^H U_nl - I_N||_F         : {kin['unitary_err']:.3e}")
        print(f"  Row Gram error  ||U11 U11^H + U12 U12^H - I_d||_F: {kin['gram_row_err']:.3e}")
        print(f"  Col Gram error  ||U11^H U11 + U21^H U21 - I_d||_F: {kin['gram_col_err']:.3e}")
        print(f"  Singular values of U11 (c_k)                    : {[round(float(x), 6) for x in kin['sv_U11']]}")
        print(f"  Singular values of A = U21 (s_k)                : {[round(float(x), 6) for x in kin['sv_U21'][:d]]}")
        print(f"  Singular values of B = U12 (s_k)                : {[round(float(x), 6) for x in kin['sv_U12'][:d]]}")
        print(f"  Forward leakage rank(A)                         : {kin['rank_A']}")
        print(f"  Reverse leakage rank(B)                         : {kin['rank_B']}")
        print(f"  Commutator rank([P_Lambda, U_nl]) = 2r          : {kin['rank_commutator']} (expected {2 * r})")
        print(f"  Commutator identity ||[P_Lambda, U_nl]-(B-A)||_F: {kin['comm_decomp_err']:.3e}")

        aligned_sweep, coord_sweep = run_depth_tradeoff_sweep(U_nl, P_Lambda, d=d, max_depth=N - d)
        print("\n  Sequential Depth-D Factorization Sweep (U_D = W_D ... W_1):")
        print(
            "  Depth D | rank((I-P)U_D P) | rank([P,U_D]) | Telescoping Err | Isometric Err ||(U_D-U)P||_F | Compression Err ||P(U_D-U)P||_F | Exact (D>=r)?"
        )
        print("  " + "-" * 132)
        for res_a, res_c in zip(aligned_sweep, coord_sweep):
            print(
                f"  {res_a.depth:7d} | {res_a.forward_leakage_rank:16d} | {res_a.commutator_rank:13d} | "
                f"{res_a.telescoping_commutator_error:15.3e} | {res_a.isometric_error:28.3e} | "
                f"{res_c.compression_error:31.3e} | {str(res_a.exact_isometric_success and res_c.exact_compression_success):>13s}"
            )

        adv = run_adversarial_falsification_check(U_nl, P_Lambda, d=d, num_random_trials=200)
        print(
            f"\n  Adversarial Falsification Check (200 random circuits at sub-critical depth D = r - 1 = {adv['tested_depth']}):"
        )
        print(f"    Max observed commutator rank : {adv['max_observed_comm_rank']} < 2r = {2 * r}")
        print(f"    Min observed isometric error : {adv['min_observed_iso_err']:.6e}")
        print(f"    Eckart-Young lower bound     : {adv['eckart_young_lower_bound']:.6e} (> 0)")

    print("\n--- Controlled Intermediate Leakage Rank Sweep r in {0, 1, ..., min(d, N-d)} ---")
    for target_r in range(0, min(d, N - d) + 1):
        for method in ("qr", "hamiltonian"):
            U_r = generate_nonlinear_unitary(
                N=N, d=d, method=method, target_rank=target_r, seed=args.seed + target_r
            )
            kin_r = verify_kinematic_commutator_identities(U_r, P_Lambda, d=d)
            assert kin_r["r"] == target_r
            assert kin_r["rank_commutator"] == 2 * target_r
            run_depth_tradeoff_sweep(U_r, P_Lambda, d=d, max_depth=N - d)
        print(
            f"  [VERIFIED] Target rank r = {target_r}: rank(A) = rank(B) = {target_r}, "
            f"rank([P_Lambda, U_nl]) = {2 * target_r}, exact reconstruction fails for D < {target_r} and succeeds for D >= {target_r}."
        )

    print("\n" + "=" * 88)
    print("ALL MILESTONE 1 VERIFICATION ASSERTIONS AND FALSIFICATION CHECKS PASSED.")
    print("=" * 88)


if __name__ == "__main__":
    main()
