# xorzen_ext.pyx — Cython Bridge: C++ Kernels → PyTorch Tensors
# ==============================================================
# Exposes all five C++ kernels to Python as zero-copy PyTorch operations.
# Each function validates tensor layout, extracts raw data pointers,
# calls the C++ kernel, and returns PyTorch tensors — no GIL held during
# the heavy computation.
#
# Build:
#   python setup_speed.py build_ext --inplace
#
# cython: language_level=3, boundscheck=False, wraparound=False

import torch
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

# ── Import C++ kernel declarations ────────────────────────────────
cdef extern from "csrc/xorzen_kernels.h" nogil:

    void ssm_parallel_scan_f32(
        const float* u, const float* A, float* out,
        int B, int T, int H, int N) nogil

    void window_attention_f32(
        const float* Q, const float* K, const float* V, float* out,
        int B, int H, int S, int D,
        int window, float scale) nogil

    void router_mlp_forward_f32(
        const float* x,
        const float* w1, const float* b1, int h1,
        const float* w2, const float* b2, int h2,
        const float* w3, const float* b3,
        float* out,
        int N, int in_dim, int out_dim) nogil

    void expert_capacity_constraint_f32(
        const int* indices, float* weights,
        int N, int top_k, int n_experts, int capacity) nogil

    void expert_weighted_sum_f32(
        const float* expert_out, const float* weights,
        const int* token_ids, float* dst,
        int n_active, int N, int hidden) nogil

    void fused_layernorm_gelu_f32(
        float* x, const float* gamma, const float* beta,
        int N, int D, float eps) nogil

    void fused_residual_add_f32(
        float* dst, const float* src, int n) nogil

    void gelu_f32_inplace(float* x, int n) nogil

    void softmax_f32_inplace(float* x, int rows, int cols) nogil

    # Extended kernels (fused_ops.cpp)
    void rms_norm_f32(
        const float* x, const float* gamma, float* out,
        int N, int D, float eps) nogil

    void fused_swiglu_f32(
        const float* gate, const float* up, float* out,
        int N, int D) nogil

    void diagonal_ssm_scan_f32(
        const float* B_seq, const float* A_diag, float* states,
        int batch, int T, int state) nogil


# ── helpers ───────────────────────────────────────────────────────

cdef float* _ptr_f32(tensor):
    """Return raw float32 data pointer from a contiguous CPU tensor."""
    assert tensor.is_contiguous(), "Tensor must be contiguous"
    assert tensor.dtype == torch.float32, "Tensor must be float32"
    assert not tensor.is_cuda, "C++ kernels run on CPU"
    return <float*><size_t>tensor.data_ptr()

cdef int* _ptr_i32(tensor):
    """Return raw int32 data pointer from a contiguous CPU tensor."""
    assert tensor.is_contiguous(), "Tensor must be contiguous"
    assert tensor.dtype == torch.int32, "Tensor must be int32"
    return <int*><size_t>tensor.data_ptr()


# ── Public Python API ─────────────────────────────────────────────

def cpp_ssm_scan(u, A):
    """
    Parallel prefix SSM scan.

    Parameters
    ----------
    u : torch.Tensor [B, T, H, N]  float32 CPU contiguous
        Input state increments.
    A : torch.Tensor [H, N]  float32 CPU contiguous
        Diagonal state transition (already discretised).

    Returns
    -------
    torch.Tensor [B, T, H, N]  float32
        Scanned (inclusive prefix sum) hidden states.
    """
    u   = u.contiguous().float()
    A   = A.contiguous().float()
    out = torch.empty_like(u)

    cdef int B = u.shape[0]
    cdef int T = u.shape[1]
    cdef int H = u.shape[2]
    cdef int N = u.shape[3]

    cdef float* pu  = _ptr_f32(u)
    cdef float* pA  = _ptr_f32(A)
    cdef float* po  = _ptr_f32(out)

    with nogil:
        ssm_parallel_scan_f32(pu, pA, po, B, T, H, N)

    return out


def cpp_window_attention(Q, K, V, int window=128):
    """
    Fused causal + window multi-head attention.

    Parameters
    ----------
    Q, K, V : torch.Tensor [B, H, S, D]  float32 CPU contiguous
    window  : int  local window size (0 = full causal)

    Returns
    -------
    torch.Tensor [B, H, S, D]  float32
    """
    Q = Q.contiguous().float()
    K = K.contiguous().float()
    V = V.contiguous().float()
    out = torch.empty_like(Q)

    cdef int B = Q.shape[0]
    cdef int H = Q.shape[1]
    cdef int S = Q.shape[2]
    cdef int D = Q.shape[3]
    cdef float scale = 1.0 / (D ** 0.5)

    cdef float* pQ = _ptr_f32(Q)
    cdef float* pK = _ptr_f32(K)
    cdef float* pV = _ptr_f32(V)
    cdef float* po = _ptr_f32(out)

    with nogil:
        window_attention_f32(pQ, pK, pV, po, B, H, S, D, window, scale)

    return out


def cpp_router_mlp(x, w1, b1, w2, b2, w3, b3):
    """
    Fused 3-layer router MLP.

    Parameters
    ----------
    x       : [N, in_dim]   float32 CPU
    w1, b1  : layer-1 weight [h1, in_dim] and bias [h1]
    w2, b2  : layer-2 weight [h2, h1] and bias [h2]
    w3, b3  : layer-3 weight [out_dim, h2] and bias [out_dim]

    Returns
    -------
    torch.Tensor [N, out_dim]  float32
    """
    x  = x.contiguous().float()
    w1 = w1.contiguous().float()
    b1 = b1.contiguous().float()
    w2 = w2.contiguous().float()
    b2 = b2.contiguous().float()
    w3 = w3.contiguous().float()
    b3 = b3.contiguous().float()

    cdef int N       = x.shape[0]
    cdef int in_dim  = x.shape[1]
    cdef int h1      = w1.shape[0]
    cdef int h2      = w2.shape[0]
    cdef int out_dim = w3.shape[0]

    out = torch.empty(N, out_dim, dtype=torch.float32)

    cdef float* px  = _ptr_f32(x)
    cdef float* pw1 = _ptr_f32(w1)
    cdef float* pb1 = _ptr_f32(b1)
    cdef float* pw2 = _ptr_f32(w2)
    cdef float* pb2 = _ptr_f32(b2)
    cdef float* pw3 = _ptr_f32(w3)
    cdef float* pb3 = _ptr_f32(b3)
    cdef float* po  = _ptr_f32(out)

    with nogil:
        router_mlp_forward_f32(
            px,
            pw1, pb1, h1,
            pw2, pb2, h2,
            pw3, pb3,
            po, N, in_dim, out_dim)

    return out


def cpp_expert_capacity(indices, weights, int n_experts, int capacity):
    """
    Apply capacity constraint to expert routing weights in-place.

    Parameters
    ----------
    indices  : torch.Tensor [N, top_k]  int32 CPU
    weights  : torch.Tensor [N, top_k]  float32 CPU  — MODIFIED IN PLACE
    n_experts: int
    capacity : int  max tokens per expert

    Returns
    -------
    weights (modified in-place)
    """
    indices = indices.contiguous().to(torch.int32)
    weights = weights.contiguous().float()

    cdef int N     = indices.shape[0]
    cdef int top_k = indices.shape[1]
    cdef int* pi   = _ptr_i32(indices)
    cdef float* pw = _ptr_f32(weights)

    with nogil:
        expert_capacity_constraint_f32(pi, pw, N, top_k, n_experts, capacity)

    return weights


def cpp_expert_weighted_sum(expert_out, weights, token_ids, int N, int hidden):
    """
    Accumulate weighted expert outputs.

    Parameters
    ----------
    expert_out : [n_active, hidden]  float32 CPU
    weights    : [n_active]          float32 CPU
    token_ids  : [n_active]          int32 CPU
    N          : total number of tokens
    hidden     : hidden dimension

    Returns
    -------
    torch.Tensor [N, hidden]  float32  (zeroed + accumulated)
    """
    expert_out = expert_out.contiguous().float()
    weights    = weights.contiguous().float()
    token_ids  = token_ids.contiguous().to(torch.int32)

    cdef int n_active = expert_out.shape[0]
    dst = torch.zeros(N, hidden, dtype=torch.float32)

    cdef float* pe = _ptr_f32(expert_out)
    cdef float* pw = _ptr_f32(weights)
    cdef int*   pt = _ptr_i32(token_ids)
    cdef float* pd = _ptr_f32(dst)

    with nogil:
        expert_weighted_sum_f32(pe, pw, pt, pd, n_active, N, hidden)

    return dst


def cpp_layernorm_gelu(x, gamma, beta, float eps=1e-5):
    """
    Fused LayerNorm → GELU in-place.

    Parameters
    ----------
    x     : [N, D]  float32 CPU — MODIFIED IN PLACE
    gamma : [D]     float32 CPU
    beta  : [D]     float32 CPU

    Returns
    -------
    x (modified in-place)
    """
    x     = x.contiguous().float()
    gamma = gamma.contiguous().float()
    beta  = beta.contiguous().float()

    cdef int N = x.shape[0]
    cdef int D = x.shape[1]
    cdef float* px = _ptr_f32(x)
    cdef float* pg = _ptr_f32(gamma)
    cdef float* pb = _ptr_f32(beta)

    with nogil:
        fused_layernorm_gelu_f32(px, pg, pb, N, D, eps)

    return x


def cpp_residual_add(dst, src):
    """dst += src  (in-place, AVX2)."""
    dst = dst.contiguous().float()
    src = src.contiguous().float()
    assert dst.numel() == src.numel()

    cdef float* pd = _ptr_f32(dst)
    cdef float* ps = _ptr_f32(src)
    cdef int n = dst.numel()

    with nogil:
        fused_residual_add_f32(pd, ps, n)

    return dst


def cpp_softmax_inplace(x):
    """Row-wise online softmax (in-place)."""
    x = x.contiguous().float()
    cdef int rows = x.shape[0]
    cdef int cols = x.shape[1] if x.dim() > 1 else 1
    cdef float* px = _ptr_f32(x)

    with nogil:
        softmax_f32_inplace(px, rows, cols)

    return x


# ── Extended kernels ──────────────────────────────────────────────

def cpp_rms_norm(x, gamma, float eps=1e-6):
    """
    RMSNorm: out = x / rms(x) * gamma   (no mean subtraction)

    Parameters
    ----------
    x     : [N, D]  float32 CPU
    gamma : [D]     float32 CPU

    Returns
    -------
    torch.Tensor [N, D]  float32
    """
    x     = x.contiguous().float()
    gamma = gamma.contiguous().float()
    out   = torch.empty_like(x)

    cdef int N = x.shape[0]
    cdef int D = x.shape[1]
    cdef float* px = _ptr_f32(x)
    cdef float* pg = _ptr_f32(gamma)
    cdef float* po = _ptr_f32(out)

    with nogil:
        rms_norm_f32(px, pg, po, N, D, eps)

    return out


def cpp_swiglu(gate, up):
    """
    Fused SwiGLU: out = silu(gate) * up

    Parameters
    ----------
    gate, up : [N, D]  float32 CPU

    Returns
    -------
    torch.Tensor [N, D]  float32
    """
    gate = gate.contiguous().float()
    up   = up.contiguous().float()
    out  = torch.empty_like(gate)

    cdef int N = gate.shape[0]
    cdef int D = gate.shape[1]
    cdef float* pg = _ptr_f32(gate)
    cdef float* pu = _ptr_f32(up)
    cdef float* po = _ptr_f32(out)

    with nogil:
        fused_swiglu_f32(pg, pu, po, N, D)

    return out


def cpp_diagonal_ssm_scan(B_seq, A_diag):
    """
    Efficient diagonal SSM scan: h_t = a * h_{t-1} + b_t

    Parameters
    ----------
    B_seq  : [batch, T, state]  float32 CPU contiguous
    A_diag : [state]            float32 CPU — diagonal of A_bar

    Returns
    -------
    torch.Tensor [batch, T, state]  float32
    """
    B_seq  = B_seq.contiguous().float()
    A_diag = A_diag.contiguous().float()
    states = torch.empty_like(B_seq)

    cdef int batch = B_seq.shape[0]
    cdef int T     = B_seq.shape[1]
    cdef int state = B_seq.shape[2]

    cdef float* pb = _ptr_f32(B_seq)
    cdef float* pa = _ptr_f32(A_diag)
    cdef float* ps = _ptr_f32(states)

    with nogil:
        diagonal_ssm_scan_f32(pb, pa, ps, batch, T, state)

    return states

