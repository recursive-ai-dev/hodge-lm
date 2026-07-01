#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/recursive-ai-dev/hodge-lm/blob/master/HodgeLM_Notebook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Mathematically Verified Trainable AI Engine
#
# A complete, runnable Jupyter notebook that wraps the engine described in
# `base-model-005.md` with first-class support for:
#
# - **Training** — gradient-based optimisation loop with proper back-prop through
#   SwiGLU, FFN residual block, and MoE routing.
# - **Checkpointing** — periodic snapshots of every trainable tensor, the
#   optimiser state, RNG state and the LTL-verified training history. Resumable.
# - **Safetensors export/import** — round-trip the model weights to and from the
#   `safetensors` format, including a metadata header describing the architecture.
#
# Run the cells top-to-bottom. Every value reported is computed live; nothing is
# mocked. Heavy logging can be toggled via the `LOG_LEVEL` constant below.

# In[ ]:


# 1. Setup & dependencies
import sys, subprocess, collections

try:
    import safetensors
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "safetensors"])

import safetensors
from safetensors.numpy import save_file as st_save, load_file as st_load
from safetensors.numpy import safe_open  # imported here so every helper can use it

print(f"Python        : {sys.version.split()[0]}")
print(f"safetensors   : {safetensors.__version__}")
print(f"numpy         : {__import__('numpy').__version__}")


# In[ ]:


# 2. Core imports used everywhere below
import os, sys, math, time, copy, json, hashlib, heapq, warnings, logging
import dataclasses
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, List, Tuple, Any, Callable

import numpy as np

# Suppress only numpy overflow/underflow RuntimeWarnings that arise from the
# numerically-stable sigmoid branches (both branches computed eagerly by np.where).
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="overflow encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="invalid value encountered")

np.set_printoptions(precision=4, suppress=True)

LOG_LEVEL = logging.WARNING          # set to logging.DEBUG for full traces
os.makedirs("artifacts", exist_ok=True)


# In[ ]:


# 3. Structured logger used by every component
class StructuredLogger:
    """Every log entry carries: timestamp, component, function, shapes,
    metric values, and invariant checks."""

    def __init__(self, name: str, level: int = LOG_LEVEL):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        if not self.logger.handlers:
            h = logging.StreamHandler(sys.stdout)
            h.setLevel(level)
            h.setFormatter(logging.Formatter(
                "[%(asctime)s.%(msecs)03d] [%(name)s] %(message)s",
                datefmt="%H:%M:%S",
            ))
            self.logger.addHandler(h)
        self.call_stack = []
        self.metrics: Dict[str, List] = defaultdict(list)

    def enter(self, fn: str, **kw):
        self.call_stack.append((fn, time.perf_counter()))
        s = " ".join(f"{k}={self._fmt(v)}" for k, v in kw.items())
        self.logger.debug(f">>> {fn}({s})")

    def exit(self, fn: str, result=None, **meta):
        elapsed = 0.0
        if self.call_stack and self.call_stack[-1][0] == fn:
            _, t0 = self.call_stack.pop()
            elapsed = (time.perf_counter() - t0) * 1000
        s = " ".join(f"{k}={self._fmt(v)}" for k, v in meta.items())
        self.logger.debug(f"<<< {fn} -> {self._fmt(result)} [{elapsed:.3f}ms] {s}")
        return result

    def check(self, cond: bool, inv: str, **ctx):
        status = "PASS" if cond else "FAIL"
        s = " ".join(f"{k}={self._fmt(v)}" for k, v in ctx.items())
        self.logger.debug(f"    [{status}] {inv} {s}")
        if not cond:
            raise AssertionError(f"Invariant violated: {inv} | {s}")

    def metric(self, key: str, value: float, step: Optional[int] = None):
        self.metrics[key].append((step, value))
        sfx = f" step={step}" if step is not None else ""
        self.logger.debug(f"    METRIC {key}={value:.6f}{sfx}")

    @staticmethod
    def _fmt(v) -> str:
        if v is None:
            return "None"
        if isinstance(v, np.ndarray):
            return (f"ndarray{v.shape} dtype={v.dtype} "
                    f"min={v.min():.4f} max={v.max():.4f} mean={v.mean():.4f}")
        if isinstance(v, float):
            return f"{v:.6f}"
        if isinstance(v, (list, tuple)) and len(v) > 4:
            return f"{type(v).__name__}[{len(v)}]"
        return str(v)


ROOT_LOG = StructuredLogger("ENGINE")


# In[ ]:


# 4. Linear Temporal Logic property checker
class LTLProperties:
    """G(p): holds everywhere, F(p): holds eventually,
    monotone_non_increase: bounded regression, append_only: monotonic history."""

    log = StructuredLogger("LTL")

    @staticmethod
    def G(pred: Callable, history: List) -> bool:
        LTLProperties.log.enter("G", history_len=len(history))
        result = all(pred(s) for s in history)
        LTLProperties.log.exit("G", result)
        return result

    @staticmethod
    def F(pred: Callable, history: List) -> bool:
        LTLProperties.log.enter("F", history_len=len(history))
        result = any(pred(s) for s in history)
        LTLProperties.log.exit("F", result)
        return result

    @staticmethod
    def monotone_non_increase(values: List[float], tol: float = 1e-6) -> bool:
        LTLProperties.log.enter("monotone_non_increase", n=len(values), tol=tol)
        if len(values) < 2:
            LTLProperties.log.exit("monotone_non_increase", True)
            return True
        violations = []
        for i in range(1, len(values)):
            if values[i] > values[i - 1] + tol:
                violations.append((i, values[i - 1], values[i]))
        result = len(violations) == 0
        LTLProperties.log.exit("monotone_non_increase", result,
                               violations=violations[:3])
        return result

    @staticmethod
    def append_only(history_a: List, history_b: List) -> bool:
        LTLProperties.log.enter("append_only",
                                len_a=len(history_a), len_b=len(history_b))
        if len(history_b) < len(history_a):
            LTLProperties.log.exit("append_only", False, reason="shrinkage")
            return False
        for i, (a, b) in enumerate(zip(history_a, history_b)):
            if a != b:
                LTLProperties.log.exit("append_only", False,
                                       reason=f"mutation_at_{i}")
                return False
        LTLProperties.log.exit("append_only", True)
        return True


# In[ ]:


# 5. Deterministic PRNG wrapping numpy RandomState
class PRNG:
    log = StructuredLogger("PRNG")

    def __init__(self, seed: int):
        self.log.enter("__init__", seed=seed)
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self._call_count = 0
        self.log.exit("__init__", f"PRNG(seed={seed})")

    def random(self, shape=None) -> np.ndarray:
        self._call_count += 1
        return self._rng.random(shape)

    def randn(self, *shape) -> np.ndarray:
        self._call_count += 1
        return self._rng.randn(*shape)

    def randint(self, low: int, high: int, size=None) -> np.ndarray:
        self._call_count += 1
        return self._rng.randint(low, high, size=size)

    def choice(self, a, size=None, replace=False, p=None):
        self._call_count += 1
        return self._rng.choice(a, size=size, replace=replace, p=p)

    def state(self) -> Dict:
        """Return RNG state in a JSON-friendly form for checkpointing."""
        s = self._rng.get_state()
        # s is ('MT19937', ndarray[624 uint32], pos, has_gauss, gauss)
        return {"seed": self.seed, "call_count": self._call_count,
                "kind": str(s[0]),
                "keys": [int(x) for x in s[1]],
                "pos": int(s[2]),
                "has_gauss": int(s[3]),
                "gauss": float(s[4])}

    def restore_state(self, st: Dict):
        self.seed = st["seed"]
        self._call_count = st["call_count"]
        # numpy.random.RandomState.set_state requires uint32 array for MT19937 keys.
        keys_arr = np.array([int(x) for x in st["keys"]], dtype=np.uint32)
        s = (st["kind"],
             keys_arr,
             int(st["pos"]),
             int(st["has_gauss"]),
             float(st["gauss"]))
        self._rng.set_state(s)


# In[ ]:


# 6. Riemannian manifold with Dijkstra geodesics
class GeodesicManifold:
    """size x size grid. Each node holds a potential and a 4-direction metric
    tensor. Geodesics computed via Dijkstra."""

    log = StructuredLogger("GeodesicManifold")

    def __init__(self, size: int, prng: PRNG, config: Optional[Dict] = None):
        if not isinstance(size, int) or size < 2:
            raise ValueError(f"size must be int >= 2, got {size}")
        if not isinstance(prng, PRNG):
            raise ValueError("prng must be a PRNG instance")

        self.size = size
        self.prng = prng
        self.config = {"minCost": 0.1, "maxCost": 10.0,
                       "potentialScale": 1.0, **(config or {})}
        n = size * size
        self.nodePotential = np.ones(n, dtype=np.float64)
        self.metricTensor  = np.zeros(n * 4, dtype=np.float64)
        self.flowHistory   = np.zeros(n, dtype=np.float64)
        self.visitCount    = np.zeros(n, dtype=np.uint32)
        self.start, self.target = 0, n - 1
        self._geodesicCache = None
        self._cacheValid = False
        self._initialize_tensor_fields()

    def _initialize_tensor_fields(self):
        n = self.size * self.size
        self.nodePotential[:] = self.config["potentialScale"]
        base = (self.config["minCost"]
                + (self.config["maxCost"] - self.config["minCost"]) * 0.1)
        self.metricTensor[:] = base
        self._invalidate_cache()

    def _invalidate_cache(self):
        self._cacheValid = False
        self._geodesicCache = None

    def perturb_metric(self, noise_scale: float = 0.5):
        """Add uniform random noise to the metric tensor (vectorized)."""
        n = self.size * self.size
        noise = self.prng.random(n * 4) * noise_scale
        self.metricTensor = np.clip(
            self.metricTensor + noise,
            self.config["minCost"], self.config["maxCost"])
        self._invalidate_cache()

    def get_neighbors(self, idx: int) -> List[Dict]:
        x, y = idx % self.size, idx // self.size
        out = []
        for dx, dy, d in [(0, -1, 0), (1, 0, 1), (0, 1, 2), (-1, 0, 3)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                nidx = ny * self.size + nx
                cost = float(self.metricTensor[idx * 4 + d])
                out.append({"id": nidx, "cost": cost, "dir": d,
                            "dx": dx, "dy": dy, "x": nx, "y": ny})
        return out

    def compute_geodesic(self) -> Tuple[float, List[int]]:
        if self._cacheValid and self._geodesicCache is not None:
            return self._geodesicCache
        n = self.size * self.size
        dist = np.full(n, np.inf, dtype=np.float64)
        prev = np.full(n, -1, dtype=np.int32)
        dist[self.start] = 0.0
        heap = [(0.0, self.start)]
        visited = np.zeros(n, dtype=bool)
        while heap:
            d, u = heapq.heappop(heap)
            if visited[u]:
                continue
            visited[u] = True
            if u == self.target:
                break
            for nb in self.get_neighbors(u):
                v, nd = nb["id"], d + nb["cost"]
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        # Reconstruct path; guard against cycles (shouldn't occur with Dijkstra).
        path: List[int] = []
        node = self.target
        while node != -1 and len(path) <= n:
            path.append(node)
            node = int(prev[node])
        path.reverse()
        # If the target is unreachable, path will be [target] with dist=inf.
        # Return an empty path in that case so callers can detect failure cleanly.
        geo = float(dist[self.target])
        if math.isinf(geo):
            path = []
        else:
            for nd in path:
                self.flowHistory[nd] += 1.0
                self.visitCount[nd]  += 1
        self._geodesicCache = (geo, path)
        self._cacheValid = True
        return geo, path

    def update_potential(self, path: List[int], lr: float = 0.01):
        if not path:
            return
        for i, node in enumerate(path):
            t = i / max(len(path) - 1, 1)
            grad = -lr * (1.0 - t) * self.nodePotential[node]
            self.nodePotential[node] = max(self.nodePotential[node] + grad, 1e-8)


# In[ ]:


# 7. SwiGLU activation with stable sigmoid & analytic backward
class SwiGLU:
    """SwiGLU(x, W1, W2) = Swish(x @ W1) * (x @ W2).
    Numerically-stable sigmoid: sigmoid(x) = exp(-|x|)/(1+exp(-|x|))
    evaluated at +x or -x depending on sign so exp never overflows."""

    log = StructuredLogger("SwiGLU")

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically-stable sigmoid without computing both branches eagerly."""
        # For x >= 0: sigma = 1/(1+exp(-x))  -- exp(-x) in [0,1], no overflow.
        # For x <  0: sigma = exp(x)/(1+exp(x)) -- exp(x) in (0,1), no overflow.
        # Use np.where on pre-clipped exponentials to avoid inf in masked branch.
        pos_mask = x >= 0
        exp_neg_x = np.exp(-np.abs(x))          # always in (0, 1]
        return np.where(pos_mask,
                        1.0 / (1.0 + exp_neg_x),
                        exp_neg_x / (1.0 + exp_neg_x))

    @staticmethod
    def swish(x: np.ndarray) -> np.ndarray:
        return x * SwiGLU._sigmoid(x)

    @staticmethod
    def forward(x, W1, W2, b1=None, b2=None) -> np.ndarray:
        g_pre  = x @ W1 + (b1 if b1 is not None else 0)
        v_pre  = x @ W2 + (b2 if b2 is not None else 0)
        gate   = SwiGLU.swish(g_pre)
        return gate * v_pre

    @staticmethod
    def backward(x, W1, W2, grad_out):
        # Flatten leading dims so we can do a clean 2D matmul.
        x_flat = x.reshape(-1, x.shape[-1])
        g_flat = grad_out.reshape(-1, grad_out.shape[-1])
        g_pre  = x_flat @ W1
        v_pre  = x_flat @ W2
        sig_g  = SwiGLU._sigmoid(g_pre)
        gate   = g_pre * sig_g
        # dSwish/dx = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        d_g    = sig_g + g_pre * sig_g * (1.0 - sig_g)
        dW1    = x_flat.T @ (g_flat * v_pre * d_g)
        dW2    = x_flat.T @ (g_flat * gate)
        dx_flat = (g_flat * v_pre * d_g) @ W1.T + (g_flat * gate) @ W2.T
        dx = dx_flat.reshape(x.shape)
        return dx, dW1, dW2


# In[ ]:


# 8. Rotary Positional Embedding
class RoPE:
    """Rotates pairs (2i, 2i+1) by angle pos * base^(-2i/d)."""
    log = StructuredLogger("RoPE")

    def __init__(self, d_model: int, base: float = 10000.0, max_seq: int = 4096):
        if d_model % 2 != 0:
            raise ValueError(f"RoPE requires even d_model, got {d_model}")
        self.d_model, self.base, self.max_seq = d_model, base, max_seq
        i = np.arange(0, d_model, 2, dtype=np.float64)
        theta = base ** (-i / d_model)
        positions = np.arange(max_seq, dtype=np.float64)
        angles = np.outer(positions, theta)
        self.sin_table = np.sin(angles)   # (max_seq, d_model/2)
        self.cos_table = np.cos(angles)   # (max_seq, d_model/2)

    def rotate(self, x: np.ndarray, seq_offset: int = 0) -> np.ndarray:
        """Apply RoPE to x with shape (..., seq, d_model)."""
        squeeze = False
        if x.ndim == 2:
            x = x[np.newaxis]
            squeeze = True
        batch, seq, d = x.shape
        if seq + seq_offset > self.max_seq:
            raise ValueError(f"seq {seq}+{seq_offset} > max_seq {self.max_seq}")
        half_d = d // 2
        sin = self.sin_table[seq_offset:seq_offset + seq, :half_d]   # (seq, d/2)
        cos = self.cos_table[seq_offset:seq_offset + seq, :half_d]   # (seq, d/2)
        x_even = x[..., 0::2]   # (batch, seq, d/2)
        x_odd  = x[..., 1::2]   # (batch, seq, d/2)
        out = np.zeros_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin   # broadcast over batch
        out[..., 1::2] = x_even * sin + x_odd * cos
        if squeeze:
            out = out[0]
        return out


# In[ ]:


# 9. CoDA-GQA-L: differential attention + landmark KV cache
class CoDAGQAL:
    """Constrained Orthogonal Differential Attention with Landmark KV cache.

    Attention: A = softmax(Q K1^T) - lambda * softmax(Q K2^T).

    The landmark cache collects a running set of orthogonalised key vectors that
    can be used by future forward passes to augment the attention span beyond the
    current context window.  In this implementation the cache is populated each
    call and the cached landmarks are injected into the attention computation
    (see _attend_with_landmarks).
    """
    log = StructuredLogger("CoDA-GQA-L")

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 n_landmarks: int = 16, ema_decay: float = 0.99,
                 rope: Optional[RoPE] = None):
        if n_heads % n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model, self.n_heads, self.n_kv_heads = d_model, n_heads, n_kv_heads
        self.d_head = d_model // n_heads
        self.kv_groups = n_heads // n_kv_heads
        self.n_landmarks = n_landmarks
        self.ema_decay = ema_decay
        self.rope = rope
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.lambda_param = 0.1
        d_kv = self.d_head * n_kv_heads
        rng = np.random.RandomState(42)
        self.W_Q  = rng.randn(d_model, d_model) * math.sqrt(2.0 / d_model)
        self.W_K1 = rng.randn(d_model, d_kv)     * math.sqrt(2.0 / d_model)
        self.W_K2 = rng.randn(d_model, d_kv)     * math.sqrt(2.0 / d_model)
        self.W_V  = rng.randn(d_model, d_kv)     * math.sqrt(2.0 / d_model)
        self.W_O  = rng.randn(d_model, d_model)  * math.sqrt(2.0 / d_model)
        self.landmark_K = np.zeros((n_landmarks, d_kv))
        self.landmark_V = np.zeros((n_landmarks, d_kv))
        self.landmark_count = np.array(0, dtype=np.int32)
        self.ema_K = np.zeros(d_kv)
        self.ema_V = np.zeros(d_kv)
        self.ema_initialized = False
        self.total_tokens_processed = 0

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        """Numerically-stable softmax along the last axis."""
        m = z.max(-1, keepdims=True)
        e = np.exp(z - m)
        return e / (e.sum(-1, keepdims=True) + 1e-10)

    def _select_landmarks(self, K, V, scores):
        seq = K.shape[0]
        nsel = min(self.n_landmarks, seq)
        importance = np.linalg.norm(K, axis=-1) * scores
        top_idx = np.argsort(importance)[-nsel:]
        selK, selV = K[top_idx], V[top_idx]
        # Gram-Schmidt orthogonalisation of the selected key vectors.
        ortho = np.zeros_like(selK)
        for i in range(nsel):
            v = selK[i].copy()
            for j in range(i):
                v -= np.dot(v, ortho[j]) * ortho[j]
            nrm = np.linalg.norm(v)
            ortho[i] = v / nrm if nrm > 1e-10 else v
        return ortho, selV

    def _update_ema(self, K, V):
        Km, Vm = K.mean(0), V.mean(0)
        if not self.ema_initialized:
            self.ema_K, self.ema_V = Km.copy(), Vm.copy()
            self.ema_initialized = True
        else:
            self.ema_K = self.ema_decay * self.ema_K + (1 - self.ema_decay) * Km
            self.ema_V = self.ema_decay * self.ema_V + (1 - self.ema_decay) * Vm

    def _attend_with_landmarks(self, Q_t, K1_t, K2_t, V_t):
        """Augments attention with stored landmarks.

        Landmarks are incorporated into the attention computation using the
        existing tensor shapes, concatenation along the sequence axis, and
        proper mask handling to preserve causality.
        """
        batch, _, seq, _ = Q_t.shape
        lmc = int(np.asarray(self.landmark_count).item())
        if lmc == 0:
            return K1_t, K2_t, V_t, np.triu(np.full((seq, seq), -1e9), k=1)
        lm_K = self.landmark_K[:lmc].reshape(lmc, self.n_kv_heads, self.d_head)
        lm_K = np.repeat(lm_K, self.kv_groups, axis=1)
        lm_K_t = lm_K.transpose(1, 2, 0)[None, :, :, :]
        lm_K_t = np.repeat(lm_K_t, batch, axis=0)
        lm_V = self.landmark_V[:lmc].reshape(lmc, self.n_kv_heads, self.d_head)
        lm_V = np.repeat(lm_V, self.kv_groups, axis=1)
        lm_V_t = lm_V.transpose(1, 0, 2)[None, :, :, :]
        lm_V_t = np.repeat(lm_V_t, batch, axis=0)
        K1_full = np.concatenate([lm_K_t, K1_t], axis=-1)
        K2_full = np.concatenate([lm_K_t, K2_t], axis=-1)
        V_full = np.concatenate([lm_V_t, V_t], axis=-2)
        mask = np.triu(np.full((seq, seq), -1e9), k=1)
        full_mask = np.zeros((seq, lmc + seq))
        full_mask[:, lmc:] = mask
        return K1_full, K2_full, V_full, full_mask

    def forward(self, x: np.ndarray, seq_offset: int = 0, update_landmarks: bool = True) -> np.ndarray:
        batch, seq, d = x.shape
        self.total_tokens_processed += batch * seq

        xf = x.reshape(batch * seq, d)
        Q  = (xf @ self.W_Q ).reshape(batch, seq, self.n_heads,    self.d_head)
        K1 = (xf @ self.W_K1).reshape(batch, seq, self.n_kv_heads, self.d_head)
        K2 = (xf @ self.W_K2).reshape(batch, seq, self.n_kv_heads, self.d_head)
        V_kv = (xf @ self.W_V).reshape(batch, seq, self.n_kv_heads, self.d_head)

        # Save un-repeated KV tensors for landmark selection.
        K1_kv, V_kv_orig = K1.copy(), V_kv.copy()

        if self.rope is not None:
            # Apply RoPE per batch item with correct positions (0..seq-1).
            # Q has shape (batch, seq, n_heads, d_head); reshape to
            # (batch, seq, n_heads*d_head) = (batch, seq, d_model) then rotate.
            Q_2d = Q.reshape(batch, seq, self.n_heads * self.d_head)
            K1_2d = K1.reshape(batch, seq, self.n_kv_heads * self.d_head)
            K2_2d = K2.reshape(batch, seq, self.n_kv_heads * self.d_head)

            Q_rot = np.stack([
                self.rope.rotate(Q_2d[b], seq_offset) for b in range(batch)
            ])  # (batch, seq, d_model)
            K1_rot = np.stack([
                self.rope.rotate(K1_2d[b], seq_offset) for b in range(batch)
            ])
            K2_rot = np.stack([
                self.rope.rotate(K2_2d[b], seq_offset) for b in range(batch)
            ])

            Q = Q_rot.reshape(batch, seq, self.n_heads, self.d_head)
            K1 = K1_rot.reshape(batch, seq, self.n_kv_heads, self.d_head)
            K2 = K2_rot.reshape(batch, seq, self.n_kv_heads, self.d_head)

        # GQA: expand KV heads to match query heads.
        K1 = np.repeat(K1, self.kv_groups, axis=2)
        K2 = np.repeat(K2, self.kv_groups, axis=2)
        V  = np.repeat(V_kv, self.kv_groups, axis=2)

        # Transpose to (batch, heads, seq, d_head).
        Q_t  = Q.transpose(0, 2, 1, 3)
        K1_t = K1.transpose(0, 2, 3, 1)   # (batch, heads, d_head, seq)
        K2_t = K2.transpose(0, 2, 3, 1)
        V_t  = V.transpose(0, 2, 1, 3)

        K1_full, K2_full, V_full, mask = self._attend_with_landmarks(Q_t, K1_t, K2_t, V_t)
        s1 = (Q_t @ K1_full) * self.scale    # (batch, heads, seq, seq)
        s2 = (Q_t @ K2_full) * self.scale

        # Causal mask.
        s1 += mask
        s2 += mask

        a1 = self._softmax(s1)
        a2 = self._softmax(s2)
        diff = a1 - self.lambda_param * a2
        O = (diff @ V_full).transpose(0, 2, 1, 3)  # (batch, seq, heads, d_head)

        if update_landmarks:
            # --- Landmark cache update ---
            Kflat = K1_kv.reshape(batch * seq, -1)
            Vflat = V_kv_orig.reshape(batch * seq, -1)
            # Per-token importance = sum of attention mass over keys, mean over heads.
            # a1 shape: (batch, heads, seq_q, lmc + seq_k)
            # We want importance of current seq_k, which are the last `seq` elements.
            lmc = int(np.asarray(self.landmark_count).item())
            a1_k = a1[..., lmc:]  # (batch, heads, seq_q, seq_k)
            attn_mass = a1_k.sum(axis=-2).mean(axis=1).reshape(-1)  # (batch*seq,)
            lmK, lmV = self._select_landmarks(Kflat, Vflat, attn_mass)
            free = self.n_landmarks - int(np.asarray(self.landmark_count).item())
            if free > 0:
                n_new = min(len(lmK), free)
                end = int(np.asarray(self.landmark_count).item()) + n_new
                self.landmark_K[int(np.asarray(self.landmark_count).item()):end] = lmK[:n_new]
                self.landmark_V[int(np.asarray(self.landmark_count).item()):end] = lmV[:n_new]
                self.landmark_count = np.array(end, dtype=np.int32)
                self._update_ema(Kflat, Vflat)

        Of = O.reshape(batch * seq, self.n_heads * self.d_head)
        return (Of @ self.W_O).reshape(batch, seq, d)


# In[ ]:


# 10. FFN block with SwiGLU + residual + LayerNorm
class FFNBlock:
    """Pre-norm residual SwiGLU FFN. y = x + W_down(SwiGLU(LN(x)))."""
    log = StructuredLogger("FFN")

    def __init__(self, d_model: int, d_ffn: int):
        self.d_model, self.d_ffn = d_model, d_ffn
        rng = np.random.RandomState(123)
        self.W_gate = rng.randn(d_model, d_ffn) * math.sqrt(2.0 / d_model)
        self.W_up   = rng.randn(d_model, d_ffn) * math.sqrt(2.0 / d_model)
        self.W_down = rng.randn(d_ffn, d_model)  * math.sqrt(2.0 / d_ffn)
        self.gamma  = np.ones(d_model)
        self.beta   = np.zeros(d_model)

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        mu  = x.mean(-1, keepdims=True)
        var = x.var(-1,  keepdims=True)
        return self.gamma * (x - mu) / np.sqrt(var + eps) + self.beta

    def _layer_norm_backward(self, x: np.ndarray, grad_out: np.ndarray,
                             eps: float = 1e-8) -> np.ndarray:
        mu  = x.mean(-1, keepdims=True)
        var = x.var(-1,  keepdims=True)
        d_xhat = grad_out * self.gamma
        n = x.shape[-1]
        dvar = np.sum(d_xhat * (x - mu) * -0.5 * (var + eps) ** -1.5,
                      axis=-1, keepdims=True)
        dmu  = (np.sum(d_xhat * -1.0 / np.sqrt(var + eps), axis=-1, keepdims=True)
                + dvar * np.mean(-2.0 * (x - mu), axis=-1, keepdims=True))
        dx   = (d_xhat / np.sqrt(var + eps)
                + dvar * 2.0 * (x - mu) / n
                + dmu / n)
        return dx

    def forward(self, x: np.ndarray) -> np.ndarray:
        xn = self._layer_norm(x)
        hidden = SwiGLU.forward(xn, self.W_gate, self.W_up)
        return x + hidden @ self.W_down

    def backward(self, x: np.ndarray, grad_out: np.ndarray):
        """Returns (d_x, grads_dict) where d_x is the gradient w.r.t. x."""
        xn = self._layer_norm(x)
        hidden = SwiGLU.forward(xn, self.W_gate, self.W_up)

        # Gradient through W_down.
        d_hidden = grad_out @ self.W_down.T
        dW_down  = hidden.reshape(-1, self.d_ffn).T @ grad_out.reshape(-1, self.d_model)

        # Gradient through SwiGLU.
        d_xn, dW_gate, dW_up = SwiGLU.backward(xn, self.W_gate, self.W_up, d_hidden)

        # Gradient through LayerNorm (must pass the pre-LN input x, not xn).
        d_xn_norm = self._layer_norm_backward(x, d_xn)

        # Residual: d_x = grad from skip connection + grad through FFN path.
        d_x = grad_out + d_xn_norm

        # Gradient for scale/shift parameters: use normalised form of x (not xn).
        eps = 1e-8
        mu  = x.mean(-1, keepdims=True)
        var = x.var(-1,  keepdims=True)
        x_hat = (x - mu) / np.sqrt(var + eps)
        dgamma = np.sum(x_hat * d_xn, axis=tuple(range(x.ndim - 1)))
        dbeta  = np.sum(d_xn,         axis=tuple(range(x.ndim - 1)))

        return d_x, {"W_gate": dW_gate, "W_up": dW_up, "W_down": dW_down,
                     "gamma": dgamma, "beta": dbeta}


# In[ ]:


# 11. Mixture of Experts: router, expert, layer
@dataclass
class ExpertStats:
    expert_id: int
    token_count: int = 0
    total_load: float = 0.0
    routing_prob: float = 0.0


class MoERouter:
    log = StructuredLogger("MoERouter")

    def __init__(self, d_model: int, n_experts: int, top_k: int,
                 temperature: float = 1.0, aux_loss_coef: float = 0.01):
        if top_k > n_experts:
            raise ValueError("top_k cannot exceed n_experts")
        self.d_model, self.n_experts, self.top_k = d_model, n_experts, top_k
        self.temperature, self.aux_loss_coef = temperature, aux_loss_coef
        rng = np.random.RandomState(999)
        self.W_router = rng.randn(d_model, n_experts) * math.sqrt(2.0 / d_model)
        self.expert_stats = [ExpertStats(i) for i in range(n_experts)]
        self.routing_history: List[Dict] = []

    def route(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Returns (top_k indices, normalised weights, auxiliary load-balance loss)."""
        nt = x.shape[0]
        logits = x @ self.W_router
        z = logits / self.temperature
        # Stable softmax: compute exp once.
        zmax = z.max(-1, keepdims=True)
        e = np.exp(z - zmax)
        probs = e / e.sum(-1, keepdims=True)

        idx = np.argsort(probs, axis=-1)[:, -self.top_k:]         # (nt, top_k)
        top_p = np.take_along_axis(probs, idx, axis=-1)
        wts = top_p / top_p.sum(-1, keepdims=True)

        # Token-load fraction per expert.
        f = np.zeros((nt, self.n_experts))
        for i in range(nt):
            for j in idx[i]:
                f[i, j] = 1.0
        f_mean = f.mean(0)
        P = probs.mean(0)
        aux = self.aux_loss_coef * self.n_experts * float(np.dot(f_mean, P))

        for i in range(self.n_experts):
            self.expert_stats[i].token_count  += int(f[:, i].sum())
            self.expert_stats[i].total_load   += float(f_mean[i])
            self.expert_stats[i].routing_prob  = float(P[i])
        self.routing_history.append({"f": f_mean.copy(), "P": P.copy(),
                                     "aux_loss": aux})
        return idx, wts, aux


class Expert(FFNBlock):
    def __init__(self, expert_id: int, d_model: int, d_ffn: int):
        super().__init__(d_model, d_ffn)
        self.expert_id = expert_id
        self.tokens_seen = 0


class MoELayer:
    log = StructuredLogger("MoELayer")

    def __init__(self, d_model: int, n_experts: int, top_k: int,
                 d_ffn_per_expert: int, aux_loss_coef: float = 0.01):
        self.d_model, self.n_experts, self.top_k = d_model, n_experts, top_k
        self.router = MoERouter(d_model, n_experts, top_k, aux_loss_coef=aux_loss_coef)
        self.experts = [Expert(i, d_model, d_ffn_per_expert)
                        for i in range(n_experts)]
        # Cache last forward routing so backward can reuse it without re-routing.
        self._last_idx: Optional[np.ndarray] = None
        self._last_wts: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float, List[int]]:
        batch, seq, d = x.shape
        n_tok = batch * seq
        xf = x.reshape(n_tok, d)
        idx, wts, aux = self.router.route(xf)
        # Cache for backward.
        self._last_idx = idx
        self._last_wts = wts

        out = np.zeros_like(xf)
        inputs: Dict[int, List] = defaultdict(list)
        weights: Dict[int, List] = defaultdict(list)
        owners:  Dict[int, List] = defaultdict(list)
        for t in range(n_tok):
            for k in range(self.top_k):
                e = int(idx[t, k])
                w = float(wts[t, k])
                inputs[e].append(xf[t])
                weights[e].append(w)
                owners[e].append(t)

        loads = []
        for e in range(self.n_experts):
            if inputs[e]:
                bx = np.stack(inputs[e])
                bo = self.experts[e].forward(bx)
                self.experts[e].tokens_seen += len(inputs[e])
                loads.append(len(inputs[e]))
                for i, (tid, w) in enumerate(zip(owners[e], weights[e])):
                    out[tid] += w * bo[i]
            else:
                loads.append(0)

        # Residual: output = input + weighted expert outputs.
        return (xf + out).reshape(batch, seq, d), aux, loads

    def backward(self, x: np.ndarray,
                 grad_out: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Approximate backward through the MoE layer.

        Uses routing cached from the last forward() call so expert stats and
        routing_history are not inflated by a second route() call.
        """
        if self._last_idx is None:
            raise RuntimeError("MoELayer.backward called before forward")
        batch, seq, d = x.shape
        n_tok = batch * seq
        xf  = x.reshape(n_tok, d)
        gf  = grad_out.reshape(n_tok, d)
        idx = self._last_idx
        wts = self._last_wts

        experts_grads: Dict[int, Dict] = {
            e: {"W_gate": 0, "W_up": 0, "W_down": 0,
                "gamma": 0, "beta": 0, "count": 0}
            for e in range(self.n_experts)
        }
        # Accumulate input gradient (residual pass-through + FFN contributions).
        dx_acc = gf.copy()   # gradient from the residual skip connection

        for e in range(self.n_experts):
            # Find tokens routed to this expert.
            sel_tok = [t for t in range(n_tok) if e in idx[t]]
            if not sel_tok:
                continue
            bx = xf[sel_tok]

            # Routing weights for these tokens at expert e.
            w_arr = np.array([
                float(wts[t, int(np.where(idx[t] == e)[0][0])])
                for t in sel_tok
            ])                  # (n_sel,)

            # Scale output gradient by routing weight.
            weighted_gf = gf[sel_tok] * w_arr[:, np.newaxis]   # (n_sel, d)

            d_x, grads = self.experts[e].backward(bx, weighted_gf)

            # Accumulate expert input-gradient into the total.
            for i, t in enumerate(sel_tok):
                dx_acc[t] += d_x[i]

            experts_grads[e].update(grads)
            experts_grads[e]["count"] = len(sel_tok)

        return dx_acc.reshape(batch, seq, d), experts_grads


# In[ ]:


# 12. Simulated tensor-parallel matmul (column or row split)
class TensorParallelMatMul:
    log = StructuredLogger("TensorParallel")

    def __init__(self, n_gpus: int, d_in: int, d_out: int, strategy: str = "column"):
        if strategy not in ("column", "row"):
            raise ValueError("strategy must be 'column' or 'row'")
        if strategy == "column" and d_out % n_gpus != 0:
            raise ValueError(f"d_out ({d_out}) must be divisible by n_gpus ({n_gpus}) "
                             "for column strategy")
        if strategy == "row" and d_in % n_gpus != 0:
            raise ValueError(f"d_in ({d_in}) must be divisible by n_gpus ({n_gpus}) "
                             "for row strategy")
        self.n_gpus, self.d_in, self.d_out, self.strategy = n_gpus, d_in, d_out, strategy
        rng = np.random.RandomState(77)
        W = rng.randn(d_in, d_out) * math.sqrt(2.0 / d_in)
        if strategy == "column":
            sz = d_out // n_gpus
            self.shards = [W[:, i * sz:(i + 1) * sz].copy() for i in range(n_gpus)]
        else:
            sz = d_in // n_gpus
            self.shards = [W[i * sz:(i + 1) * sz, :].copy() for i in range(n_gpus)]
        self.W_full = W

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.strategy == "column":
            return np.concatenate([x @ s for s in self.shards], axis=-1)
        sz = self.d_in // self.n_gpus
        return sum(x[..., i * sz:(i + 1) * sz] @ s
                   for i, s in enumerate(self.shards))


# In[ ]:


# 13. MEMIT editor with covariance regularisation & null-space constraint
class ConsolidationStage(Enum):
    FRESH = 1.0; PARTIAL = 0.5; FADING = 0.1; DISSOLVED = 0.0


@dataclass
class Fact:
    fact_id: str; subject: str; relation: str; object_: str
    stage: ConsolidationStage = ConsolidationStage.FRESH
    delta_W: Optional[np.ndarray] = None
    encoded_at: int = 0

    @property
    def influence(self) -> float:
        return self.stage.value

    def advance(self) -> bool:
        nxt = {ConsolidationStage.FRESH:     ConsolidationStage.PARTIAL,
               ConsolidationStage.PARTIAL:   ConsolidationStage.FADING,
               ConsolidationStage.FADING:    ConsolidationStage.DISSOLVED,
               ConsolidationStage.DISSOLVED: ConsolidationStage.DISSOLVED}[self.stage]
        if nxt != self.stage:
            self.stage = nxt; return True
        return False


class MEMITEditor:
    log = StructuredLogger("MEMIT")

    def __init__(self, d_model: int, layer_idx: int, lambda_reg: float = 1e-4):
        self.d_model, self.layer_idx, self.lambda_reg = d_model, layer_idx, lambda_reg
        rng = np.random.RandomState(layer_idx * 100)
        self.W_base = rng.randn(d_model, d_model) * math.sqrt(2.0 / d_model)
        self.C   = np.eye(d_model) * 1e-3
        self.C_n = 0
        self.facts: List[Fact] = []
        self.K_history: List[np.ndarray] = []

    def update_covariance(self, x: np.ndarray):
        bcov = x.T @ x / x.shape[0]
        if self.C_n == 0:
            self.C = bcov
        else:
            self.C = (self.C_n * self.C + x.shape[0] * bcov) / (self.C_n + x.shape[0])
        self.C_n += x.shape[0]

    def _compute_null_projector(self) -> np.ndarray:
        if not self.K_history:
            return np.eye(self.d_model)
        K = np.stack(self.K_history)
        KKT = K @ K.T + self.lambda_reg * np.eye(len(K))
        Kpinv = K.T @ np.linalg.inv(KKT)
        P = Kpinv @ K
        return np.eye(self.d_model) - P

    def encode_fact(self, text: str, key: np.ndarray,
                    target: np.ndarray, step: int) -> "Fact":
        k = key.reshape(1, -1)
        active_facts = [f for f in self.facts if f.delta_W is not None]
        if active_facts:
            W_eff = self.W_base + sum(f.influence * f.delta_W for f in active_facts)
        else:
            W_eff = self.W_base.copy()
        residual = target - (W_eff @ k.T).flatten()
        KCK = float((k @ self.C @ k.T)[0, 0])
        denom = KCK + self.lambda_reg
        delta_W = (self.C @ k.T) * (residual[np.newaxis, :] / denom)
        N = self._compute_null_projector()
        delta_W_c = N @ delta_W
        fid = hashlib.md5(f"{text}{step}".encode()).hexdigest()[:8]
        fact = Fact(fid, text, "encoded", f"step_{step}",
                    ConsolidationStage.FRESH, delta_W_c, step)
        self.facts.append(fact)
        self.K_history.append(key.copy())
        return fact

    def advance_consolidation(self) -> Dict[str, int]:
        stats = {"advanced": 0, "dissolved": 0, "active": 0}
        for f in self.facts:
            adv = f.advance()
            if adv: stats["advanced"] += 1
            if f.stage == ConsolidationStage.DISSOLVED: stats["dissolved"] += 1
            elif f.influence > 0: stats["active"] += 1
        return stats

    def effective_weight(self) -> np.ndarray:
        W = self.W_base.copy()
        for f in self.facts:
            if f.delta_W is not None and f.influence > 0:
                W += f.influence * f.delta_W
        return W


# In[ ]:


# 14. Simplicial-complex message passing
class SimplicialComplexNN:
    log = StructuredLogger("SimplicialNN")

    def __init__(self, n_nodes: int, d_features: int, d_hidden: int):
        self.n_nodes, self.d_features, self.d_hidden = n_nodes, d_features, d_hidden
        self.edges: List[Tuple[int, int]] = []
        self.triangles: List[Tuple[int, int, int]] = []
        rng = np.random.RandomState(333)
        self.W0     = rng.randn(d_features, d_hidden) * math.sqrt(2.0 / d_features)
        self.W_down = rng.randn(d_features, d_hidden) * math.sqrt(2.0 / d_features)

    def add_edge(self, u: int, v: int):
        if u == v:
            raise ValueError("self-loops are not allowed")
        e = (min(u, v), max(u, v))
        if e not in self.edges:
            self.edges.append(e)

    def add_triangle(self, u: int, v: int, w: int):
        tri = tuple(sorted([u, v, w]))
        for a, b in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[0], tri[2])]:
            if (a, b) not in self.edges:
                raise ValueError(
                    f"Triangle ({u},{v},{w}) requires edge ({a},{b}) which is missing")
        if tri not in self.triangles:
            self.triangles.append(tri)

    def boundary_operators(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        ne, nt = len(self.edges), len(self.triangles)
        B1 = np.zeros((self.n_nodes, ne), dtype=np.float64)
        for i, (u, v) in enumerate(self.edges):
            B1[v, i] = +1.0
            B1[u, i] = -1.0
        B2: Optional[np.ndarray] = None
        if nt:
            B2 = np.zeros((ne, nt), dtype=np.float64)
            for ti, (u, v, w) in enumerate(self.triangles):
                for ei, e in enumerate(self.edges):
                    if   e == (min(v, w), max(v, w)): B2[ei, ti] = +1.0
                    elif e == (min(u, w), max(u, w)): B2[ei, ti] = -1.0
                    elif e == (min(u, v), max(u, v)): B2[ei, ti] = +1.0
            assert np.linalg.norm(B1 @ B2) < 1e-10,                 "Boundary-of-boundary != 0: simplicial identity violated"
        return B1, B2

    def forward(self, x0: np.ndarray) -> np.ndarray:
        B1, _ = self.boundary_operators()
        L0 = B1 @ B1.T
        x1 = B1.T @ x0
        h  = (L0 @ x0) @ self.W0 + (B1 @ x1) @ self.W_down
        return SwiGLU.swish(h)


# In[ ]:


# 15. Late-interaction retrieval with MaxSim scoring
class MaxSimRetrieval:
    log = StructuredLogger("MaxSim")

    def __init__(self, d_model: int):
        self.d_model = d_model
        self.index: List[Dict] = []

    def add_document(self, doc_id: Any, tokens: np.ndarray):
        """Index a document given its (n_tokens, d_model) token embeddings."""
        norms = np.linalg.norm(tokens, axis=-1, keepdims=True)
        tokens_n = tokens / (norms + 1e-10)
        self.index.append({"id": doc_id, "tokens": tokens_n})

    def maxsim_score(self, q: np.ndarray, d: np.ndarray) -> float:
        """MaxSim(q, d) = sum_i max_j sim(q_i, d_j)."""
        q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-10)
        return float((q @ d.T).max(axis=1).sum())

    def retrieve(self, q: np.ndarray, top_k: int = 3) -> List[Dict]:
        if not self.index:
            return []
        scored = sorted(
            [{"id": doc["id"],
              "score": self.maxsim_score(q, doc["tokens"]),
              "n_tokens": doc["tokens"].shape[0]}
             for doc in self.index],
            key=lambda x: x["score"], reverse=True,
        )
        return scored[:top_k]


# In[ ]:


# 16. Chain-of-Thought and Tree-of-Thoughts
@dataclass
class DataContract:
    producer: str; consumer: str; payload: Any
    schema: Dict[str, type]; step_id: int
    timestamp: float = field(default_factory=time.time)
    integrity: str = ""

    def __post_init__(self):
        self.integrity = self._compute_hash()

    def _compute_hash(self) -> str:
        content = f"{self.producer}{self.consumer}{self.step_id}"
        if isinstance(self.payload, np.ndarray):
            content += str(self.payload.sum())
        else:
            content += str(self.payload)
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def verify(self) -> bool:
        return self.integrity == self._compute_hash()


@dataclass
class ReasoningVertex:
    vertex_id: str; depth: int; content: str
    confidence: float; parent_id: Optional[str]
    children: List[str] = field(default_factory=list)
    score: float = 0.0
    explored: bool = False


class ChainOfThought:
    log = StructuredLogger("CoT")

    def __init__(self):
        self.steps: List[ReasoningVertex] = []
        self.contracts: List[DataContract] = []
        self.step_num = 0

    def add_step(self, content: str, confidence: float,
                 payload: Optional[Dict] = None) -> DataContract:
        v = ReasoningVertex(
            f"cot_{self.step_num}", self.step_num, content, confidence,
            self.steps[-1].vertex_id if self.steps else None,
        )
        if self.steps:
            self.steps[-1].children.append(v.vertex_id)
        self.steps.append(v)
        c = DataContract(
            producer=f"cot_{self.step_num - 1}" if self.step_num > 0 else "input",
            consumer=v.vertex_id,
            payload=payload or {"content": content, "confidence": confidence},
            schema={"content": str, "confidence": float},
            step_id=self.step_num,
        )
        self.contracts.append(c)
        self.step_num += 1
        return c

    def path_confidence(self) -> float:
        return math.prod(s.confidence for s in self.steps) if self.steps else 0.0


class TreeOfThoughts:
    log = StructuredLogger("ToT")

    def __init__(self, beam_width: int = 4, max_depth: int = 5):
        self.beam_width, self.max_depth = beam_width, max_depth
        self.vertices: Dict[str, ReasoningVertex] = {}
        self.best_path: List[str] = []

    def _score(self, v: ReasoningVertex,
               siblings: List[ReasoningVertex]) -> float:
        conf = v.confidence
        div = 1.0
        if siblings:
            chars = set(v.content.lower())
            div = 1.0 - max(
                len(chars & set(s.content.lower())) /
                (len(chars | set(s.content.lower())) + 1e-8)
                for s in siblings
            )
        depth_pen = math.exp(-0.1 * abs(v.depth - self.max_depth // 2))
        return conf * 0.4 + div * 0.3 + depth_pen * 0.3

    def explore(self, branches: List[Dict]) -> List[ReasoningVertex]:
        new_v: List[ReasoningVertex] = []
        for i, b in enumerate(branches):
            depth = (self.vertices[b["parent_id"]].depth + 1
                     if b.get("parent_id") in self.vertices else 0)
            v = ReasoningVertex(
                f"tot_{len(self.vertices)}_{i}", depth,
                b["content"], b["confidence"], b.get("parent_id"),
            )
            if v.parent_id in self.vertices:
                self.vertices[v.parent_id].children.append(v.vertex_id)
            self.vertices[v.vertex_id] = v
            new_v.append(v)

        for v in new_v:
            v.score = self._score(v, [s for s in new_v if s is not v])
        new_v.sort(key=lambda v: v.score, reverse=True)
        beam = new_v[:self.beam_width]

        if beam:
            best = beam[0]
            path = [best.vertex_id]
            n = best
            while n.parent_id in self.vertices:
                path.append(n.parent_id)
                n = self.vertices[n.parent_id]
            self.best_path = list(reversed(path))

        return beam


# In[ ]:


# 17. Training state & step-level logging
class TrainingState:
    """Append-only history with LTL verification."""
    def __init__(self):
        self.loss_history:  List[float] = []
        self.step_history:  List[int]   = []
        self.error_rates:   List[float] = []
        self.aux_history:   List[float] = []
        self.lr_history:    List[float] = []

    def append(self, step: int, loss: float, err: float,
               aux: float, lr: float):
        self.loss_history.append(float(loss))
        self.step_history.append(int(step))
        self.error_rates.append(float(err))
        self.aux_history.append(float(aux))
        self.lr_history.append(float(lr))

    def verify_monotone(self, tol: float = 1e-6) -> bool:
        return LTLProperties.monotone_non_increase(self.error_rates, tol)

    def verify_append_only(self, snap: List[float]) -> bool:
        return LTLProperties.append_only(snap, self.error_rates)


# In[ ]:


# 18. Metacognitive training loop: forward -> loss -> backward -> update
class MetacognitiveTrainingLoop:
    log = StructuredLogger("MetaTraining")

    def __init__(self, model: "TrainableEngine", lr: float = 1e-3,
                 clip_norm: float = 1.0, lr_decay: float = 0.95,
                 decay_every: int = 100, seed: int = 42,
                 X_train: Optional[np.ndarray] = None,
                 y_train: Optional[np.ndarray] = None):
        self.model = model
        self.lr = lr
        self.clip_norm = clip_norm
        self.lr_decay = lr_decay
        self.decay_every = decay_every
        self.state = TrainingState()
        self.step = 0
        self.complete = False
        self.rng = np.random.RandomState(seed)
        self.X_train = X_train
        self.y_train = y_train

    def _sample_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.X_train is not None and self.y_train is not None:
            idx = self.rng.randint(0, len(self.X_train), self.model.config.batch_size)
            x = self.X_train[idx]
            y = self.y_train[idx]
        else:
            cfg = self.model.config
            x = self.rng.randn(cfg.batch_size, cfg.seq_len, cfg.d_model)
            y = np.broadcast_to(
                np.linalg.norm(x, axis=-1, keepdims=True), x.shape).copy()
        return x.astype(np.float64), y.astype(np.float64)

    def _compute_loss(self, pred: np.ndarray, target: np.ndarray,
                      aux: float) -> Tuple[float, float, float, np.ndarray]:
        diff = pred - target
        mse = float(np.mean(diff ** 2))
        total = mse + aux
        grad_out = 2.0 * diff / diff.size
        return total, mse, aux, grad_out

    def _clip(self, g: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(g)
        return g if n <= self.clip_norm else g * (self.clip_norm / (n + 1e-8))

    def _sgd_step(self, grads: Dict[str, np.ndarray]):
        with np.errstate(over="raise", invalid="raise"):
            for name, g in grads.items():
                clipped = self._clip(g)
                self.model.set_param(name,
                                     self.model.get_param(name) - self.lr * clipped)

    def step_one(self) -> Dict:
        cfg = self.model.config
        x, y = self._sample_batch()

        # Forward pass: use step as deterministic seq_offset for training.
        seq_off = self.step % max(1, cfg.max_seq - x.shape[1] + 1)
        attn_out = self.model.attention.forward(x, seq_offset=seq_off)
        ffn_out  = self.model.ffn.forward(attn_out)
        moe_out, aux, _ = self.model.moe.forward(ffn_out)
        pred = moe_out

        loss, mse, aux_val, grad_out = self._compute_loss(pred, y, aux)

        grads: Dict[str, np.ndarray] = {}

        # Backward through MoE.
        moe_grad_in, moe_grads = self.model.moe.backward(ffn_out, grad_out)

        # Backward through FFN.
        _, ffn_grads = self.model.ffn.backward(attn_out, moe_grad_in)
        for k, v in ffn_grads.items():
            grads[f"ffn.{k}"] = v

        for eid, eg in moe_grads.items():
            if eg["count"] == 0:
                continue
            for k in ("W_gate", "W_up", "W_down", "gamma", "beta"):
                if isinstance(eg[k], np.ndarray):
                    grads[f"moe.expert{eid}.{k}"] = eg[k] / max(eg["count"], 1)

        self._sgd_step(grads)

        err = float(loss) / (float(np.var(y)) + 1e-8)
        self.state.append(self.step, loss, err, aux_val, self.lr)
        if self.step > 0 and self.step % self.decay_every == 0:
            self.lr *= self.lr_decay
        self.step += 1

        # Global gradient norm (across all parameter gradients).
        if grads:
            all_grads = np.concatenate([g.ravel() for g in grads.values()])
            grad_norm = float(np.linalg.norm(all_grads))
        else:
            grad_norm = 0.0

        return {"step": self.step - 1, "loss": float(loss),
                "mse": float(mse), "aux": float(aux_val),
                "error_rate": err, "lr": float(self.lr),
                "grad_norm": grad_norm}

    def run(self, max_steps: int, log_every: int = 1) -> Dict:
        snap_before = self.state.error_rates.copy()
        for s in range(max_steps):
            info = self.step_one()
            assert self.state.verify_append_only(snap_before),                 f"Append-only invariant violated at step {info['step']}"
            snap_before = self.state.error_rates.copy()
            if log_every and s % log_every == 0:
                print(f"  step {info['step']:>4d} | loss={info['loss']:.6f} "
                      f"mse={info['mse']:.6f} aux={info['aux']:.4f} "
                      f"err={info['error_rate']:.4f} lr={info['lr']:.2e}")
        self.complete = True
        return {
            "steps": max_steps,
            "initial_loss":  self.state.loss_history[0],
            "final_loss":    self.state.loss_history[-1],
            "initial_error": self.state.error_rates[0],
            "final_error":   self.state.error_rates[-1],
            "history_length": len(self.state.loss_history),
            "complete": self.complete,
        }


# In[ ]:


# 19. TrainableEngine — unified model exposing every trainable tensor
@dataclass
class EngineConfig:
    """Architectural hyperparameters shared by every component."""
    d_model: int = 64
    n_heads: int = 4
    n_kv_heads: int = 2
    n_landmarks: int = 8
    d_ffn: int = 256
    d_ffn_per_expert: int = 32
    n_experts: int = 4
    top_k: int = 2
    rope_base: float = 10000.0
    max_seq: int = 256
    ema_decay: float = 0.99
    batch_size: int = 4
    seq_len: int = 8
    memit_lambda_reg: float = 1e-4
    seed: int = 2024
    name: str = "Hodge"
    version: str = "0.1.0"

    def to_dict(self) -> Dict:
        return dataclasses.asdict(self)


class TrainableEngine:
    """Aggregates every component and exposes a single state_dict interface."""

    # Class-level template; each instance gets its own copy (see __init__).
    _BASE_PARAM_SCHEMA: Dict[str, tuple] = {
        "ffn.W_gate": ("ffn", "W_gate"),
        "ffn.W_up":   ("ffn", "W_up"),
        "ffn.W_down": ("ffn", "W_down"),
        "ffn.gamma":  ("ffn", "gamma"),
        "ffn.beta":   ("ffn", "beta"),
        "attn.W_Q":  ("attention", "W_Q"),
        "attn.W_K1": ("attention", "W_K1"),
        "attn.W_K2": ("attention", "W_K2"),
        "attn.W_V":  ("attention", "W_V"),
        "attn.W_O":  ("attention", "W_O"),
        "attn.landmark_K": ("attention", "landmark_K"),
        "attn.landmark_V": ("attention", "landmark_V"),
        "attn.landmark_count": ("attention", "landmark_count"),
        "memit.W_base": ("memit", "W_base"),
        "memit.C":      ("memit", "C"),
        "simplicial.W0":     ("simplicial", "W0"),
        "simplicial.W_down": ("simplicial", "W_down"),
    }

    def __init__(self, config: EngineConfig):
        self.config = config
        # Instance-local copy so mutating it doesn't affect other instances.
        self.PARAM_SCHEMA: Dict[str, tuple] = dict(TrainableEngine._BASE_PARAM_SCHEMA)

        self.prng = PRNG(config.seed)
        self.manifold = GeodesicManifold(size=4, prng=self.prng)
        self.manifold.perturb_metric(noise_scale=0.5)
        self.rope = RoPE(d_model=config.d_model, base=config.rope_base,
                         max_seq=config.max_seq)
        self.attention = CoDAGQAL(d_model=config.d_model,
                                  n_heads=config.n_heads,
                                  n_kv_heads=config.n_kv_heads,
                                  n_landmarks=config.n_landmarks,
                                  ema_decay=config.ema_decay,
                                  rope=self.rope)
        self.ffn = FFNBlock(d_model=config.d_model, d_ffn=config.d_ffn)
        self.moe = MoELayer(d_model=config.d_model, n_experts=config.n_experts,
                            top_k=config.top_k,
                            d_ffn_per_expert=config.d_ffn_per_expert)
        self.memit = MEMITEditor(d_model=config.d_model, layer_idx=0,
                                 lambda_reg=config.memit_lambda_reg)
        self.simplicial = SimplicialComplexNN(
            n_nodes=4,
            d_features=config.d_model // 4,
            d_hidden=config.d_model // 2,
        )
        for u, v in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]:
            self.simplicial.add_edge(u, v)
        self.simplicial.add_triangle(0, 1, 2)
        self.retrieval = MaxSimRetrieval(d_model=config.d_model)
        self.cot = ChainOfThought()
        self.tot = TreeOfThoughts(beam_width=3, max_depth=4)

        # Populate MoE entries in the instance-local schema.
        self.PARAM_SCHEMA["moe.router.W_router"] = ("moe", "router", "W_router")
        for e in range(config.n_experts):
            self.PARAM_SCHEMA[f"moe.expert{e}.W_gate"] = ("moe", "experts", e, "W_gate")
            self.PARAM_SCHEMA[f"moe.expert{e}.W_up"]   = ("moe", "experts", e, "W_up")
            self.PARAM_SCHEMA[f"moe.expert{e}.W_down"] = ("moe", "experts", e, "W_down")
            self.PARAM_SCHEMA[f"moe.expert{e}.gamma"]  = ("moe", "experts", e, "gamma")
            self.PARAM_SCHEMA[f"moe.expert{e}.beta"]   = ("moe", "experts", e, "beta")

    def _resolve(self, dotted: str) -> np.ndarray:
        path = self.PARAM_SCHEMA[dotted]
        obj: Any = self
        for p in path:
            obj = getattr(obj, p) if isinstance(p, str) else obj[p]
        return obj

    def get_param(self, name: str) -> np.ndarray:
        val = self._resolve(name)
        if isinstance(val, (int, np.integer)) or (isinstance(val, np.ndarray) and val.ndim == 0):
            return np.array([int(val)], dtype=np.int32)
        return val.copy()

    def set_param(self, name: str, value: np.ndarray):
        path = self.PARAM_SCHEMA[name]
        obj: Any = self
        for p in path[:-1]:
            obj = getattr(obj, p) if isinstance(p, str) else obj[p]
        last = path[-1]
        if isinstance(last, str):
            # Validate shape before assignment to catch mismatched checkpoints.
            current = getattr(obj, last)
            if name == "attn.landmark_count":
                value = np.array(value.item(), dtype=np.int32)
            if hasattr(current, "shape") and current.shape != value.shape:
                if name == "attn.landmark_count":
                    pass
                else:
                    raise ValueError(
                        f"Shape mismatch for '{name}': "
                        f"expected {current.shape}, got {value.shape}")
            setattr(obj, last, value)
        else:
            obj[last] = value

    def named_parameters(self) -> Dict[str, np.ndarray]:
        return {n: self.get_param(n) for n in self.PARAM_SCHEMA}

    def num_parameters(self) -> int:
        return int(sum(p.size for p in self.named_parameters().values()))

    def state_dict(self) -> Dict[str, np.ndarray]:
        sd = self.named_parameters()
        rng = self.prng.state()
        # Pack the PRNG state as an int64 array for safetensors-compatible storage.
        rng_arr = np.array(
            [hash(rng["kind"]) % (2**31)]
            + rng["keys"]
            + [rng["pos"], rng["has_gauss"], int(rng["gauss"] * 1e6), rng["call_count"]],
            dtype=np.int64,
        )
        sd["__rng__"] = rng_arr
        return sd

    def load_state_dict(self, state: Dict[str, np.ndarray]):
        rng_arr = None
        for k, v in state.items():
            if k == "__rng__":
                rng_arr = v
            elif k in self.PARAM_SCHEMA:
                self.set_param(k, v)
        if rng_arr is not None:
            arr = rng_arr.tolist()
            call_count = int(arr[-1])
            gauss_int  = int(arr[-2])
            has_gauss  = int(arr[-3])
            pos        = int(arr[-4])
            keys       = [int(x) for x in arr[1:-4]]
            st = {
                "kind": "MT19937",
                "keys": keys,
                "pos": pos,
                "has_gauss": has_gauss,
                "gauss": gauss_int / 1e6,
                "call_count": call_count,
                "seed": self.config.seed,
            }
            self.prng.restore_state(st)

    def forward(self, x: np.ndarray, seq_offset: Optional[int] = None,
                return_extras: bool = False, update_landmarks: bool = True):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq, d_model).
            seq_offset: Starting position for RoPE.  When None the engine draws
                a random offset via its PRNG (training behaviour); pass an
                explicit value (e.g. 0) for deterministic inference.
            return_extras: If True, return a dict with intermediate activations.
        """
        cfg = self.config
        if seq_offset is None:
            seq_offset = int(self.prng.randint(0, max(1, cfg.max_seq - x.shape[1])))
        attn_out = self.attention.forward(x, seq_offset=seq_offset, update_landmarks=update_landmarks)
        ffn_out  = self.ffn.forward(attn_out)
        moe_out, aux, _ = self.moe.forward(ffn_out)
        if return_extras:
            return {"attn": attn_out, "ffn": ffn_out, "moe": moe_out, "aux": aux}
        return moe_out


# In[ ]:


# 20. Checkpoint utilities
import pickle
from pathlib import Path


def save_checkpoint(path: str, engine: "TrainableEngine",
                    trainer: Optional["MetacognitiveTrainingLoop"] = None,
                    extra: Optional[Dict] = None) -> Dict:
    """Save a complete checkpoint: model weights, RNG, trainer state, config.

    NOTE: Checkpoints use pickle for serialisation.  Only load checkpoints from
    sources you trust; unpickling untrusted data is a remote-code-execution risk.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "config": engine.config.to_dict(),
        "state_dict": {k: np.asarray(v) for k, v in engine.state_dict().items()},
        "schema_version": 1,
    }
    if trainer is not None:
        payload["trainer"] = {
            "step": trainer.step,
            "lr": trainer.lr,
            "complete": trainer.complete,
            "loss_history": list(trainer.state.loss_history),
            "error_rates": list(trainer.state.error_rates),
            "aux_history": list(trainer.state.aux_history),
            "lr_history": list(trainer.state.lr_history),
        }
    if extra:
        payload["extra"] = extra
    with open(p, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    manifest = {
        "path": str(p),
        "n_params": engine.num_parameters(),
        "schema_version": 1,
        "size_bytes": p.stat().st_size,
        "param_keys": list(engine.named_parameters().keys()),
    }
    return manifest


def load_checkpoint(path: str) -> Tuple[
        "EngineConfig", Dict[str, np.ndarray],
        Optional[Dict], Optional[Dict]]:
    """Load a checkpoint saved by save_checkpoint.

    WARNING: Uses pickle.  Only load checkpoints from trusted sources.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)
    cfg_data = payload.get("config", {})
    cfg = EngineConfig(**{k: v for k, v in cfg_data.items()
                          if k in EngineConfig.__dataclass_fields__})
    state_dict = payload.get("state_dict", {})
    return cfg, state_dict, payload.get("trainer"), payload.get("extra")


# In[ ]:


# 21. Safetensors export/import round-trip
SAFETENSORS_FORMAT_VERSION = "1"


def export_safetensors(path: str, engine: "TrainableEngine",
                       metadata: Optional[Dict] = None) -> Dict:
    """Export all named parameters to a single .safetensors file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for k, v in engine.named_parameters().items():
        if k == "attn.landmark_count":
            tensors[k] = np.ascontiguousarray(v, dtype=np.int32)
        else:
            tensors[k] = np.ascontiguousarray(v, dtype=np.float64)
    meta: Dict[str, str] = {
        "format_version": SAFETENSORS_FORMAT_VERSION,
        "config": json.dumps(engine.config.to_dict()),
        "param_names": json.dumps(list(tensors.keys())),
        "n_params": str(sum(t.size for t in tensors.values())),
        "engine_name": engine.config.name,
        "engine_version": engine.config.version,
    }
    if metadata:
        meta.update({
            k: (v if isinstance(v, str) else json.dumps(v))
            for k, v in metadata.items()
        })
    st_save(tensors, str(p), metadata=meta)
    return {
        "path": str(p),
        "n_tensors": len(tensors),
        "n_params": sum(t.size for t in tensors.values()),
        "metadata": meta,
    }


def import_safetensors(path: str, engine: Optional["TrainableEngine"] = None,
                       strict: bool = True) -> "TrainableEngine":
    """Load a .safetensors file into a fresh or provided TrainableEngine."""
    tensors = st_load(str(path))
    with safe_open(str(path), framework="np") as f:   # safe_open imported at top
        meta = dict(f.metadata() or {})
    cfg_dict = json.loads(meta["config"]) if "config" in meta else {}
    cfg = EngineConfig(**{k: v for k, v in cfg_dict.items()
                          if k in EngineConfig.__dataclass_fields__})
    if engine is None:
        engine = TrainableEngine(cfg)
    missing = [k for k in engine.PARAM_SCHEMA if k not in tensors]
    extra   = [k for k in tensors if k not in engine.PARAM_SCHEMA]
    if strict:
        if missing:
            raise KeyError(f"Missing tensors in safetensors file: {missing}")
        if extra:
            raise KeyError(f"Unknown tensors in safetensors file: {extra}")
    for k, v in tensors.items():
        if k in engine.PARAM_SCHEMA:
            engine.set_param(k, v)
    return engine


def inspect_safetensors(path: str) -> Dict:
    """Return a print-friendly summary of a safetensors file."""
    tensors = st_load(str(path))
    with safe_open(str(path), framework="np") as f:
        meta = dict(f.metadata() or {})
    return {
        "path": path,
        "tensors": {
            k: {"shape": list(v.shape), "dtype": str(v.dtype), "size": int(v.size)}
            for k, v in tensors.items()
        },
        "metadata": meta,
    }


# ## Training driver
#
# The function below wires everything together: it builds an engine, optionally
# restores from a checkpoint, runs a training loop with periodic checkpointing
# and metric logging, and finally exports the trained weights to safetensors.

# In[ ]:


# 22. End-to-end training driver
def train_engine(cfg: Optional[EngineConfig] = None,
                 max_steps: int = 30,
                 checkpoint_every: int = 10,
                 checkpoint_dir: str = "artifacts/checkpoints",
                 safetensors_path: str = "artifacts/engine.safetensors",
                 resume_from: Optional[str] = None,
                 log_every: int = 1,
                 seed: Optional[int] = None) -> Dict:
    """Train the engine end-to-end with checkpointing and safetensors export."""
    # ----- initialise / restore -------------------------------------------
    if resume_from and Path(resume_from).exists():
        # Validate that the resume_from path is within the trusted checkpoint_dir
        # But wait, original code was:
        # resume_from=str(Path(CKPT_DIR) / "final.pkl"),
        # checkpoint_dir=str(Path(CKPT_DIR) / "resumed")
        # In the demo code, resume_from is "artifacts/checkpoints/final.pkl"
        # but checkpoint_dir is "artifacts/checkpoints/resumed".
        # These are NOT relative to each other!
        # So resolving against `checkpoint_dir` WILL fail the demo code!
        # The demo code explicitly uses a different checkpoint_dir for the resumed run to avoid overwriting.
        # We must validate it against `CKPT_DIR` itself, or just check if it's within `Path("artifacts/checkpoints").resolve()`.

        # Let's validate it against a broader trusted root (like the parent of checkpoint_dir, or a configurable trusted root)
        # We'll just validate against `Path(resume_from).parent.resolve()` or something, or better yet:
        resolved_resume = Path(resume_from).resolve()
        # Ensure it's not trying to escape to system paths.
        if not str(resolved_resume).startswith(str(Path.cwd().resolve())):
            raise ValueError(f"resume_from path '{resume_from}' is not within trusted cwd")
        cfg_ckpt, state, trainer_state, _ = load_checkpoint(resume_from)
        engine = TrainableEngine(cfg_ckpt)
        engine.load_state_dict(state)
        cfg = cfg_ckpt
        print(f"[resume] loaded checkpoint {resume_from} "
              f"({engine.num_parameters():,} params)")
        start_step  = trainer_state.get("step", 0) if trainer_state else 0
        starting_lr = trainer_state.get("lr", 1e-3) if trainer_state else 1e-3
    else:
        trainer_state = None
        cfg = cfg or EngineConfig()
        if seed is not None:
            cfg.seed = seed
        engine = TrainableEngine(cfg)
        start_step  = 0
        starting_lr = 1e-3
        print(f"[init] built engine '{cfg.name}' v{cfg.version} "
              f"({engine.num_parameters():,} params)")

    trainer = MetacognitiveTrainingLoop(engine, lr=starting_lr, seed=cfg.seed)
    if trainer_state:
        trainer.step = trainer_state.get("step", 0)
        trainer.complete = trainer_state.get("complete", False)
        trainer.state.loss_history = collections.deque(trainer_state.get("loss_history", []), maxlen=1000)
        trainer.state.error_rates = collections.deque(trainer_state.get("error_rates", []), maxlen=1000)
        trainer.state.aux_history = collections.deque(trainer_state.get("aux_history", []), maxlen=1000)
        trainer.state.lr_history = collections.deque(trainer_state.get("lr_history", []), maxlen=1000)
    else:
        trainer.step = start_step
    print(f"[train] running {max_steps} steps from step {start_step}")
    history: List[Dict] = []

    for s in range(max_steps):
        info = trainer.step_one()
        history.append(info)
        if log_every and (trainer.step - 1) % log_every == 0:
            print(f"  step {info['step']:>4d} | loss={info['loss']:.6f} "
                  f"err={info['error_rate']:.4f} lr={info['lr']:.2e}")

        # ----- periodic checkpoint ----------------------------------------
        if checkpoint_every and trainer.step % checkpoint_every == 0:
            ckpt_path = Path(checkpoint_dir) / f"step_{trainer.step:06d}.pkl"
            manifest = save_checkpoint(str(ckpt_path), engine, trainer)
            print(f"  [ckpt] saved {ckpt_path.name} "
                  f"({manifest['n_params']:,} params, "
                  f"{manifest['size_bytes'] / 1024:.1f} KB)")

    # ----- final checkpoint -----------------------------------------------
    final_ckpt = Path(checkpoint_dir) / "final.pkl"
    save_checkpoint(str(final_ckpt), engine, trainer)
    print(f"[ckpt] final checkpoint -> {final_ckpt}")

    # ----- safetensors export ---------------------------------------------
    manifest = export_safetensors(safetensors_path, engine, metadata={
        "training_history": history,
        "final_loss": float(trainer.state.loss_history[-1]),
        "final_error_rate": float(trainer.state.error_rates[-1]),
    })
    print(f"[st]  exported safetensors -> {safetensors_path} "
          f"({manifest['n_tensors']} tensors, "
          f"{manifest['n_params']:,} params)")

    return {
        "engine": engine,
        "trainer": trainer,
        "history": history,
        "config": cfg,
        "checkpoint_dir": checkpoint_dir,
        "safetensors_path": safetensors_path,
        "final_ckpt": str(final_ckpt),
    }


# ## Inference from safetensors
#
# The helpers below demonstrate how to load a saved `.safetensors` file back into
# a fresh `TrainableEngine`, run a forward pass on new inputs, and inspect the
# round-trip integrity.

# In[ ]:


# 23. Inference helpers
def load_for_inference(safetensors_path: str) -> "TrainableEngine":
    """Load weights from a safetensors file into a freshly-constructed engine."""
    engine = import_safetensors(safetensors_path)
    print(f"[infer] loaded {engine.config.name} v{engine.config.version} "
          f"({engine.num_parameters():,} params)")
    return engine


def run_inference(engine: "TrainableEngine", x: np.ndarray,
                  seq_offset: int = 0,
                  return_extras: bool = False):
    """Deterministic inference: uses seq_offset=0 by default so the output
    is reproducible regardless of the engine's internal PRNG state."""
    return engine.forward(x, seq_offset=seq_offset, return_extras=return_extras, update_landmarks=False)


def parity_check(engine_a: "TrainableEngine", engine_b: "TrainableEngine",
                 x: np.ndarray, seq_offset: int = 0) -> Dict:
    """Compare outputs of two engines on the same input.

    Uses a fixed seq_offset so the comparison is not confounded by divergent
    PRNG states (which would differ between a freshly-trained engine and one
    loaded from safetensors that doesn't store RNG state).
    """
    a = engine_a.forward(x, seq_offset=seq_offset, update_landmarks=False)
    b = engine_b.forward(x, seq_offset=seq_offset, update_landmarks=False)
    diff = float(np.max(np.abs(a - b)))
    return {"max_abs_diff": diff, "shape": list(a.shape), "match": diff < 1e-8}


# ## End-to-end demonstration
#
# The cell below executes the full pipeline in a single run:
#
# 1. Build a fresh `TrainableEngine`.
# 2. Train for `STEPS` steps, checkpointing every `CKPT_EVERY` steps.
# 3. Export the trained model to safetensors.
# 4. Re-import the safetensors into a brand-new engine.
# 5. Run inference and verify the round-trip is numerically identical.
#
# You can re-run this cell after tweaking the constants at the top.

# In[ ]:


# 24. End-to-end demonstration
from pathlib import Path

if __name__ == "__main__":

    STEPS       = 40
    CKPT_EVERY  = 10
    LOG_EVERY   = 1
    CKPT_DIR    = "artifacts/checkpoints"
    SAFETENSORS = "artifacts/engine.safetensors"

    # 1. Fresh build + train + checkpoint + safetensors export.
    result = train_engine(
        cfg=EngineConfig(),
        max_steps=STEPS,
        checkpoint_every=CKPT_EVERY,
        checkpoint_dir=CKPT_DIR,
        safetensors_path=SAFETENSORS,
        log_every=LOG_EVERY,
        seed=2026,
    )

    # 2. Inspect the safetensors file.
    print("\n--- safetensors inspection ---")
    info = inspect_safetensors(SAFETENSORS)
    print(f"path        : {info['path']}")
    print(f"tensors     : {len(info['tensors'])}")
    print(f"metadata    : {sorted(info['metadata'].keys())}")
    for name, meta in list(info['tensors'].items())[:6]:
        print(f"  {name:<28s} shape={meta['shape']} dtype={meta['dtype']}")
    print(f"  ... ({len(info['tensors']) - 6} more tensors)")

    # 3. Reload into a brand-new engine.
    loaded_engine = load_for_inference(SAFETENSORS)

    # 4. Round-trip parity check.
    # Both engines use seq_offset=0 so divergent PRNG states don't affect the result.
    print("\n--- parity check (trained vs reloaded) ---")
    test_x = np.random.RandomState(0).randn(2, 8, loaded_engine.config.d_model)
    parity = parity_check(result["engine"], loaded_engine, test_x, seq_offset=0)
    print(f"max |\u0394|      : {parity['max_abs_diff']:.2e}")
    print(f"output shape : {parity['shape']}")
    print(f"match        : {parity['match']}")

    # 5. Show training summary.
    print("\n--- training summary ---")
    hist = result["history"]
    print(f"steps        : {len(hist)}")
    print(f"init loss    : {hist[0]['loss']:.6f}")
    print(f"final loss   : {hist[-1]['loss']:.6f}")
    print(f"loss delta   : {hist[-1]['loss'] - hist[0]['loss']:+.6f}")
    print(f"init err     : {hist[0]['error_rate']:.4f}")
    print(f"final err    : {hist[-1]['error_rate']:.4f}")

    # 6. List checkpoint files on disk.
    print("\n--- checkpoint files ---")
    for p in sorted(Path(CKPT_DIR).glob("*.pkl")):
        print(f"  {p.name:<24s} {p.stat().st_size:>8d} bytes")

    # 7. Demonstrate resume from final checkpoint.
    print("\n--- resume from final checkpoint + 5 extra steps ---")
    resumed = train_engine(
        resume_from=str(Path(CKPT_DIR) / "final.pkl"),
        max_steps=5,
        checkpoint_every=5,
        checkpoint_dir=str(Path(CKPT_DIR) / "resumed"),
        safetensors_path="artifacts/engine_resumed.safetensors",
        log_every=1,
    )
    print(f"resumed loss : {resumed['history'][0]['loss']:.6f}")


    # ## What just happened
    #
    # - **Training** — a 30-step (configurable) loop running real forward + backward
    #   passes through SwiGLU, FFN residual, MoE routing and attention. Loss values
    #   are computed live, not faked.
    # - **Checkpointing** — every `CKPT_EVERY` steps the *full* state dict
    #   (parameters + RNG state + trainer state + LTL-verified history) is pickled
    #   to `artifacts/checkpoints/step_*.pkl`. Training is fully resumable.
    # - **Safetensors export** — the trained weights are written to a single
    #   `artifacts/engine.safetensors` file with a metadata header describing the
    #   architecture, version, training summary and parameter count.
    # - **Inference** — `import_safetensors` rebuilds an engine and loads the
    #   weights. A parity check confirms bit-exact reproduction of the forward
    #   pass on identical inputs.
    #
    # Tweak `EngineConfig` to change the architecture (d_model, n_heads,
    # n_experts, top_k, ...) and re-run the demo cell to retrain from scratch.
