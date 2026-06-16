# HodgeLM — Mathematically Verified Trainable AI Engine

A complete, runnable Jupyter notebook implementing a transformer-like architecture built entirely in NumPy, with gradient-based training, checkpointing, and safetensors export.

## Components

- **SwiGLU** — numerically-stable Swish-gated linear unit with analytic backward
- **RoPE** — rotary positional embeddings
- **CoDA-GQA-L** — constrained orthogonal differential attention with landmark KV cache and EMA
- **FFNBlock** — pre-norm residual FFN with SwiGLU
- **MoE Layer** — mixture of experts with top-k routing and auxiliary load-balancing loss
- **MEMIT Editor** — model editing with covariance regularisation and null-space constraints
- **Simplicial Complex NN** — message passing on simplicial complexes (graph + triangle boundaries)
- **MaxSim Retrieval** — late-interaction retrieval with MaxSim scoring
- **CoT / ToT** — chain-of-thought and tree-of-thoughts reasoning
- **Geodesic Manifold** — Dijkstra-based geodesic computation on a Riemannian grid
- **LTL Property Checker** — monotonicity and append-only invariants over training history

## Usage

Run the notebook cells top-to-bottom. The final cell (`#24`) executes the full pipeline:
1. Build a fresh `TrainableEngine`
2. Train for 40 steps with periodic checkpointing
3. Export weights to safetensors
4. Re-import and verify round-trip numerical parity
5. Resume from checkpoint for additional training

## Requirements

- Python 3.10+
- `numpy`
- `safetensors` (auto-installed if missing)

## Output Structure

```
artifacts/
  checkpoints/
    step_000010.pkl
    step_000020.pkl
    ...
    final.pkl
  engine.safetensors
```

## License

MIT
