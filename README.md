# 🔑 KEY: Evolve LoRA Adapters Using Neuroevolution

**Optimize neural network adapters through genetic algorithms instead of gradient descent.**

Evolve ~100K adapter parameters while keeping base models frozen — 100x more efficient than full fine-tuning, and works when your objective isn't differentiable.

---

## 🎮 Try It

| | |
|---|---|
| **[🌌 Live Demo](https://huggingface.co/spaces/tostido/Cascade-Hyperlattice)** | Watch evolution in action |
| **[🧠 Champion Model](https://huggingface.co/datasets/tostido/key-data/tree/main/models)** | The evolved DreamerV3 model |
| **[📊 Dataset](https://huggingface.co/datasets/tostido/key-data)** | 40K+ logged evolutionary events |

---

## What KEY Does

KEY evolves **LoRA adapters** on frozen base models (MiniLM-L6, DreamerV3) using NEAT-style neuroevolution:

1. **Freeze** the base model (22M-200M parameters)
2. **Evolve** only the adapter layer (~100K parameters)
3. **Evaluate** using pluggable fitness functions
4. **Select** via tournament + speciation (prevents premature convergence)
5. **Log** every mutation, crossover, and fitness score

### Example: Evolving Semantic Similarity

**Task**: Adapt MiniLM embeddings to preserve semantic relationships

**Test Pair**: "The cat sat on the mat" ↔ "A feline rested on the rug"

| Generation | Cosine Similarity | Fitness |
|------------|-------------------|---------|
| 0          | 0.42 (random)     | 0.35    |
| 50         | 0.76              | 0.64    |
| 100        | 0.89              | 0.82    |

The evolved adapter learned to preserve semantic similarity while improving output quality.

---

## Why Evolve Instead of Gradient Descent?

Neuroevolution works when:
- ✅ Your objective **isn't differentiable** (human preference, discrete outputs)
- ✅ You want **population diversity** (speciation prevents local optima)
- ✅ You're optimizing for **interface quality**, not task loss
- ✅ You need **full auditability** (every mutation logged with provenance)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PopulationManager                            │
│              NEAT-style speciation + tournament                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   ┌─────────┐         ┌─────────┐         ┌─────────┐
   │  Node   │         │  Node   │         │  Node   │  × N
   │ fitness │         │ fitness │         │ fitness │
   │ + brain │         │ + brain │         │ + brain │
   └────┬────┘         └────┬────┘         └────┬────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
              ┌──────────────────────────┐
              │      Evolvable Brain     │
              │  ┌────────────────────┐  │
              │  │ Base Model (frozen)│  │  ← MiniLM / DreamerV3
              │  │     22M-200M       │  │
              │  └─────────┬──────────┘  │
              │            ▼             │
              │  ┌────────────────────┐  │
              │  │  LoRA Adapter      │  │  ← EVOLVED (~12K)
              │  │  + Projection Head │  │  ← EVOLVED (~99K)
              │  └────────────────────┘  │
              └──────────────────────────┘
```

**Total evolved parameters**: ~111K (LoRA rank-4 + projection)

---

## 🔐 Get Full Source Access

| Tier | Price | What You Get |
|------|-------|--------------|
| **🔑 Source Access** | $100 one-time | Full codebase, private repo invite |
| **🤝 Hands-On** | $50/hour | I coach you through wiring your own model |
| **🛠️ Done-For-You** | $500 flat | I wire up your custom model for you |
| **🎤 Speaking** | $2,000 | Talk at your company on gradient-free optimization |

### **[→ Sponsor on GitHub](https://github.com/sponsors/Yufok1)**

---

## FAQ

**Q: What's a "quine brain"?**
> A brain that can serialize its weights → mutate → deserialize. This enables genetic algorithms to evolve neural networks. Think "self-modifying adapter."

**Q: Why not just use backprop?**
> Backprop requires differentiable objectives. Evolution works with any fitness function: human ratings, game scores, discrete metrics, or even "does this output look good?"

**Q: Is this real?**
> Yes. The [dataset](https://huggingface.co/datasets/tostido/key-data) contains 40K+ real logged events from actual evolutionary runs. $100 tier includes full source.

---

## Contact

**DM on X: [@Toasteedo](https://x.com/Toasteedo)**

---

## License

Source code available to sponsors. Dataset and demo are MIT licensed.
