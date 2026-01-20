# 🔑 Ouroboros-Key

**A production neural network that converts existing models into quine-replicant capable models.**

Transform any `.pt`, `.onnx`, or compute-oriented model into a self-replicating, self-verifying quine brain with full provenance tracking.

---

## 🎮 Try the Demo

**[🌌 Live Demo on Hugging Face](https://huggingface.co/spaces/tostido/Ouroboros)** — no purchase required

---

## 📦 Dataset & Documentation

**[📊 Ouroboros-Key Dataset](https://huggingface.co/datasets/tostido/key-data)** — Full documentation, schemas, and exported logs

---

## 🔐 Get Full Source Access

The Ouroboros-Key source code is available via GitHub Sponsors.

| Tier | Price | What you get |
|------|-------|--------------|
| **🔑 Source Access** | $100 one-time | Private repo invite, full codebase |
| **📚 Guided** | $150/month | Access + ongoing coaching |
| **🤝 Hands-On** | $500 one-time | I do one conversion with you + support |

### **[→ Sponsor on GitHub](https://github.com/sponsors/Yufok1)**

Sponsors at paid tiers automatically receive an invite to the private `Ouroboros-key` repository.

---

## What It Does

- **Input-agnostic**: Supports `.pt`, `.onnx`, and other model formats
- **Quine conversion**: Wraps models with self-reference & re-instantiation hooks
- **Provenance tracking**: Merkle-linked decisions, CASCADE-LATTICE integration
- **Glass-box transparency**: Every weight, decision, and mutation is fully auditable
- **Evolution engine**: NEAT-style speciation, tournament selection, fitness sharing

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       PopulationManager                          │
│                  (NEAT-style speciation)                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   ┌────▼────┐         ┌────▼────┐         ┌────▼────┐
   │  Node   │         │  Node   │         │  Node   │  × N
   │ traits  │         │ traits  │         │ traits  │
   │ + brain │         │ + brain │         │ + brain │
   └────┬────┘         └────┬────┘         └────┬────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                    ┌───────▼───────┐
                    │  DreamerBrain │ (~200M params)
                    └───────────────┘
```

---

## Contact

**DM on X: [@Toasteedo](https://x.com/Toasteedo)**

---

## License

Source code available to sponsors. Dataset and demo are MIT licensed.
