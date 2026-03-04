# WeTriC-AutoTune  
### Adaptive Spread Autotuning for Wedge-Parallel Triangle Counting on GPUs

This repository extends the WeTriC (Wedge-Parallel Triangle Counting) algorithm from Euro-Par 2025 with an adaptive runtime spread autotuning mechanism.

Original paper:  
Spaan et al., “Wedge-Parallel Triangle Counting for GPUs,” Euro-Par 2025  

Extension:  
Çapar & Guluyev, “Adaptive Spread Autotuning for Wedge-Parallel Triangle Counting on GPUs,” 2025  

---

## 🚀 Overview

Triangle counting is a core primitive in graph analytics used in:

- Social network analysis  
- Clustering coefficient computation  
- k-truss decomposition  
- Link prediction  

WeTriC achieves excellent GPU load balancing by assigning individual wedges to threads.

For a vertex v with degree d(v), the number of wedges is:

    C(d(v), 2)

Each thread processes one or more wedges depending on the spread parameter (s).

---

## 🎯 The Problem

Performance strongly depends on the spread value:

- Low s → better locality, higher overhead  
- High s → lower overhead, reduced locality  

The optimal spread varies across graphs and hardware.  
Manual tuning is impractical.

---

## 🔥 Our Contribution: Adaptive Spread Autotuning

We introduce an automatic runtime mechanism that:

1. Tests candidate spread values  
2. Respects shared memory constraints  
3. Measures GPU kernel execution time  
4. Selects the best-performing spread  

Candidate set:

    S = {1, 2, 3, 4, 6, 7, 8, 10, 12, 16}

### Results

- 83% exact optimal selection  
- Average performance gap: 1.01×  
- Worst-case gap: 3%  
- Negligible overhead in multi-iteration workloads  

No manual tuning required.

---

## ⚙️ Requirements

- CUDA Toolkit (tested on 13.1)
- NVIDIA GPU (tested on RTX 4060 Laptop GPU)
- C++ compiler
- CUB (included in CUDA)

---

## 🔧 Build

Set your GPU architecture in the Makefile (default: sm_86):

```bash
make
```

Clean:

```bash
make clean
```

---

## ▶️ Run

Matrix Market:

```bash
./tc -m graph.mtx -s <spread> -a <adj_len>
```

Edge list:

```bash
./tc -e graph.txt -s <spread> -a <adj_len>
```

---

## 🤖 Enable Autotuning

Use the -A flag:

```bash
./tc -m Amazon0302.mtx -A -a 8192 -l 10
```

The autotuner runs once, selects the optimal spread, and uses it for benchmarking.

---

## 📁 Repository Structure

```
src/              Main implementation
experiments/      Alternative triangle counting variants
Makefile
README.md
```

Includes:
- Vertex-parallel variants  
- Edge-parallel variants  
- Wedge-parallel baselines  
- Big graph (>2^32 edges) support  

---

## 📌 Why It Matters

Manual tuning:
- Graph-specific  
- Hardware-specific  
- Time-consuming  

Adaptive tuning:
- Automatic  
- Hardware-aware  
- Near-optimal performance  
- Production-ready  

Just run with `-A` and let the algorithm choose the best configuration.
