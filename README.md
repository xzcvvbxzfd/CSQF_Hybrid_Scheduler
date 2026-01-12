# CSQF_Hybrid_Scheduler
# CSQF-Based Hybrid Traffic Scheduling Method for WANs

This repository contains the official source code implementation for the paper:  
**"CSQF-Based Hybrid Traffic Scheduling Method for Wide Area Networks"**

It implements a dynamic collaborative scheduling algorithm for Time-Sensitive Networking (TSN) over WANs, featuring Integer Linear Programming (ILP) for Time-Triggered (TT) flows and a Dynamic Cycle Scaling mechanism for Audio-Video Bridging (AVB) flows.

## Key Features

**Internet2 Topology Model**: Accurate reconstruction of the WAN topology (16 nodes, heterogeneous links) as described in the paper.
**Algorithm 1 (TT Flows)**: Deterministic scheduling using ILP with Fixed Resource Blocks (FB).
**Algorithm 2 (AVB Flows)**: Elastic scheduling with Dynamic Cycle Scaling and Sliding Window Preemption.
**GPU Acceleration**: Parallel resource conflict detection using **Numba/CUDA** to handle large-scale traffic analysis.
Note: Includes automatic CPU fallback if no NVIDIA GPU is detected.

## Note on Reproducibility

The code segments provided above represent the core implementation of the proposed method. For the complete source code artifact and experimental data required for full reproducibility, please contact the author via email at: liu_xiaokai@bistu.edu.cn.

## GPU Acceleration Implementation

As described in **Section 4.1** of the paper, the proposed method leverages GPU parallelism to solve the computationally intensive resource conflict detection problem. This artifact reproduces the acceleration mechanism using **Numba/CUDA**.

### Implementation Details
The GPU acceleration logic is encapsulated in `gpu_kernels.py` and follows the parallelization strategy outlined in the manuscript:

1.  **Task Partitioning**: The conflict detection for AVB flows is decoupled from the ILP solving for TT flows. Each CUDA thread processes the resource validation for a single AVB flow independently.
2.  **Data Parallelism**: The global resource reservation matrix (Dimensions: $Links \times Cycles \times Queues$) is transferred to the GPU memory. Threads access this shared data structure to check availability within the sliding window.
3.  **Automatic Fallback**: To ensure reproducibility across different hardware environments, the code includes an automatic check (`cuda.is_available()`). If no NVIDIA GPU is detected, the system gracefully degrades to a CPU-based serial execution mode.

### How to Verify
When running `main.py`, the console output will explicitly state whether the GPU kernel is active:
* **GPU Mode**: `[INFO] GPU detected. Executing parallel conflict detection via CUDA.`
* **CPU Mode**: `[WARNING] No GPU detected. Falling back to CPU serial execution.`

---

## Experimental Results Verification

Running the `main.py` script will perform a simulation that validates the key performance metrics presented in **Section 4** of the paper.

### 1. Hybrid Traffic Scheduling Performance (Fig. 6 & Fig. 7)
The simulation generates a mixed traffic scenario (TT/AVB) to demonstrate the algorithm's effectiveness:
**TT Flows**: The algorithm guarantees **100% scheduling success rate** for Time-Triggered flows due to the strict priority of Fixed Resource Blocks (FB), matching the results in **Table 3**.
**AVB Flows**: Under heavy load scenarios (e.g., 3000 flows), the proposed Dynamic Cycle algorithm maintains a significantly higher success rate (~75.6%) compared to static strategies.

### 2. Resource Fragmentation Optimization (Fig. 8)
The console output tracks the dynamic adjustment of cycle lengths. You will observe:
**Light Load**: The cycle length compresses (e.g., `< 0.125ms`) to merge scattered resource blocks.
**Fragmentation Rate**: The dynamic adjustment reduces the Resource Fragmentation Rate (RFR) to approximately **5%-12%**, compared to >25% in static approaches.

### 3. Comparison with State-of-the-Art (Table 6)
The simulation outputs key metrics that align with the comparative analysis in the paper:
**Jitter Control**: TT flow jitter is strictly bounded (Standard Deviation < 1.5μs).
**Efficiency**: The GPU-accelerated implementation demonstrates the scalability required for WANs, achieving scheduling times significantly lower than traditional ILP-only methods.

## Repository Structure
```text
CSQF_Hybrid_Scheduler/
├── main.py                 # Entry point for the simulation
├── config.py               # System parameters and constraints
├── topology.py             # Network topology construction (Internet2)
├── tt_scheduler.py         # ILP-based TT flow scheduler
├── avb_scheduler.py        # Dynamic Cycle AVB flow scheduler
├── gpu_kernels.py          # CUDA kernels for parallel conflict detection
└── requirements.txt        # Python dependencies



