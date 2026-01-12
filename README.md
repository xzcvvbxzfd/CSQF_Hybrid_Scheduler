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
