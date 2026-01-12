import numpy as np

# System Basic Parameters  -
BASE_CYCLE = 0.125     # Base cycle length (unit: ms)
HYPER_CYCLE = 32       # Hypercycle (unit: ms), LCM of all flow periods
LINK_CAPACITY = 1000   # Link bandwidth
MTU = 1500             # Maximum Transmission Unit

# Queue Resource Configuration
TOTAL_QUEUES = 8       # Total number of queues per port
TT_QUEUES = 4          # First 4 queues for TT flows (Fixed Resource Blocks - FB)
AVB_QUEUES = 4         # Last 4 queues for AVB flows (Elastic Resource Blocks - EB)
QUEUE_CAPACITY = 10    # Maximum packet capacity per queue

# Dynamic Cycle & Load Control
LOAD_THRESHOLD_HEAVY = 0.7  # Threshold to trigger cycle expansion
LOAD_THRESHOLD_LIGHT = 0.3  # Threshold to trigger cycle compression
EXPANSION_FACTOR = 0.5      # Cycle adjustment factor (alpha)
SLIDING_WINDOW_SIZE = 4     # Sliding window size for AVB preemption

# --- GPU Configuration ---
TPB = 16  # Threads Per Block for CUDA kernels