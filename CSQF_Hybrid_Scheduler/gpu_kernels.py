from numba import cuda
import numpy as np
import config


# GPU Kernel: Parallel Resource Conflict Detection
@cuda.jit
def check_resource_conflict_kernel(schedule_matrix, flow_reqs, flow_paths, results,
                                   n_cycles, n_links, sliding_window):

    # Get the unique thread index (Flow ID)
    flow_idx = cuda.grid(1)

    if flow_idx < flow_reqs.shape[0]:
        conflict = 0

        # Define the range for Elastic Resource Blocks (EB)
        start_q = config.TT_QUEUES
        end_q = config.TOTAL_QUEUES

        # Iterate through the sliding window
        for t in range(sliding_window):
            cycle_idx = t % n_cycles

            # Iterate through each hop in the flow's path
            for hop in range(16):  # Assuming max hops = 16
                link_id = flow_paths[flow_idx, hop]
                if link_id == -1:
                    break  # End of path

                # Check if the link has any free EB queues in the current cycle
                # Logic: If all EB queues are occupied (!=0), the link is full
                is_link_full = 1
                for q in range(start_q, end_q):
                    if schedule_matrix[link_id, cycle_idx, q] == 0:  # 0 means free
                        is_link_full = 0
                        break

                if is_link_full == 1:
                    conflict = 1
                    break

            if conflict == 1:
                break

        # Write result
        results[flow_idx] = conflict


def gpu_batch_conflict_check(global_schedule, avb_flows, link_map):

    n_flows = len(avb_flows)
    if n_flows == 0:
        return []

    # Data Prep: Convert flow objects to Numpy arrays
    flow_reqs = np.array([f['bandwidth_slots'] for f in avb_flows], dtype=np.int32)

    # Path Mapping (Flow -> Link IDs Matrix)
    max_hops = 16
    paths_array = np.full((n_flows, max_hops), -1, dtype=np.int32)
    for i, flow in enumerate(avb_flows):
        path = flow['path']
        # Convert node pairs to Link IDs
        for j in range(len(path) - 1):
            u, v = path[j], path[j + 1]
            if (u, v) in link_map and j < max_hops:
                paths_array[i, j] = link_map[(u, v)]

    results = np.zeros(n_flows, dtype=np.int32)

    # Attempt GPU Execution
    if cuda.is_available():
        try:
            # Copy data to Device
            d_matrix = cuda.to_device(global_schedule)
            d_reqs = cuda.to_device(flow_reqs)
            d_paths = cuda.to_device(paths_array)
            d_results = cuda.to_device(results)

            # Calculate Grid and Block dimensions
            threadsperblock = config.TPB
            blockspergrid = (n_flows + (threadsperblock - 1)) // threadsperblock

            # Launch Kernel
            check_resource_conflict_kernel[blockspergrid, threadsperblock](
                d_matrix, d_reqs, d_paths, d_results,
                global_schedule.shape[1], global_schedule.shape[0], config.SLIDING_WINDOW_SIZE
            )
            # Copy results back to Host
            results = d_results.copy_to_host()
        except Exception as e:
            print(f"Warning: GPU execution failed ({e}). Falling back to CPU mode.")
            results = _cpu_fallback_check(global_schedule, avb_flows, link_map)
    else:
        # print("Info: No GPU detected. Using CPU mode.")
        results = _cpu_fallback_check(global_schedule, avb_flows, link_map)

    return results


def _cpu_fallback_check(matrix, flows, link_map):
    """CPU version of conflict detection (for compatibility without GPU)"""
    results = np.zeros(len(flows), dtype=np.int32)
    start_q, end_q = config.TT_QUEUES, config.TOTAL_QUEUES
    n_cycles = matrix.shape[1]

    for i, flow in enumerate(flows):
        conflict = False
        path = flow['path']
        for t in range(config.SLIDING_WINDOW_SIZE):
            cycle_idx = t % n_cycles
            for j in range(len(path) - 1):
                u, v = path[j], path[j + 1]
                if (u, v) not in link_map: continue
                link_id = link_map[(u, v)]

                # Check for free queues
                is_full = True
                for q in range(start_q, end_q):
                    if matrix[link_id, cycle_idx, q] == 0:
                        is_full = False
                        break
                if is_full:
                    conflict = True
                    break
            if conflict: break
        if conflict: results[i] = 1
    return results