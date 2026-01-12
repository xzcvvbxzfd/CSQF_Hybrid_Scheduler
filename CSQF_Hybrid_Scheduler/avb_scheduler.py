import numpy as np
import networkx as nx
import config
from gpu_kernels import gpu_batch_conflict_check


class AVBScheduler:
    def __init__(self, topology, link_map, schedule_matrix):
        self.topo = topology
        self.link_map = link_map
        self.matrix = schedule_matrix
        self.current_cycle = config.BASE_CYCLE

    def dynamic_cycle_scaling(self):
        # Calculate global load factor
        used_blocks = np.count_nonzero(self.matrix)
        total_blocks = self.matrix.size
        load = used_blocks / total_blocks if total_blocks > 0 else 0

        old_cycle = self.current_cycle

        # Adjust cycle based on thresholds
        if load >= config.LOAD_THRESHOLD_HEAVY:
            self.current_cycle *= (1 + config.EXPANSION_FACTOR)  # Heavy load -> Expand
        elif load <= config.LOAD_THRESHOLD_LIGHT:
            self.current_cycle *= (1 - config.EXPANSION_FACTOR)  # Light load -> Compress

        # Clamp adjustment range
        self.current_cycle = max(config.BASE_CYCLE * 0.7,
                                 min(self.current_cycle, config.BASE_CYCLE * 1.3))

        print(f"   [Dynamic Scaling] Current Load: {load:.2f}, Cycle: {old_cycle:.3f}ms -> {self.current_cycle:.3f}ms")

    def schedule_flows(self, avb_flows):

        print(f"--- Algorithm 2: Scheduling {len(avb_flows)} AVB Flows ---")

        # Execute Dynamic Cycle Scaling
        self.dynamic_cycle_scaling()

        # Calculate path and required resources for all flows
        for flow in avb_flows:
            try:
                flow['path'] = nx.shortest_path(self.topo, flow['src'], flow['dst'], weight='weight')
            except:
                flow['path'] = []
            # Calculate required bandwidth slots
            flow['bandwidth_slots'] = max(1, int(flow['bandwidth'] / 10))

        # GPU Batch Conflict Detection
        # Returns: 0 = Available, 1 = Conflict
        conflict_results = gpu_batch_conflict_check(self.matrix, avb_flows, self.link_map)

        success_count = 0
        for i, flow in enumerate(avb_flows):
            if not flow['path']: continue

            # If GPU check passes
            if conflict_results[i] == 0:
                self._reserve_resources(flow)
                success_count += 1

        return success_count

    def _reserve_resources(self, flow):
        """
        Allocates Elastic Resource Blocks (EB).
        """
        start_q, end_q = config.TT_QUEUES, config.TOTAL_QUEUES
        n_cycles = self.matrix.shape[1]

        # Reserve resources along the path
        for i, (u, v) in enumerate(zip(flow['path'][:-1], flow['path'][1:])):
            if (u, v) in self.link_map:
                link_id = self.link_map[(u, v)]

                # Search for resources within Sliding Window
                # Simplified logic: Occupy if any cycle in window is valid
                for t in range(config.SLIDING_WINDOW_SIZE):
                    cycle_idx = t % n_cycles
                    allocated = False
                    # Iterate through EB queues
                    for q in range(start_q, end_q):
                        if self.matrix[link_id, cycle_idx, q] == 0:
                            self.matrix[link_id, cycle_idx, q] = 1  # Mark as occupied
                            allocated = True
                            break
                    if allocated: break