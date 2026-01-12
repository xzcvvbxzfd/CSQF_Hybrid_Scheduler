import pulp
import networkx as nx
import config


class TTScheduler:
    def __init__(self, topology, link_map, schedule_matrix):
        self.topo = topology
        self.link_map = link_map
        self.matrix = schedule_matrix  # Reference to global resource matrix
        self.scheduled_flows = []

    def schedule_flows(self, tt_flows):
        print(f"Scheduling {len(tt_flows)} TT Flows (via ILP)")
        # Sort by deadline (Basic priority strategy)
        tt_flows.sort(key=lambda x: x['deadline'])

        success_count = 0
        for flow in tt_flows:
            try:
                path = nx.shortest_path(self.topo, flow['src'], flow['dst'], weight='weight')
            except nx.NetworkXNoPath:
                print(f"Flow {flow['id']} path unreachable. Skipping.")
                continue

            if self._solve_ilp(flow, path):
                self._reserve_resources(flow)  # Update resource matrix
                self.scheduled_flows.append(flow)
                success_count += 1

        return success_count

    def _solve_ilp(self, flow, path):

        prob = pulp.LpProblem(f"TT_Flow_{flow['id']}", pulp.LpMinimize)

        # Decision Variable: Initial transmission cycle offset 'phi' (Integer)
        max_cycles = int(config.HYPER_CYCLE / config.BASE_CYCLE)
        phi = pulp.LpVariable("phi", 0, max_cycles, cat='Integer')

        # Objective Function: Minimize offset (transmit as early as possible)
        prob += phi

        # Calculate total physical propagation delay along path
        path_delay = sum(self.topo[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
        # Transmission Time + Propagation Time <= Deadline
        prob += (phi * config.BASE_CYCLE) + path_delay <= flow['deadline']

        # Solve using CBC Solver
        status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[status] == 'Optimal':
            flow['scheduled_offset'] = int(pulp.value(phi))
            flow['path'] = path
            return True
        return False

    def _reserve_resources(self, flow):
        """
        Reserves Fixed Resource Blocks (FB) in the global matrix.
        """
        start_cycle = flow['scheduled_offset']
        # Occupy resources hop-by-hop
        for i, (u, v) in enumerate(zip(flow['path'][:-1], flow['path'][1:])):
            if (u, v) in self.link_map:
                link_id = self.link_map[(u, v)]
                # Assuming 1 cycle consumption per hop
                cycle_idx = (start_cycle + i) % self.matrix.shape[1]

                # Occupy FB queue
                # Find the first free FB queue
                for q in range(config.TT_QUEUES):
                    if self.matrix[link_id, cycle_idx, q] == 0:
                        self.matrix[link_id, cycle_idx, q] = 1  # Mark as occupied
                        break