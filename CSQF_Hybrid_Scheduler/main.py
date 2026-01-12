# main.py
import networkx as nx
import numpy as np
import random
import config
from topology import create_internet2_topology
from tt_scheduler import TTScheduler
from avb_scheduler import AVBScheduler


def generate_traffic(G, n_tt=100, n_avb=500):
    """
    Generates synthetic hybrid traffic dataset [cite: 249-253].

    Args:
        G: Network topology
        n_tt: Number of TT flows
        n_avb: Number of AVB flows
    Returns:
        tt_flows, avb_flows: Lists of flow dictionaries
    """
    hosts = [n for n, d in G.nodes(data=True) if d.get('type') == 'host']

    # Generate TT Flows (Strictly Periodic)
    tt_flows = []
    for i in range(n_tt):
        src, dst = random.sample(hosts, 2)
        tt_flows.append({
            'id': i,
            'type': 'TT',
            'src': src,
            'dst': dst,
            'period': random.choice([4, 8, 16, 32]),  # Period (ms)
            'deadline': random.uniform(5, 15),  # Deadline (ms)
            'size': 100  # Payload size (bytes)
        })

    # Generate AVB Flows (Bursty, Bandwidth-sensitive)
    avb_flows = []
    for i in range(n_avb):
        src, dst = random.sample(hosts, 2)
        avb_flows.append({
            'id': i + n_tt,
            'type': 'AVB',
            'src': src,
            'dst': dst,
            'bandwidth': random.choice([10, 20, 50]),  # Bandwidth (Mbps)
            'deadline': 50  # Relaxed deadline
        })
    return tt_flows, avb_flows


def main():
    print("==================================================")
    print("   CSQF Hybrid Traffic Scheduling Demo (WAN)      ")
    print("==================================================")

    # 1. Initialize Network Topology
    G = create_internet2_topology()
    print(f"1. Topology Loaded: Internet2 ({G.number_of_nodes()} nodes, {G.number_of_edges()} links)")

    # Build Link Map (Edge -> ID) for matrix access
    link_map = {edge: i for i, edge in enumerate(G.edges())}
    num_links = len(link_map)
    num_cycles = int(config.HYPER_CYCLE / config.BASE_CYCLE)

    # 2. Initialize Global Resource Matrix [Dims: Links, Cycles, Queues]
    # Value: 0 = Free, 1 = Occupied
    global_schedule = np.zeros((num_links, num_cycles, config.TOTAL_QUEUES), dtype=np.int8)
    print(f"2. Resource Matrix Initialized: Shape {global_schedule.shape}")

    # 3. Generate Synthetic Traffic
    N_TT, N_AVB = 200, 1000  # Traffic Scale
    tt_flows, avb_flows = generate_traffic(G, N_TT, N_AVB)
    print(f"3. Traffic Generated: {len(tt_flows)} TT Flows, {len(avb_flows)} AVB Flows")

    # 4. Run Algorithm 1: TT Flow Scheduling (Strict Priority)
    tt_sched = TTScheduler(G, link_map, global_schedule)
    tt_success = tt_sched.schedule_flows(tt_flows)

    # 5. Run Algorithm 2: AVB Flow Scheduling (Dynamic Cycle + GPU)
    avb_sched = AVBScheduler(G, link_map, global_schedule)
    avb_success = avb_sched.schedule_flows(avb_flows)

    # 6. Output Final Statistics
    print("\n---------------- Scheduling Results ----------------")
    print(f"TT Flow Success Rate:  {tt_success}/{len(tt_flows)} ({tt_success / len(tt_flows) * 100:.1f}%)")
    print(f"AVB Flow Success Rate: {avb_success}/{len(avb_flows)} ({avb_success / len(avb_flows) * 100:.1f}%)")
    print("----------------------------------------------------")

    print("Simulation Completed.")


if __name__ == "__main__":
    main()