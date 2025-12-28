import cupy as cp
from cpu_branching_and_logic import BarOptimizer
from plot_results import plot_results

if __name__ == "__main__":
    # Verifica GPU
    try:
        dev = cp.cuda.Device(0)
        print(f"Using GPU: {dev.mem_info[1] / 1024**2:.0f} MB VRAM available")
    except Exception as e:
        print("CUDA non rilevato o errore Cupy. Assicurati di avere GPU NVIDIA e driver.")
        exit()

    sim = BarOptimizer()
    best_hist, best_state, best_squads = sim.run()

    plot_results(best_hist, best_state, best_squads)
