import cupy as cp
from db import N_UNITS, REV_UNIT_MAP

def plot_results(history, final_state, final_squads):
    # Nota: Per i grafici dettagliati (linea verde BP, risorse nel tempo)
    # avremmo dovuto salvare lo storico dello stato vincente tick per tick.
    # Dato che abbiamo solo lo stato finale e la history degli eventi,
    # mostriamo lo stato finale e stampiamo il Build Order.

    print("\n=== WINNING BUILD ORDER ===")
    for event in history:
        print(event)

    print("\n=== FINAL STATS ===")
    # Scarica da GPU
    fs = cp.asnumpy(final_state)
    print(f"Metal: {fs[0]:.1f}")
    print(f"Energy: {fs[1]:.1f} / {fs[2]:.1f}")
    print(f"Mex Count: {int(fs[4 + N_UNITS])}")

    print("\n=== UNIT COUNTS ===")
    for i in range(N_UNITS):
        count = int(fs[4 + i])
        if count > 0:
            print(f"{REV_UNIT_MAP[i]}: {count}")
