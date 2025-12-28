import cupy as cp
from db import (
    N_UNITS, G_M_YIELD, G_E_YIELD, G_M_COST, G_E_COST, G_BP_COST, G_E_DRAIN,
    ST_BUILDING, DB
)

def run_gpu_physics(sim_states, squads, duration_sec):
    """
    Esegue la simulazione fisica sulla GPU per N secondi.
    Calcola drain, income, stalli e progressi.
    """
    dt = 1.0 # step di 1 secondo
    steps = int(duration_sec)

    # Alias per leggibilità (viste, non copie)
    metal = sim_states[:, 0]
    energy = sim_states[:, 1]
    max_e = sim_states[:, 2]
    time_sim = sim_states[:, 3]
    inventory = sim_states[:, 4:4+N_UNITS]

    # Squad aliases
    sq_bp = squads[:, :, 0]
    sq_status = squads[:, :, 1]
    sq_task = squads[:, :, 2].astype(cp.int32)
    sq_progress = squads[:, :, 3]
    sq_wait = squads[:, :, 4]

    for _ in range(steps):
        # 1. Calcolo Income
        base_m_income = cp.sum(inventory * G_M_YIELD, axis=1) + DB['Commander']['M']
        base_e_income = cp.sum(inventory * G_E_YIELD, axis=1) + DB['Commander']['E']

        # 2. Calcolo Demand (Drain)
        is_building = (sq_status == ST_BUILDING)

        task_m_cost = cp.take(G_M_COST, sq_task)
        task_e_cost = cp.take(G_E_COST, sq_task)
        task_bp_cost = cp.take(G_BP_COST, sq_task)

        task_bp_cost = cp.maximum(task_bp_cost, 1.0)

        drain_ratio_m = task_m_cost / task_bp_cost
        drain_ratio_e = task_e_cost / task_bp_cost

        current_drain_m = cp.sum(cp.where(is_building, sq_bp * drain_ratio_m, 0), axis=1)
        current_drain_e = cp.sum(cp.where(is_building, sq_bp * drain_ratio_e, 0), axis=1)

        passive_e_drain = cp.sum(inventory * G_E_DRAIN, axis=1)
        total_drain_e = current_drain_e + passive_e_drain

        # 3. Gestione Stalli (Logic Cascade)
        net_e = energy + base_e_income - total_drain_e

        available_e = energy + base_e_income
        safe_demand_e = cp.maximum(total_drain_e, 0.001)
        e_efficiency = cp.clip(available_e / safe_demand_e, 0.0, 1.0)

        actual_m_income = base_m_income * ((e_efficiency + 1.0) / 2.0)

        available_m = metal + actual_m_income
        safe_demand_m = cp.maximum(current_drain_m, 0.001)
        m_efficiency = cp.clip(available_m / safe_demand_m, 0.0, 1.0)

        global_efficiency = cp.minimum(e_efficiency, m_efficiency)

        # 4. Update Risorse
        spent_m = current_drain_m * global_efficiency
        spent_e = total_drain_e * e_efficiency

        new_m = metal + actual_m_income - spent_m
        new_e = energy + base_e_income - spent_e

        sim_states[:, 0] = cp.clip(new_m, 0, 100000)
        sim_states[:, 1] = cp.clip(new_e, 0, max_e)

        # 5. Avanzamento Costruzione e Tempo
        squads[:, :, 4] = cp.maximum(sq_wait - 1, 0)

        global_eff_reshaped = global_efficiency[:, None]
        bp_applied = sq_bp * global_eff_reshaped
        active_builds = (sq_status == ST_BUILDING) & (sq_wait <= 0)

        squads[:, :, 3] = cp.where(active_builds, sq_progress + bp_applied, sq_progress)

        sim_states[:, 3] += 1 # Time increment

    return sim_states, squads
