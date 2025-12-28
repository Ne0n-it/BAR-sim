import numpy as np
import cupy as cp
import copy
from tqdm import tqdm

from db import (
    N_UNITS, N_SQUADS, START_M, START_E, UNIT_MAP, REV_UNIT_MAP, DB,
    ST_IDLE, ST_INACTIVE, ST_BUILDING, LOOKUP_BP_COST, FACTORY_TYPE,
    ADV_FACTORY_TYPE, TOTAL_MEX_ON_MAP, MEX_DISTANCES, SQUAD_NAMES,
    BEAM_WIDTH, TIME_LIMIT, EVAL_INTERVAL, G_M_YIELD, G_M_COST, G_E_YIELD
)
from gpu_physics import run_gpu_physics

class BarOptimizer:
    def __init__(self):
        # Stato Globale: [Metal, Energy, MaxE, Time, ...Inventory(N_UNITS), Mex_Count_Tracker]
        self.sim_states = None

        # Squadre: [BP, Status, TaskUnitID, Progress, WaitTime, MissionID, LocID]
        self.squads = None

        self.histories = [] # Lista di liste per tracciare gli eventi (CPU side)
        self.n_sims = 0

    def initialize(self, initial_build_queue=None):
        """Inizializza con 1 simulazione base"""
        global_cols = 4 + N_UNITS + 1

        self.sim_states = cp.zeros((1, global_cols), dtype=cp.float32)
        self.sim_states[0, 0] = START_M
        self.sim_states[0, 1] = START_E
        self.sim_states[0, 2] = 1000 # Max E iniziale
        self.sim_states[0, 3] = 0    # Time
        self.sim_states[0, 4 + UNIT_MAP['Commander']] = 1

        self.squads = cp.zeros((1, N_SQUADS, 7), dtype=cp.float32)

        self.squads[0, 0, 0] = DB['Commander']['BP']
        self.squads[0, 0, 1] = ST_IDLE
        self.squads[0, 0, 2] = -1
        self.squads[0, 0, 5] = -1

        self.histories = [[]]
        self.n_sims = 1

    def run_gpu_physics(self, duration_sec):
        """Wrapper to call the GPU physics engine."""
        self.sim_states, self.squads = run_gpu_physics(self.sim_states, self.squads, duration_sec)

    def cpu_branching_and_logic(self):
        np_states = cp.asnumpy(self.sim_states)
        np_squads = cp.asnumpy(self.squads)

        new_states_list = []
        new_squads_list = []
        new_histories_list = []

        for i in range(self.n_sims):
            parent_st = np_states[i]
            parent_sq = np_squads[i]
            parent_hist = self.histories[i]
            current_time = parent_st[3]

            # 1. GESTIONE COMPLETAMENTI
            for s_idx in range(N_SQUADS):
                s_props = parent_sq[s_idx]
                status = s_props[1]
                task_id = int(s_props[2])
                progress = s_props[3]

                if status == ST_BUILDING:
                    needed_bp = LOOKUP_BP_COST[task_id]
                    if progress >= needed_bp:
                        unit_name = REV_UNIT_MAP[task_id]
                        parent_st[4 + task_id] += 1

                        if unit_name == FACTORY_TYPE:
                            parent_sq[1][0] = DB[FACTORY_TYPE]['BP']
                            parent_sq[1][1] = ST_IDLE
                        elif unit_name == 'Adv_VehLab':
                             parent_sq[5][0] = DB['Adv_VehLab']['BP']
                             parent_sq[5][1] = ST_IDLE
                        elif unit_name == 'Adv_ConVeh':
                             parent_sq[6][0] = DB['Adv_ConVeh']['BP']
                             parent_sq[6][1] = ST_IDLE
                        elif unit_name in ['ConVeh', 'ConBot', 'ConAir']:
                             for slot in range(2, 5):
                                 if parent_sq[slot][1] == ST_INACTIVE:
                                     parent_sq[slot][0] = DB[unit_name]['BP']
                                     parent_sq[slot][1] = ST_IDLE
                                     break

                        if unit_name == 'Mex':
                            parent_st[4 + N_UNITS] += 1
                        elif unit_name == 'EStorage':
                            parent_st[2] += DB['EStorage']['storage']

                        s_props[1] = ST_IDLE
                        s_props[2] = -1
                        s_props[3] = 0
                        s_props[4] = 0

                        parent_hist.append(f"{int(current_time)}: Completed {unit_name} by {SQUAD_NAMES[s_idx]}")

            # 2. CASCATA DI BRANCHING
            step_universes = [(parent_st, parent_sq, parent_hist)]

            for s_idx in range(N_SQUADS):
                next_step_universes = []

                for p_st, p_sq, p_hist in step_universes:
                    if p_sq[s_idx][1] != ST_IDLE:
                        next_step_universes.append((p_st, p_sq, p_hist))
                        continue

                    options = []
                    squad_name = SQUAD_NAMES[s_idx]

                    builder_type = None
                    if s_idx == 0: builder_type = 'Commander'
                    elif 1 <= s_idx <= 4: builder_type = 'ConVeh' if s_idx > 1 else 'VehLab'
                    elif s_idx == 5: builder_type = 'Adv_VehLab'
                    elif s_idx == 6: builder_type = 'Adv_ConVeh'

                    if 'Lab' in builder_type:
                        raw_opts = DB[builder_type].get('unit_options', [])
                        for opt in raw_opts:
                            options.append(opt)
                    else:
                        raw_opts = DB[builder_type].get('build_options', [])
                        for opt in raw_opts:
                            if opt == 'EStorage' and p_st[4 + UNIT_MAP['EStorage']] > 0: continue
                            if opt.endswith('Lab'):
                                if opt != FACTORY_TYPE and opt != ADV_FACTORY_TYPE: continue
                                if opt == FACTORY_TYPE and (p_sq[1][0] > 0 or p_st[4+UNIT_MAP[opt]] > 0): continue
                                if opt == ADV_FACTORY_TYPE and p_sq[5][0] > 0: continue
                            if opt == 'Mex' and int(p_st[4 + N_UNITS]) >= TOTAL_MEX_ON_MAP: continue
                            if opt == 'Adv_Mex' and p_st[4+UNIT_MAP['Mex']] == 0: continue
                            options.append(opt)

                    if not options:
                        next_step_universes.append((p_st, p_sq, p_hist))
                    else:
                        for opt in options:
                            branch_st = p_st.copy()
                            branch_sq = p_sq.copy()
                            branch_hist = copy.copy(p_hist)

                            t_idx = UNIT_MAP[opt]
                            wait_time = 2

                            if opt == 'Mex':
                                curr_m = int(branch_st[4 + N_UNITS])
                                if curr_m < TOTAL_MEX_ON_MAP:
                                    wait_time += MEX_DISTANCES[curr_m]

                            branch_sq[s_idx][1] = ST_BUILDING
                            branch_sq[s_idx][2] = t_idx
                            branch_sq[s_idx][3] = 0
                            branch_sq[s_idx][4] = wait_time

                            branch_hist.append(f"{int(current_time)}: {squad_name} starts {opt}")
                            next_step_universes.append((branch_st, branch_sq, branch_hist))

                step_universes = next_step_universes

            for final_st, final_sq, final_hist in step_universes:
                new_states_list.append(final_st)
                new_squads_list.append(final_sq)
                new_histories_list.append(final_hist)

        if not new_states_list: return

        self.sim_states = cp.array(new_states_list, dtype=cp.float32)
        self.squads = cp.array(new_squads_list, dtype=cp.float32)
        self.histories = new_histories_list
        self.n_sims = len(new_states_list)

    def prune(self):
        if self.n_sims <= BEAM_WIDTH:
            return

        scores = cp.zeros(self.n_sims, dtype=cp.float32)
        m_curr = self.sim_states[:, 0]
        e_curr = self.sim_states[:, 1]
        max_e = self.sim_states[:, 2]
        inventory = self.sim_states[:, 4:4+N_UNITS]

        m_income = cp.sum(inventory * G_M_YIELD, axis=1) + 2.0
        total_bp = cp.sum(self.squads[:, :, 0], axis=1)

        est_cons = total_bp * 0.025
        eff_throughput = cp.minimum(m_income, est_cons)

        m_invested = cp.sum(inventory * G_M_COST, axis=1)

        penalty = cp.zeros(self.n_sims, dtype=cp.float32)
        penalty += cp.where(e_curr < 50, 500, 0)

        has_storage = inventory[:, UNIT_MAP['EStorage']] > 0
        penalty += cp.where((e_curr > max_e * 0.9) & has_storage, 100, 0)
        penalty += cp.where((e_curr > 1000) & (~has_storage), 200, 0)

        scores = (eff_throughput * 100) + m_invested - penalty + total_bp

        sorted_indices = cp.argsort(scores)[::-1]
        keep_indices = sorted_indices[:BEAM_WIDTH]

        self.sim_states = self.sim_states[keep_indices]
        self.squads = self.squads[keep_indices]

        keep_indices_np = cp.asnumpy(keep_indices)
        self.histories = [self.histories[i] for i in keep_indices_np]
        self.n_sims = len(self.histories)

    def run(self):
        self.initialize()

        for t in tqdm(range(0, TIME_LIMIT, EVAL_INTERVAL), desc="Simulating"):
            self.run_gpu_physics(EVAL_INTERVAL)
            self.cpu_branching_and_logic()

            if self.n_sims > BEAM_WIDTH:
                self.prune()

            if self.n_sims == 0:
                print("Extinction Event.")
                break

        self.prune()
        return self.histories[0], self.sim_states[0], self.squads[0]
