import numpy as np
import cupy as cp
import time
from tqdm import tqdm

# ==========================================
# 1. CONFIGURAZIONE E DATABASE
# ==========================================

# Parametri Simulazione
BEAM_WIDTH = 120000
MAX_CANDIDATES = 3000000
TIME_LIMIT = 5 * 60
EVAL_INTERVAL = 900
WIND_LEVEL = 14
MEX_METAL_YIELD = 2.0
MEX_DISTANCES = [0, 5, 5, 10, 10, 15, 15, 20, 20, 25]
GEO_DISTANCES = [15, 10, 10]

# Coda di costruzione iniziale
INITIAL_BUILD_QUEUE = [
    ('Commander', 0, 'Mex'),
    ('Commander', 0, 'Solar'),
    ('Commander', 0, 'Vehicle_Plant'),
    ('FactoryT1', 0, 'ConVeh'),
]

# --- DATABASE ARMADA COMPLETO ---
ADB = {
    'Commander': {'bp': 0, 'm': 0, 'e': 0, 'BP': 300, 'M': 2, 'E': 25, 'build_options': ['Mex', 'Solar', 'Wind', 'Vehicle_Plant'], 'category': 'BUILDER'},
    'Solar': {'m': 155, 'e': 0, 'bp': 2600, 'E': 20, 'M': 0, 'BP': 0, 'e_drain': 0, 'category': 'ECONOMY'},
    'Adv_Solar': {'m': 350, 'e': 5000, 'bp': 7950, 'E': 75, 'M': 0, 'BP': 0, 'e_drain': 0, 'category': 'ECONOMY'},
    'Wind': {'m': 40, 'e': 175, 'bp': 1600, 'E': WIND_LEVEL, 'M': 0, 'BP': 0, 'e_drain': 0, 'category': 'ECONOMY'},
    'FuR': {'m': 4300, 'e': 21000, 'bp': 70000, 'E': 1000, 'M': 0, 'BP': 0, 'e_drain': 0, 'category': 'ECONOMY'},
    'AFur': {'m': 9700, 'e': 69000, 'bp': 3125000, 'E': 3000, 'M': 0, 'BP': 0, 'e_drain': 0, 'category': 'ECONOMY'},
    'Mex': {'m': 50, 'e': 500, 'bp': 1800, 'E': 0, 'M': MEX_METAL_YIELD, 'BP': 0, 'e_drain': 3, 'category': 'ECONOMY'},
    'Adv_Mex': {'m': 620, 'e': 7700, 'bp': 14900, 'E': 0, 'M': MEX_METAL_YIELD * 4, 'BP': 0, 'e_drain': 20, 'category': 'ECONOMY'},
    'EStorage': {'m': 260, 'e': 1700, 'bp': 4110, 'E': 0, 'M': 0, 'BP': 0, 'storage': 6000, 'e_drain': 0, 'category': 'ECONOMY'},
    'GEO': {'m': 560, 'e': 13000, 'bp': 13100, 'E': 300, 'M': 0, 'BP': 0, 'e_drain': 0, 'category': 'ECONOMY'},
    'Adv_GEO': {'m': 1600, 'e': 27000, 'bp': 33300, 'E': 1250, 'M': 0, 'BP': 0, 'e_drain': 0, 'category': 'ECONOMY'},
    'EConv': {'m': 1, 'e': 1150, 'bp': 2600, 'E': 0, 'M': 1, 'BP': 0, 'e_drain': 70, 'category': 'ECONOMY'},
    'Adv_EConv': {'m': 380, 'e': 21000, 'bp': 35000, 'E': 0, 'M': 10.3, 'BP': 0, 'e_drain': 600, 'category': 'ECONOMY'},
    'Vehicle_Plant': {'m': 590, 'e': 1550, 'bp': 4280, 'BP': 150, 'M': 0, 'E': 0, 'e_drain': 0, 'unit_options': ['ConVeh', 'Offensive'], 'category': 'FACTORY'},
    'Advanced_Vehicle_Plant_T2': {'m': 2900, 'e': 14000, 'bp': 33800, 'BP': 300, 'M': 0, 'E': 0, 'e_drain': 0, 'unit_options': ['Adv_ConVeh', 'Adv_Offensive'], 'category': 'FACTORY'},
    'Experimental_Gantry_T3': {'m': 7900, 'e': 58000, 'bp': 131800, 'BP': 600, 'M': 0, 'E': 0, 'e_drain': 0, 'unit_options': ['Titan'], 'category': 'FACTORY'},
    'ConVeh': {'m': 135, 'e': 1950, 'bp': 4100, 'BP': 90, 'e_drain': 0, 'build_options': ['Mex', 'Solar', 'Wind', 'EStorage', 'Adv_Solar', 'GEO', 'EConv'], 'category': 'BUILDER'},
    'Adv_ConVeh': {'m': 550, 'e': 6800, 'bp': 12400, 'BP': 250, 'e_drain': 0, 'build_options': ['Adv_Mex', 'Adv_Solar', 'FuR', 'AFur', 'Adv_GEO', 'Adv_EConv'], 'category': 'BUILDER'},
    'ConTurret': {'m': 210, 'e': 3200, 'bp': 5300, 'BP': 200, 'e_drain': 0, 'build_options': [], 'category': 'BUILDER'},
    'Offensive': {'m': 200, 'e': 2250, 'bp': 3500, 'BP': 0, 'e_drain': 0, 'category': 'OFFENSIVE'},
    'Adv_Offensive': {'m': 950, 'e': 13000, 'bp': 17200, 'BP': 0, 'e_drain': 0, 'category': 'OFFENSIVE'},
    'Titan' : {'m': 13500, 'e': 286000, 'bp': 276000, 'BP': 0, 'e_drain': 0, 'category': 'OFFENSIVE'}
}

# Mapping ID
UNIT_NAMES = list(ADB.keys())
UNIT_MAP = {name: i for i, name in enumerate(UNIT_NAMES)}
N_UNITS = len(UNIT_NAMES)

SQUAD_NAMES = ['Commander', 'FactoryT1', 'ConSquad1', 'ConSquad2', 'ConSquad3', 'FactoryT2', 'Adv_ConSquad', 'Gantry', 'Unassigned']
N_SQUADS = len(SQUAD_NAMES)
SQ_MAP = {name: i for i, name in enumerate(SQUAD_NAMES)}

# --- COMPILAZIONE MATRIX GPU (Host -> Device) ---
IDX_M_COST = 0
IDX_E_COST = 1
IDX_BP_COST = 2
IDX_M_YIELD = 3
IDX_E_YIELD = 4
IDX_E_DRAIN = 5
IDX_BP_ADD = 6
IDX_STORAGE = 7
IDX_CATEGORY = 8
STATS_COLS = 9

HOST_STATS = np.zeros((N_UNITS, STATS_COLS), dtype=np.float32)

CATEGORY_MAP = {'BUILDER': 1, 'FACTORY': 2, 'OFFENSIVE': 3, 'ECONOMY': 4}

for name, data in ADB.items():
    idx = UNIT_MAP[name]
    HOST_STATS[idx, IDX_M_COST] = data.get('m', 0)
    HOST_STATS[idx, IDX_E_COST] = data.get('e', 0)
    HOST_STATS[idx, IDX_BP_COST] = data.get('bp', 1)
    HOST_STATS[idx, IDX_M_YIELD] = data.get('M', 0)
    HOST_STATS[idx, IDX_E_YIELD] = data.get('E', 0)
    HOST_STATS[idx, IDX_E_DRAIN] = data.get('e_drain', 0)
    HOST_STATS[idx, IDX_BP_ADD] = data.get('BP', 0)
    HOST_STATS[idx, IDX_STORAGE] = data.get('storage', 0)
    HOST_STATS[idx, IDX_CATEGORY] = CATEGORY_MAP.get(data.get('category', ''), 0)

    if name == 'Commander':
        HOST_STATS[idx, IDX_STORAGE] = 1000

G_STATS = cp.asarray(HOST_STATS)

# --- MATRICE OPZIONI (Lookup Table) ---
MAX_OPTS = 12
HOST_BUILD_OPTS = np.full((N_SQUADS, MAX_OPTS), -1, dtype=np.int32)

def register_options(squad_name, options_list):
    if squad_name not in SQ_MAP: return
    idx = SQ_MAP[squad_name]
    valid_opts = [UNIT_MAP[u] for u in options_list if u in UNIT_MAP]
    if WIND_LEVEL < 6:
        wind_idx = UNIT_MAP.get('Wind', -1)
        if wind_idx != -1:
            valid_opts = [u for u in valid_opts if u != wind_idx]

    count = len(valid_opts)
    if count > MAX_OPTS:
        valid_opts = valid_opts[:MAX_OPTS]
    HOST_BUILD_OPTS[idx, :len(valid_opts)] = valid_opts

register_options('Commander', ADB['Commander']['build_options'])
register_options('FactoryT1', ADB['Vehicle_Plant']['unit_options'])
register_options('FactoryT2', ADB['Advanced_Vehicle_Plant_T2']['unit_options'])
register_options('Gantry', ADB['Experimental_Gantry_T3']['unit_options'])
register_options('ConSquad1', ADB['ConVeh']['build_options'])
register_options('ConSquad2', ADB['ConVeh']['build_options'])
register_options('ConSquad3', ADB['ConVeh']['build_options'])
register_options('Adv_ConSquad', ADB['Adv_ConVeh']['build_options'])

G_BUILD_OPTS = cp.asarray(HOST_BUILD_OPTS)

# Stati Squadra
ST_INACTIVE = 0
ST_IDLE = 1
ST_BUILDING = 2
ST_TRAVEL = 3
ST_WAITING = 4
ST_UNASSIGNED_BUILDER = 5

# ==========================================
# 2. CUSTOM CUDA KERNEL
# ==========================================

physics_kernel = cp.ElementwiseKernel(
    'float32 m_curr, float32 e_curr, float32 max_e, float32 m_income, float32 e_income, float32 m_drain_build, float32 e_drain_build, float32 e_drain_passive',
    'float32 m_new, float32 e_new, float32 e_eff',
    '''
    float m_avail = m_curr + m_income;
    float e_avail = e_curr + e_income;

    float e_demand = e_drain_build + e_drain_passive;
    float current_e_eff = 1.0;

    if (e_demand > 0.0001) {
        current_e_eff = e_avail / e_demand;
        if (current_e_eff > 1.0) current_e_eff = 1.0;
    }

    float m_real_demand = m_drain_build * current_e_eff;
    float m_eff = 1.0;

    if (m_real_demand > 0.0001) {
        m_eff = m_avail / m_real_demand;
        if (m_eff > 1.0) m_eff = 1.0;
    }

    float m_spent = m_real_demand * m_eff;
    m_new = m_avail - m_spent;
    if (m_new < 0) m_new = 0;
    if (m_new > 100000) m_new = 100000;

    float e_spent = e_demand * current_e_eff;
    e_new = e_avail - e_spent;
    if (e_new < 0) e_new = 0;
    if (e_new > max_e) e_new = max_e;

    e_eff = current_e_eff * m_eff;
    ''',
    'physics_update_kernel'
)

# ==========================================
# 3. MOTORE DI SIMULAZIONE
# ==========================================

class BarSimOptimized:
    def __init__(self):
        self.total_sims = 1

        self.res_buf = cp.zeros((MAX_CANDIDATES, 4), dtype=cp.float32)
        self.inv_buf = cp.zeros((MAX_CANDIDATES, N_UNITS), dtype=cp.int32)
        self.sq_buf = cp.zeros((MAX_CANDIDATES, N_SQUADS, 7), dtype=cp.float32)

        self.res_buf[0] = cp.array([1000, 1000, 1000, 0])
        self.inv_buf[0, UNIT_MAP['Commander']] = 1

        self.sq_buf[0, SQ_MAP['Commander'], 0] = ST_IDLE
        self.sq_buf[0, SQ_MAP['Commander'], 1] = ADB['Commander']['BP']
        self.sq_buf[0, SQ_MAP['Commander'], 2] = -1

        self.history_parents = []
        self.history_actions = []

    def _apply_initial_build_queue(self):
        for squad_name, action_type, unit_name in INITIAL_BUILD_QUEUE:
            squad_idx = SQ_MAP.get(squad_name)
            unit_idx = UNIT_MAP.get(unit_name)

            if squad_idx is None or unit_idx is None:
                print(f"WARNING: Invalid squad or unit in initial build queue: {squad_name}, {unit_name}")
                continue

            self.sq_buf[0, squad_idx, 0] = ST_BUILDING
            self.sq_buf[0, squad_idx, 2] = unit_idx

            initial_unit_count = self.inv_buf[0, unit_idx]
            while self.inv_buf[0, unit_idx] == initial_unit_count and self.res_buf[0, 3] < TIME_LIMIT:
                self.physics_step()
                self.resolve_completions()

    def physics_step(self):
        n = self.total_sims
        if n == 0: return

        inv_f = self.inv_buf[:n].astype(cp.float32)
        m_inc = cp.dot(inv_f, G_STATS[:, IDX_M_YIELD])
        e_inc = cp.dot(inv_f, G_STATS[:, IDX_E_YIELD])

        sq_view = self.sq_buf[:n]
        is_building = (sq_view[:, :, 0] == ST_BUILDING)
        task_idx = sq_view[:, :, 2].astype(cp.int32)
        squad_bp = sq_view[:, :, 1]

        safe_tasks = cp.maximum(task_idx, 0)
        target_stats = G_STATS[safe_tasks]

        bp_req = target_stats[:, :, IDX_BP_COST]
        m_req = target_stats[:, :, IDX_M_COST]
        e_req = target_stats[:, :, IDX_E_COST]

        safe_bp_req = bp_req + 1e-5
        ratio_e = e_req / safe_bp_req
        ratio_m = m_req / safe_bp_req

        drain_e_build = cp.sum(ratio_e * squad_bp * is_building, axis=1)
        drain_m_build = cp.sum(ratio_m * squad_bp * is_building, axis=1)
        drain_e_passive = cp.dot(inv_f, G_STATS[:, IDX_E_DRAIN])

        m_curr = self.res_buf[:n, 0]
        e_curr = self.res_buf[:n, 1]
        max_e = self.res_buf[:n, 2]

        m_new, e_new, eff_factor = physics_kernel(
            m_curr, e_curr, max_e, m_inc, e_inc,
            drain_m_build, drain_e_build, drain_e_passive
        )

        self.res_buf[:n, 0] = m_new
        self.res_buf[:n, 1] = e_new
        self.res_buf[:n, 3] += 1

        eff_expanded = eff_factor[:, None]
        self.sq_buf[:n, :, 3] += (squad_bp * is_building * eff_expanded)

        is_travel = (sq_view[:, :, 0] == ST_TRAVEL)
        self.sq_buf[:n, :, 4] -= 1 * is_travel

        is_waiting = (sq_view[:, :, 0] == ST_WAITING)
        self.sq_buf[:n, :, 4] -= 1 * is_waiting

    def resolve_completions(self):
        n = self.total_sims
        sq_view = self.sq_buf[:n]

        task_idx = sq_view[:, :, 2].astype(cp.int32)
        safe_tasks = cp.maximum(task_idx, 0)
        bp_costs = G_STATS[safe_tasks, IDX_BP_COST]

        is_building = (sq_view[:, :, 0] == ST_BUILDING)
        finished = is_building & (sq_view[:, :, 3] >= bp_costs)

        if not cp.any(finished): return

        sim_idxs, sq_idxs = cp.where(finished)
        unit_ids = task_idx[sim_idxs, sq_idxs]

        unit_categories = G_STATS[unit_ids, IDX_CATEGORY]
        is_builder = (unit_categories == CATEGORY_MAP['BUILDER'])

        if cp.any(is_builder):
            builder_sim_idxs = sim_idxs[is_builder]
            builder_unit_ids = unit_ids[is_builder]
            unassigned_squad_idx = SQ_MAP['Unassigned']
            self.sq_buf[builder_sim_idxs, unassigned_squad_idx, 0] = ST_UNASSIGNED_BUILDER
            self.sq_buf[builder_sim_idxs, unassigned_squad_idx, 2] = builder_unit_ids.astype(cp.float32)

        non_builder_mask = ~is_builder
        if cp.any(non_builder_mask):
            cp.add.at(self.inv_buf, (sim_idxs[non_builder_mask], unit_ids[non_builder_mask]), 1)

        added_storage = G_STATS[unit_ids, IDX_STORAGE]
        cp.add.at(self.res_buf[:, 2], sim_idxs, added_storage)

        self.sq_buf[sim_idxs, sq_idxs, 0] = ST_IDLE
        self.sq_buf[sim_idxs, sq_idxs, 2] = -1
        self.sq_buf[sim_idxs, sq_idxs, 3] = 0
        self.sq_buf[sim_idxs, sq_idxs, 6] = unit_ids.astype(cp.float32)

    def branching_step(self):
        n = self.total_sims

        self.sq_buf[:n][(self.sq_buf[:n, :, 0] == ST_TRAVEL) & (self.sq_buf[:n, :, 4] <= 0)] = ST_IDLE
        self.sq_buf[:n][(self.sq_buf[:n, :, 0] == ST_WAITING) & (self.sq_buf[:n, :, 4] <= 0)] = ST_IDLE

        unassigned_squad_idx = SQ_MAP['Unassigned']
        is_unassigned_mask = (self.sq_buf[:n, unassigned_squad_idx, 0] == ST_UNASSIGNED_BUILDER)

        is_idle_mask = (self.sq_buf[:n, :, 0] == ST_IDLE)
        sims_with_idle_squads_mask = cp.any(is_idle_mask, axis=1) & ~is_unassigned_mask

        all_parent_idxs, all_squads, all_actions = [], [], []

        unassigned_sim_indices = cp.where(is_unassigned_mask)[0]
        if len(unassigned_sim_indices) > 0:
            num_unassigned = len(unassigned_sim_indices)
            target_squad_options = cp.asarray([SQ_MAP['ConSquad1'], SQ_MAP['ConSquad2'], SQ_MAP['ConSquad3']], dtype=cp.int32)
            num_options = len(target_squad_options)

            all_parent_idxs.append(cp.repeat(unassigned_sim_indices, num_options))
            all_squads.append(cp.full(num_unassigned * num_options, unassigned_squad_idx, dtype=cp.int32))
            all_actions.append(cp.tile(target_squad_options, num_unassigned))

        idle_sim_indices = cp.where(sims_with_idle_squads_mask)[0]
        if len(idle_sim_indices) > 0:
            first_idle_sq = cp.argmax(is_idle_mask[idle_sim_indices], axis=1)
            opts_matrix = G_BUILD_OPTS[first_idle_sq]

            mex_idx = UNIT_MAP['Mex']
            adv_mex_idx = UNIT_MAP.get('Adv_Mex', -1)
            mex_counts = self.inv_buf[idle_sim_indices, mex_idx] + self.inv_buf[idle_sim_indices, adv_mex_idx] if adv_mex_idx != -1 else self.inv_buf[idle_sim_indices, mex_idx]
            at_mex_limit = (mex_counts >= len(MEX_DISTANCES))
            opts_matrix[at_mex_limit, :] = cp.where(opts_matrix[at_mex_limit, :] == mex_idx, -1, opts_matrix[at_mex_limit, :])

            factory_squads = cp.asarray([SQ_MAP['FactoryT1'], SQ_MAP['FactoryT2'], SQ_MAP['Gantry']])
            is_factory_mask = cp.isin(first_idle_sq, factory_squads)

            valid_mask = (opts_matrix != -1)
            counts_gpu = cp.sum(valid_mask, axis=1) + is_factory_mask
            total_idle_branches = int(cp.sum(counts_gpu))

            if total_idle_branches > 0:
                ends = cp.cumsum(counts_gpu)
                map_idxs = cp.searchsorted(ends, cp.arange(total_idle_branches, dtype=cp.int32), side='right')

                all_parent_idxs.append(idle_sim_indices[map_idxs])
                all_squads.append(first_idle_sq[map_idxs])

                actions_idle = cp.full(total_idle_branches, -1, dtype=cp.int32)
                wait_action_mask = (cp.arange(total_idle_branches) == ends[map_idxs] - 1) & is_factory_mask[map_idxs]
                actions_idle[~wait_action_mask] = opts_matrix[valid_mask]
                all_actions.append(actions_idle)

        if not all_parent_idxs:
            self.history_parents.append(cp.arange(n, dtype=cp.int32))
            self.history_actions.append(cp.full(n, -1, dtype=cp.int16))
            return

        parent_idxs = cp.concatenate(all_parent_idxs)
        squads = cp.concatenate(all_squads)
        actions = cp.concatenate(all_actions)
        total_new = len(parent_idxs)

        if total_new == 0 or n + total_new > MAX_CANDIDATES:
            self.history_parents.append(cp.arange(n, dtype=cp.int32))
            self.history_actions.append(cp.full(n, -1, dtype=cp.int16))
            return

        start, end = n, n + total_new
        new_range = cp.arange(start, end)
        self.res_buf[start:end] = self.res_buf[parent_idxs]
        self.inv_buf[start:end] = self.inv_buf[parent_idxs]
        self.sq_buf[start:end] = self.sq_buf[parent_idxs]

        assign_builder_mask = (squads == unassigned_squad_idx)
        if cp.any(assign_builder_mask):
            indices = new_range[assign_builder_mask]
            target_squads = actions[assign_builder_mask]
            builder_unit_ids = self.sq_buf[indices, unassigned_squad_idx, 2].astype(cp.int32)
            builder_bps = G_STATS[builder_unit_ids, IDX_BP_ADD]

            cp.add.at(self.sq_buf, (indices, target_squads, 1), builder_bps)
            self.sq_buf[indices, target_squads, 0] = cp.maximum(self.sq_buf[indices, target_squads, 0], ST_IDLE)
            self.sq_buf[indices, unassigned_squad_idx, 0] = ST_INACTIVE
            self.sq_buf[indices, unassigned_squad_idx, 2] = -1

        build_wait_mask = ~assign_builder_mask
        if cp.any(build_wait_mask):
            indices, p_idxs = new_range[build_wait_mask], parent_idxs[build_wait_mask]
            squads_bw, actions_bw = squads[build_wait_mask], actions[build_wait_mask]

            waits = cp.zeros(len(actions_bw), dtype=cp.float32)
            mex_idx, adv_mex_idx = UNIT_MAP['Mex'], UNIT_MAP.get('Adv_Mex', -1)
            is_mex_action = (actions_bw == mex_idx) | (actions_bw == adv_mex_idx if adv_mex_idx !=-1 else False)
            if cp.any(is_mex_action):
                mex_counts = self.inv_buf[p_idxs[is_mex_action], mex_idx] + (self.inv_buf[p_idxs[is_mex_action], adv_mex_idx] if adv_mex_idx !=-1 else 0)
                valid_counts = cp.minimum(mex_counts, len(MEX_DISTANCES) - 1)
                waits[is_mex_action] = cp.asarray(MEX_DISTANCES, dtype=cp.float32)[valid_counts]

            geo_idx = UNIT_MAP.get('GEO', -1)
            is_geo_action = (actions_bw == geo_idx)
            if cp.any(is_geo_action):
                geo_counts = self.inv_buf[p_idxs[is_geo_action], geo_idx]
                valid_counts = cp.minimum(geo_counts, len(GEO_DISTANCES) - 1)
                waits[is_geo_action] = cp.asarray(GEO_DISTANCES, dtype=cp.float32)[valid_counts]

            build_actions_mask = (actions_bw != -1)
            last_built = self.sq_buf[indices[build_actions_mask], squads_bw[build_actions_mask], 6]
            unit_changed = (last_built != actions_bw[build_actions_mask]) & (last_built != 0)
            generic_travel_mask = (waits[build_actions_mask] == 0) & unit_changed
            waits[build_actions_mask] = cp.where(generic_travel_mask, 5, waits[build_actions_mask])

            wait_mask_bw = (actions_bw == -1)
            self.sq_buf[indices[wait_mask_bw], squads_bw[wait_mask_bw], 0] = ST_WAITING
            self.sq_buf[indices[wait_mask_bw], squads_bw[wait_mask_bw], 4] = 20

            build_mask_bw = ~wait_mask_bw
            is_travel_mask = (waits > 0) & build_mask_bw
            self.sq_buf[indices[build_mask_bw], squads_bw[build_mask_bw], 0] = ST_BUILDING
            self.sq_buf[indices[build_mask_bw], squads_bw[build_mask_bw], 2] = actions_bw[build_mask_bw]
            self.sq_buf[indices[build_mask_bw], squads_bw[build_mask_bw], 3] = 0

            travel_indices = indices[build_mask_bw][is_travel_mask[build_mask_bw]]
            travel_squads = squads_bw[build_mask_bw][is_travel_mask[build_mask_bw]]
            travel_waits = waits[build_mask_bw][is_travel_mask[build_mask_bw]]
            self.sq_buf[travel_indices, travel_squads, 0] = ST_TRAVEL
            self.sq_buf[travel_indices, travel_squads, 4] = travel_waits

        self.total_sims = end

        sims_branched_from = cp.unique(parent_idxs)
        sims_not_branched = cp.ones(n, dtype=bool)
        sims_not_branched[sims_branched_from] = False
        existing_indices = cp.where(sims_not_branched)[0]

        full_parents = cp.concatenate((existing_indices, parent_idxs))
        act_exist = cp.full(len(existing_indices), -1, dtype=cp.int16)
        act_new = ((squads.astype(cp.int16) << 8) | actions.astype(cp.int16))
        full_actions = cp.concatenate((act_exist, act_new))

        self.history_parents.append(full_parents)
        self.history_actions.append(full_actions)

    def prune(self):
        n = self.total_sims
        if n <= BEAM_WIDTH: return
        inv_f = self.inv_buf[:n].astype(cp.float32)
        m_yield = cp.dot(inv_f, G_STATS[:, IDX_M_YIELD])
        m_spent_value = cp.dot(inv_f, G_STATS[:, IDX_M_COST])
        score = (m_yield * 2000) + m_spent_value
        m_curr = self.res_buf[:n, 0]
        score -= (m_curr * 2.0)
        e_curr = self.res_buf[:n, 1]
        e_max = self.res_buf[:n, 2]
        low_e = (e_curr < 50)
        score -= (low_e * 800)
        e_overflow = (e_curr > e_max * 0.95)
        score -= (e_overflow * 800)
        total_bp = cp.dot(inv_f, G_STATS[:, IDX_BP_ADD])
        spending_cap = total_bp * 0.025
        income_threshold = (m_yield * 1.5) + 1e-5
        excess_ratio = spending_cap / income_threshold
        is_over_capacity = (excess_ratio > 1.0)
        bp_penalty = (excess_ratio - 1.0) * 2000
        score -= (bp_penalty * is_over_capacity)
        k = BEAM_WIDTH
        top_idx = cp.argpartition(score, -k)[-k:]
        self.res_buf[:k] = self.res_buf[top_idx]
        self.inv_buf[:k] = self.inv_buf[top_idx]
        self.sq_buf[:k] = self.sq_buf[top_idx]
        self.total_sims = k
        self.history_parents.append(top_idx.astype(cp.int32))
        self.history_actions.append(cp.full(k, -2, dtype=cp.int16))

    def run(self):
        print(f"STARTING GPU SIM: {BEAM_WIDTH} Beams, {MAX_CANDIDATES} VRAM Buffer")
        self._apply_initial_build_queue()
        pbar = tqdm(range(int(self.res_buf[0, 3].item()), TIME_LIMIT))
        for t in pbar:
            self.physics_step()
            self.resolve_completions()
            self.branching_step()
            if t > 0 and (t % EVAL_INTERVAL == 0 or self.total_sims > MAX_CANDIDATES * 0.8):
                self.prune()
                pbar.set_description(f"Sims: {self.total_sims}")
        self.prune()
        return self.get_result()

    def get_result(self):
        n = self.total_sims
        inv_f = self.inv_buf[:n].astype(cp.float32)
        m_yield = cp.dot(inv_f, G_STATS[:, IDX_M_YIELD])
        score = (m_yield * 100) + cp.dot(inv_f, G_STATS[:, IDX_M_COST])
        best_idx = int(cp.argmax(score))
        h_parents = [h.get() for h in self.history_parents]
        h_actions = [h.get() for h in self.history_actions]
        bo_log = []
        curr = best_idx
        for i in range(len(h_parents)-1, -1, -1):
            if curr >= len(h_parents[i]): break
            p = h_parents[i][curr]
            a = h_actions[i][curr]
            if a >= 0:
                sq = (a >> 8)
                u = (a & 0xFF)
                bo_log.append(f"{SQUAD_NAMES[sq]} -> {UNIT_NAMES[u]}")
            curr = p
        return list(reversed(bo_log)), float(score[best_idx])

if __name__ == "__main__":
    sim = BarSimOptimized()
    st = time.time()
    bo, score = sim.run()
    et = time.time()
    print(f"\nCompleted in {et-st:.2f}s")
    print(f"Best Score: {score}")
    print("Build Order:")
    for i, line in enumerate(bo):
        print(f"{i+1}. {line}")
    input("Press Enter to exit...")
