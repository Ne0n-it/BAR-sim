import numpy as np
import cupy as cp
import time
from tqdm import tqdm

# ==========================================
# 1. CONFIGURAZIONE E DATABASE
# ==========================================

# Parametri Simulazione
BEAM_WIDTH = 120000         # Numero di simulazioni sopravvissute
MAX_CANDIDATES = 3000000    # Buffer VRAM (circa 500MB totali)
TIME_LIMIT = 8 * 60        # 8 Minuti
EVAL_INTERVAL = 900         # Pruning ogni 30s
WIND_LEVEL = 0            # Vento medio
MEX_METAL_YIELD = 2.0

# --- DATABASE ARMADA COMPLETO ---
ADB = {
    'Commander': {'bp': 0, 'm': 0, 'e': 0, 'BP': 300, 'M': 2, 'E': 25, 'build_options': ['Mex', 'Solar', 'Wind', 'Vehicle_Plant']},
    'Solar': {'m': 155, 'e': 0, 'bp': 2600, 'E': 20, 'M': 0, 'BP': 0, 'e_drain': 0},
    'Adv_Solar': {'m': 350, 'e': 5000, 'bp': 7950, 'E': 75, 'M': 0, 'BP': 0, 'e_drain': 0},
    'Wind': {'m': 40, 'e': 175, 'bp': 1600, 'E': WIND_LEVEL, 'M': 0, 'BP': 0, 'e_drain': 0},
    'FuR': {'m': 4300, 'e': 21000, 'bp': 70000, 'E': 1000, 'M': 0, 'BP': 0, 'e_drain': 0},
    'AFur': {'m': 9700, 'e': 69000, 'bp': 3125000, 'E': 3000, 'M': 0, 'BP': 0, 'e_drain': 0},
    'Mex': {'m': 50, 'e': 500, 'bp': 1800, 'E': 0, 'M': MEX_METAL_YIELD, 'BP': 0, 'e_drain': 3},
    'Adv_Mex': {'m': 620, 'e': 7700, 'bp': 14900, 'E': 0, 'M': MEX_METAL_YIELD * 4, 'BP': 0, 'e_drain': 20},
    'EStorage': {'m': 260, 'e': 1700, 'bp': 4110, 'E': 0, 'M': 0, 'BP': 0, 'storage': 6000, 'e_drain': 0},
    'DefMex': {'m': 190, 'e': 0, 'bp': 2400, 'E': 0, 'M': 0, 'BP': 0, 'e_drain': 0},
    'GEO': {'m': 560, 'e': 13000, 'bp': 13100, 'E': 300, 'M': 0, 'BP': 0, 'e_drain': 0},
    'Adv_GEO': {'m': 1600, 'e': 27000, 'bp': 33300, 'E': 1250, 'M': 0, 'BP': 0, 'e_drain': 0},
    'EConv': {'m': 1, 'e': 1150, 'bp': 2600, 'E': 0, 'M': 1, 'BP': 0, 'e_drain': 70},
    'Adv_EConv': {'m': 380, 'e': 21000, 'bp': 35000, 'E': 0, 'M': 10.3, 'BP': 0, 'e_drain': 600},

    # Factories
    'Vehicle_Plant': {'m': 590, 'e': 1550, 'bp': 4280, 'BP': 150, 'M': 0, 'E': 0, 'e_drain': 0, 'unit_options': ['ConVeh', 'Offensive']},
    'Advanced_Vehicle_Plant_T2': {'m': 2900, 'e': 14000, 'bp': 33800, 'BP': 300, 'M': 0, 'E': 0, 'e_drain': 0, 'unit_options': ['Adv_ConVeh', 'Adv_Offensive']},
    'Experimental_Gantry_T3': {'m': 7900, 'e': 58000, 'bp': 131800, 'BP': 600, 'M': 0, 'E': 0, 'e_drain': 0, 'unit_options': ['Titan']},

    # Builders
    'ConVeh': {'m': 135, 'e': 1950, 'bp': 4100, 'BP': 90, 'e_drain': 0, 'build_options': ['Mex', 'Solar', 'Wind', 'EStorage', 'Adv_Solar', 'GEO', 'EConv', 'Advanced_Vehicle_Plant_T2']},
    'Adv_ConVeh': {'m': 550, 'e': 6800, 'bp': 12400, 'BP': 250, 'e_drain': 0, 'build_options': ['Adv_Mex', 'Adv_Solar', 'FuR', 'AFur', 'Adv_GEO', 'Adv_EConv', 'Experimental_Gantry_T3']},
    'ConTurret': {'m': 210, 'e': 3200, 'bp': 5300, 'BP': 200, 'e_drain': 0, 'build_options': []},

    # Offensive
    'Offensive': {'m': 200, 'e': 2250, 'bp': 3500, 'BP': 0, 'e_drain': 0},
    'Adv_Offensive': {'m': 950, 'e': 13000, 'bp': 17200, 'BP': 0, 'e_drain': 0},
    'Titan' : {'m': 13500, 'e': 286000, 'bp': 276000, 'BP': 0, 'e_drain': 0}
}

# Mapping ID
UNIT_NAMES = list(ADB.keys())
UNIT_MAP = {name: i for i, name in enumerate(UNIT_NAMES)}
N_UNITS = len(UNIT_NAMES)

SQUAD_NAMES = ['Commander', 'FactoryT1', 'ConSquad1', 'ConSquad2', 'ConSquad3', 'FactoryT2', 'Adv_ConSquad', 'Gantry']
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
STATS_COLS = 8

HOST_STATS = np.zeros((N_UNITS, STATS_COLS), dtype=np.float32)

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
        
        # Buffer Allocations
        self.res_buf = cp.zeros((MAX_CANDIDATES, 4), dtype=cp.float32)
        
        # FIX: Changed to int32 for atomic add support in CuPy
        self.inv_buf = cp.zeros((MAX_CANDIDATES, N_UNITS), dtype=cp.int32)
        
        self.sq_buf = cp.zeros((MAX_CANDIDATES, N_SQUADS, 6), dtype=cp.float32)
        
        # Init Commander
        self.res_buf[0] = cp.array([1000, 1000, 1000, 0])
        self.inv_buf[0, UNIT_MAP['Commander']] = 1
        
        self.sq_buf[0, SQ_MAP['Commander'], 0] = ST_IDLE
        self.sq_buf[0, SQ_MAP['Commander'], 1] = ADB['Commander']['BP']
        self.sq_buf[0, SQ_MAP['Commander'], 2] = -1
        
        # Traceback Arrays
        self.history_parents = []
        self.history_actions = []

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
            m_curr, e_curr, max_e,
            m_inc, e_inc,
            drain_m_build, drain_e_build, drain_e_passive
        )
        
        self.res_buf[:n, 0] = m_new
        self.res_buf[:n, 1] = e_new
        self.res_buf[:n, 3] += 1
        
        eff_expanded = eff_factor[:, None]
        self.sq_buf[:n, :, 3] += (squad_bp * is_building * eff_expanded)
        
        is_travel = (sq_view[:, :, 0] == ST_TRAVEL)
        self.sq_buf[:n, :, 4] -= 1 * is_travel

    def resolve_completions(self):
        n = self.total_sims
        sq_view = self.sq_buf[:n]
        
        task_idx = sq_view[:, :, 2].astype(cp.int32)
        safe_tasks = cp.maximum(task_idx, 0)
        bp_costs = G_STATS[safe_tasks, IDX_BP_COST]
        
        is_building = (sq_view[:, :, 0] == ST_BUILDING)
        finished = is_building & (sq_view[:, :, 3] >= bp_costs)
        
        if not cp.any(finished):
            return
            
        sim_idxs, sq_idxs = cp.where(finished)
        unit_ids = task_idx[sim_idxs, sq_idxs]
        
        # Now safe because inv_buf is int32
        cp.add.at(self.inv_buf, (sim_idxs, unit_ids), 1)
        
        added_storage = G_STATS[unit_ids, IDX_STORAGE]
        cp.add.at(self.res_buf[:, 2], sim_idxs, added_storage)
        
        # Logic Builder Update
        is_conveh = (unit_ids == UNIT_MAP.get('ConVeh', -1))
        if cp.any(is_conveh):
            s_idx = sim_idxs[is_conveh]
            t_sq = SQ_MAP['ConSquad1']
            self.sq_buf[s_idx, t_sq, 0] = cp.maximum(self.sq_buf[s_idx, t_sq, 0], ST_IDLE)
            cp.add.at(self.sq_buf[:, t_sq, 1], s_idx, 90)
            
        is_vp = (unit_ids == UNIT_MAP.get('Vehicle_Plant', -1))
        if cp.any(is_vp):
            s_idx = sim_idxs[is_vp]
            t_sq = SQ_MAP['FactoryT1']
            self.sq_buf[s_idx, t_sq, 0] = ST_IDLE
            self.sq_buf[s_idx, t_sq, 1] = 150
            
        self.sq_buf[sim_idxs, sq_idxs, 0] = ST_IDLE
        self.sq_buf[sim_idxs, sq_idxs, 2] = -1
        self.sq_buf[sim_idxs, sq_idxs, 3] = 0

    def branching_step(self):
        n = self.total_sims
        
        # Reset travelers
        travel_done = (self.sq_buf[:n, :, 0] == ST_TRAVEL) & (self.sq_buf[:n, :, 4] <= 0)
        self.sq_buf[:n][travel_done] = ST_IDLE
        
        # Find IDLE
        is_idle = (self.sq_buf[:n, :, 0] == ST_IDLE)
        has_idle = cp.any(is_idle, axis=1)
        
        idle_sim_indices = cp.where(has_idle)[0]
        if len(idle_sim_indices) == 0:
             self.history_parents.append(cp.arange(n, dtype=cp.int32))
             self.history_actions.append(cp.full(n, -1, dtype=cp.int16))
             return

        first_idle_sq = cp.argmax(is_idle[idle_sim_indices], axis=1)
        
        opts_matrix = G_BUILD_OPTS[first_idle_sq]
        valid_mask = (opts_matrix != -1)
        
        # --- FIX: USE GPU ONLY (NO REPEAT) ---
        counts_gpu = cp.sum(valid_mask, axis=1)
        total_new = int(cp.sum(counts_gpu))
        
        if total_new + n > MAX_CANDIDATES:
            self.history_parents.append(cp.arange(n, dtype=cp.int32))
            self.history_actions.append(cp.full(n, -1, dtype=cp.int16))
            return
            
        # --- GPU INVERSE RLE (Vectorized Expansion) ---
        ends = cp.cumsum(counts_gpu)
        dest_indices = cp.arange(total_new, dtype=cp.int32)
        map_idxs = cp.searchsorted(ends, dest_indices, side='right')
        
        parent_idxs = idle_sim_indices[map_idxs]
        squads = first_idle_sq[map_idxs]
        actions = opts_matrix[valid_mask]
        
        # Write Block
        start = n
        end = n + total_new
        new_range = cp.arange(start, end)
        
        self.res_buf[start:end] = self.res_buf[parent_idxs]
        self.inv_buf[start:end] = self.inv_buf[parent_idxs]
        self.sq_buf[start:end] = self.sq_buf[parent_idxs]
        
        # Set Action Parameters
        mex_idx = UNIT_MAP['Mex']
        geo_idx = UNIT_MAP.get('GEO', -1)
        
        waits = cp.zeros_like(actions)
        waits[actions == mex_idx] = 5
        waits[actions == geo_idx] = 15
        
        # Default: Building
        self.sq_buf[new_range, squads, 0] = ST_BUILDING
        self.sq_buf[new_range, squads, 2] = actions.astype(cp.float32)
        self.sq_buf[new_range, squads, 3] = 0
        
        # Override: Travel
        is_travel_act = (waits > 0)
        idx_tr = new_range[is_travel_act]
        sq_tr = squads[is_travel_act]
        
        self.sq_buf[idx_tr, sq_tr, 0] = ST_TRAVEL
        self.sq_buf[idx_tr, sq_tr, 4] = waits[is_travel_act].astype(cp.float32)

        # History Update
        self.total_sims = end
        
        full_parents = cp.concatenate((cp.arange(n, dtype=cp.int32), parent_idxs))
        
        act_exist = cp.full(n, -1, dtype=cp.int16)
        act_new = ((squads.astype(cp.int16) << 8) | actions.astype(cp.int16))
        full_actions = cp.concatenate((act_exist, act_new))
        
        self.history_parents.append(full_parents)
        self.history_actions.append(full_actions)

    def prune(self):
        n = self.total_sims
        if n <= BEAM_WIDTH: return
        
        inv_f = self.inv_buf[:n].astype(cp.float32)
        m_yield = cp.dot(inv_f, G_STATS[:, IDX_M_YIELD])
        m_spent = cp.dot(inv_f, G_STATS[:, IDX_M_COST])
        
        score = (m_yield * 100) + m_spent
        
        low_e = (self.res_buf[:n, 1] < 50)
        score -= (low_e * 5000)
        
        overflow = (self.res_buf[:n, 1] > self.res_buf[:n, 2] * 0.9)
        score -= (overflow * 500)
        
        k = BEAM_WIDTH
        # Top K selection
        top_idx = cp.argpartition(score, -k)[-k:]
        
        self.res_buf[:k] = self.res_buf[top_idx]
        self.inv_buf[:k] = self.inv_buf[top_idx]
        self.sq_buf[:k] = self.sq_buf[top_idx]
        self.total_sims = k
        
        self.history_parents.append(top_idx.astype(cp.int32))
        self.history_actions.append(cp.full(k, -2, dtype=cp.int16))

    def run(self):
        print(f"STARTING GPU SIM: {BEAM_WIDTH} Beams, {MAX_CANDIDATES} VRAM Buffer")
        pbar = tqdm(range(TIME_LIMIT))
        
        for t in pbar:
            self.physics_step()
            self.resolve_completions()
            self.branching_step()
            
            if t % EVAL_INTERVAL == 0 or self.total_sims > MAX_CANDIDATES * 0.8:
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
        
        # Traceback Logic
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
