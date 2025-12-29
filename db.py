import numpy as np
import cupy as cp
import copy

# ==========================================
# 1. CONFIGURAZIONE E DATABASE (CPU)
# ==========================================

# Faction Selection
FACTION = 'armada' # or 'cortex'

# Parametri Ambientali
WIND_LEVEL = 0
MEX_METAL_YIELD = 2.0
START_M = 1000
START_E = 1000
TIME_LIMIT = 8 * 60
EVAL_INTERVAL = 1
BEAM_WIDTH = 16384
FACTORY_TYPE = 'Vehicle_Plant' # Scelta iniziale
if FACTORY_TYPE == 'Vehicle_Plant':
    ADV_FACTORY_TYPE = 'Advanced_Vehicle_Plant_T2'
# (Espandi se vuoi supportare Air/Bot, per ora lo script assume VehLab come da prompt)

# Mappa Mex (Distanze in secondi dal precedente)
MEX_DISTANCES = [0, 5, 5, 10, 10, 15, 15, 20, 20, 25, 25, 30]
TOTAL_MEX_ON_MAP = len(MEX_DISTANCES)
MEX_FORT_THRESHOLD = 2
MEX_FORT_COST = {'m': 170, 'e': 1360, 'bp': 4800}

# Armada Database
ADB = {
    'Commander': {
        'bp': 0, 'm': 0, 'e': 0, 'BP': 300, 'M': 2, 'E': 25,
        'build_options': ['Mex', 'Solar', 'Wind', 'EStorage', 'Vehicle_Plant', 'Aircraft_Plant', 'Shipyard']
    },
    'Solar': {'m': 155, 'e': 0, 'bp': 2600, 'E': 20, 'M': 0, 'BP': 0, 'e_drain': 0},
    'Adv_Solar': {'m': 350, 'e': 5000, 'bp': 7950, 'E': 75, 'M': 0, 'BP': 0, 'e_drain': 0},
    'Wind': {'m': 40, 'e': 175, 'bp': 1600, 'E': WIND_LEVEL, 'M': 0, 'BP': 0, 'e_drain': 0},
    'Reactor': {'m': 4300, 'e': 21000, 'bp': 70000, 'E': 1000, 'M': 0, 'BP': 0, 'e_drain': 0},
    'Mex': {'m': 50, 'e': 500, 'bp': 1800, 'E': 0, 'M': MEX_METAL_YIELD, 'BP': 0, 'e_drain': 3},
    'Adv_Mex': {'m': 620, 'e': 7700, 'bp': 14900, 'E': 0, 'M': MEX_METAL_YIELD * 4, 'BP': 0, 'e_drain': 20},
    'EStorage': {'m': 260, 'e': 1700, 'bp': 4110, 'E': 0, 'M': 0, 'BP': 0, 'storage': 6000, 'e_drain': 0},
    'DefMex': {'m': 190, 'e': 0, 'bp': 2400, 'E': 0, 'M': 0, 'BP': 0, 'e_drain': 0},

    # Armada Factories
    'Shipyard': {'m': 450, 'e': 950, 'bp': 2800, 'BP': 150, 'M': 0, 'E': 0, 'e_drain': 0, 'unit_options': []},
    'Vehicle_Plant': {'m': 590, 'e': 1550, 'bp': 4280, 'BP': 150, 'M': 0, 'E': 0, 'e_drain': 0, 'unit_options': ['ConVeh', 'Offensive']},
    'Aircraft_Plant': {'m': 710, 'e': 1100, 'bp': 3620, 'BP': 150, 'M': 0, 'E': 0, 'e_drain': 0, 'unit_options': []},
    'Naval_Hovercraft_Platform': {'m': 750, 'e': 2750, 'bp': 7000, 'BP': 150, 'M': 0, 'E': 0, 'e_drain': 0, 'unit_options': []},
    'Amphibious_Complex': {'m': 1200, 'e': 5500, 'bp': 13400, 'BP': 150, 'M': 0, 'E': 0, 'e_drain': 0, 'unit_options': []},
    'Seaplane_Platform': {'m': 1450, 'e': 5000, 'bp': 12900, 'BP': 200, 'M': 0, 'E': 0, 'e_drain': 0, 'unit_options': []},
    'Advanced_Bot_Lab_T2': {'m': 2900, 'e': 15000, 'bp': 35800, 'BP': 300, 'M': 0, 'E': 0, 'e_drain': 0, 'unit_options': []},
    'Advanced_Vehicle_Plant_T2': {'m': 2900, 'e': 14000, 'bp': 33800, 'BP': 300, 'M': 0, 'E': 0, 'e_drain': 0, 'unit_options': ['Adv_ConVeh', 'Adv_Offensive']},
    'Advanced_Aircraft_Plant_T2': {'m': 3200, 'e': 29000, 'bp': 64400, 'BP': 200, 'M': 0, 'E': 0, 'e_drain': 0, 'unit_options': []},
    'Advanced_Shipyard_T2': {'m': 3200, 'e': 9700, 'bp': 25800, 'BP': 300, 'M': 0, 'E': 0, 'e_drain': 0, 'unit_options': []},
    'Experimental_Gantry_T3': {'m': 7900, 'e': 58000, 'bp': 131800, 'BP': 600, 'M': 0, 'E': 0, 'e_drain': 0, 'unit_options': []},

    # Builders
    'ConAir': {'m': 100, 'e': 2400, 'bp': 5000, 'BP': 50, 'e_drain': 0, 'build_options': ['Mex', 'Solar', 'Wind', 'EStorage', 'Adv_Solar', 'Advanced_Vehicle_Plant_T2']},
    'ConBot': {'m': 110, 'e': 2200, 'bp': 4500, 'BP': 80, 'e_drain': 0, 'build_options': ['Mex', 'Solar', 'Wind', 'EStorage', 'Adv_Solar', 'Advanced_Vehicle_Plant_T2']},
    'ConVeh': {'m': 135, 'e': 1950, 'bp': 4100, 'BP': 90, 'e_drain': 0, 'build_options': ['Mex', 'Solar', 'Wind', 'EStorage', 'Adv_Solar', 'Advanced_Vehicle_Plant_T2']},
    'Adv_ConVeh': {'m': 550, 'e': 6800, 'bp': 12400, 'BP': 250, 'e_drain': 0, 'build_options': ['Adv_Mex', 'Reactor', 'Adv_Solar']},

    # Offensive
    'Offensive': {'m': 200, 'e': 2250, 'bp': 3500, 'BP': 0, 'e_drain': 0},
    'Adv_Offensive': {'m': 950, 'e': 13000, 'bp': 17200, 'BP': 0, 'e_drain': 0}
}

# Cortex Database (Placeholder)
CDB = copy.deepcopy(ADB)

# Select DB based on Faction
if FACTION == 'armada':
    DB = ADB
elif FACTION == 'cortex':
    DB = CDB

# Mapping ID numerici per GPU
UNIT_NAMES = list(DB.keys())
UNIT_MAP = {name: i for i, name in enumerate(UNIT_NAMES)}
REV_UNIT_MAP = {i: name for name, i in UNIT_MAP.items()}
N_UNITS = len(UNIT_NAMES)

# Mapping Squadre (Slot fissi nel tensore 3D)
SQUAD_NAMES = ['Commander', 'Vehicle_Plant', 'ConVeh_1', 'ConVeh_2', 'ConVeh_3', 'Advanced_Vehicle_Plant_T2', 'Adv_ConVeh']
N_SQUADS = len(SQUAD_NAMES)

# Costruzione vettori statici per GPU (Lookups)
LOOKUP_M_COST = np.zeros(N_UNITS, dtype=np.float32)
LOOKUP_E_COST = np.zeros(N_UNITS, dtype=np.float32)
LOOKUP_BP_COST = np.zeros(N_UNITS, dtype=np.float32)
LOOKUP_M_YIELD = np.zeros(N_UNITS, dtype=np.float32)
LOOKUP_E_YIELD = np.zeros(N_UNITS, dtype=np.float32)
LOOKUP_E_DRAIN = np.zeros(N_UNITS, dtype=np.float32)

for name, data in DB.items():
    idx = UNIT_MAP[name]
    LOOKUP_M_COST[idx] = data['m']
    LOOKUP_E_COST[idx] = data['e']
    LOOKUP_BP_COST[idx] = data['bp']
    LOOKUP_M_YIELD[idx] = data.get('M', 0)
    LOOKUP_E_YIELD[idx] = data.get('E', 0)
    LOOKUP_E_DRAIN[idx] = data.get('e_drain', 0)

# Spostiamo i lookup su GPU
G_M_COST = cp.asarray(LOOKUP_M_COST)
G_E_COST = cp.asarray(LOOKUP_E_COST)
G_BP_COST = cp.asarray(LOOKUP_BP_COST)
G_M_YIELD = cp.asarray(LOOKUP_M_YIELD)
G_E_YIELD = cp.asarray(LOOKUP_E_YIELD)
G_E_DRAIN = cp.asarray(LOOKUP_E_DRAIN)

# Status Codes
ST_INACTIVE = 0
ST_IDLE = 1
ST_TRAVELING = 2
ST_BUILDING = 3
