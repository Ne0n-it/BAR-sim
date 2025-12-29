import numpy as np
import cupy as cp

# ==========================================
# 1. CONFIGURAZIONE E DATABASE (CPU)
# ==========================================

# Parametri Ambientali
WIND_LEVEL = 0
MEX_METAL_YIELD = 2.0  # Base yield per mex standard
START_M = 1000
START_E = 1000
TIME_LIMIT = 8*60  # 5 minuti in secondi
EVAL_INTERVAL = 1  # Valuta e ramifica ogni X secondi
BEAM_WIDTH = 16384   # Numero di universi paralleli da mantenere (ridurre se GPU OOM)
FACTORY_TYPE = 'VehLab' # Scelta iniziale
if FACTORY_TYPE == 'VehLab':
    ADV_FACTORY_TYPE = 'Adv_VehLab'
elif FACTORY_TYPE == 'BotLab':
    ADV_FACTORY_TYPE = 'Adv_BotLab'
# (Espandi se vuoi supportare Air/Bot, per ora lo script assume VehLab come da prompt)

# Mappa Mex (Distanze in secondi dal precedente)
# [0] = Start pos, [1] = 5s dal primo, etc.
MEX_DISTANCES = [0, 5, 5, 10, 10, 15, 15, 20, 20, 25, 25, 30]
TOTAL_MEX_ON_MAP = len(MEX_DISTANCES)
MEX_FORT_THRESHOLD = 2 # I primi 2 sono gratis, dal 3 (indice 2) costano extra
MEX_FORT_COST = {'m': 170, 'e': 1360, 'bp': 4800}

# Database Unità
DB = {
    'Commander': {
        'bp': 0, 'm': 0, 'e': 0, 'BP': 300, 'M': 2, 'E': 25,
        'build_options': ['Mex', 'Solar', 'Wind', 'EStorage', 'BotLab', 'VehLab', 'AirLab']
    },
    'Solar':    {'m': 155, 'e': 0, 'bp': 2600, 'E': 20, 'M': 0, 'BP': 0, 'e_drain': 0},
    'Adv_Solar':{'m': 350, 'e': 5000, 'bp': 7950, 'E': 75, 'M': 0, 'BP': 0, 'e_drain': 0},
    'Wind':     {'m': 40, 'e': 175, 'bp': 1600, 'E': WIND_LEVEL, 'M': 0, 'BP': 0, 'e_drain': 0},
    'Reactor':  {'m': 4300, 'e': 21000, 'bp': 70000, 'E': 1000, 'M': 0, 'BP': 0, 'e_drain': 0},
    'Mex':      {'m': 50, 'e': 500, 'bp': 1800, 'E': 0, 'M': MEX_METAL_YIELD, 'BP': 0, 'e_drain': 3},
    'Adv_Mex':  {'m': 620, 'e': 7700, 'bp': 14900, 'E': 0, 'M': MEX_METAL_YIELD*4, 'BP': 0, 'e_drain': 20},
    'EStorage': {'m': 260, 'e': 1700, 'bp': 4110, 'E': 0, 'M': 0, 'BP': 0, 'storage': 6000, 'e_drain': 0},
    'DefMex':   {'m': 190, 'e': 0, 'bp': 2400, 'E': 0, 'M': 0, 'BP': 0, 'e_drain': 0}, # Dummy entry per costi

    # Factories
    'AirLab':    {'m': 600, 'e': 1200, 'bp': 7000, 'BP': 100, 'e_drain': 0, 'unit_options': ['ConAir', 'Offensive']},
    'BotLab':    {'m': 650, 'e': 1200, 'bp': 6500, 'BP': 100, 'e_drain': 0, 'unit_options': ['ConBot', 'Offensive']},
    'VehLab':    {'m': 590, 'e': 1550, 'bp': 5700, 'BP': 100, 'e_drain': 0, 'unit_options': ['ConVeh', 'Offensive']},
    'Adv_VehLab':{'m': 2900, 'e': 14000, 'bp': 18000, 'BP': 300, 'e_drain': 0, 'unit_options': ['Adv_ConVeh', 'Adv_Offensive']},

    # Builders
    'ConAir':    {'m': 100, 'e': 2400, 'bp': 5000, 'BP': 50, 'e_drain': 0, 'build_options': ['Mex', 'Solar', 'Wind', 'EStorage', 'Adv_Solar', 'Adv_VehLab']},
    'ConBot':    {'m': 110, 'e': 2200, 'bp': 4500, 'BP': 80, 'e_drain': 0, 'build_options': ['Mex', 'Solar', 'Wind', 'EStorage', 'Adv_Solar', 'Adv_VehLab']},
    'ConVeh':    {'m': 135, 'e': 1950, 'bp': 4100, 'BP': 90, 'e_drain': 0, 'build_options': ['Mex', 'Solar', 'Wind', 'EStorage', 'Adv_Solar', 'Adv_VehLab']},
    'Adv_ConVeh':{'m': 550, 'e': 6800, 'bp': 12400, 'BP': 250, 'e_drain': 0, 'build_options': ['Adv_Mex', 'Reactor', 'Adv_Solar']},

    # Offensive
    'Offensive': {'m': 200, 'e': 2250, 'bp': 3500, 'BP': 0, 'e_drain': 0},
    'Adv_Offensive': {'m': 950, 'e': 13000, 'bp': 17200, 'BP': 0, 'e_drain': 0}
}

# Mapping ID numerici per GPU
UNIT_NAMES = list(DB.keys())
UNIT_MAP = {name: i for i, name in enumerate(UNIT_NAMES)}
REV_UNIT_MAP = {i: name for name, i in UNIT_MAP.items()}
N_UNITS = len(UNIT_NAMES)

# Mapping Squadre (Slot fissi nel tensore 3D)
SQUAD_NAMES = ['Commander', 'VehLab', 'ConVeh_1', 'ConVeh_2', 'ConVeh_3', 'Adv_VehLab', 'Adv_ConVeh']
N_SQUADS = len(SQUAD_NAMES)

# Costruzione vettori statici per GPU (Lookups)
# Costi M, E, BP, e Yields per ogni unità (indicizzata da UNIT_MAP)
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
