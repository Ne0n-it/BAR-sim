# OTTIMO
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

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
        'is_structure': False, 
        'build_options': ['Mex', 'Solar', 'Wind', 'EStorage', 'BotLab', 'VehLab', 'AirLab']
    },
    'Solar':    {'m': 155, 'e': 0, 'bp': 2600, 'E': 20, 'M': 0, 'BP': 0, 'is_structure': True, 'e_drain': 0},
    'Adv_Solar':{'m': 350, 'e': 5000, 'bp': 7950, 'E': 75, 'M': 0, 'BP': 0, 'is_structure': True, 'e_drain': 0},
    'Wind':     {'m': 40, 'e': 175, 'bp': 1600, 'E': WIND_LEVEL, 'M': 0, 'BP': 0, 'is_structure': True, 'e_drain': 0},
    'Reactor':  {'m': 4300, 'e': 21000, 'bp': 70000, 'E': 1000, 'M': 0, 'BP': 0, 'is_structure': True, 'e_drain': 0},
    'Mex':      {'m': 50, 'e': 500, 'bp': 1800, 'E': 0, 'M': MEX_METAL_YIELD, 'BP': 0, 'is_structure': True, 'e_drain': 3},
    'Adv_Mex':  {'m': 620, 'e': 7700, 'bp': 14900, 'E': 0, 'M': MEX_METAL_YIELD*4, 'BP': 0, 'is_structure': True, 'e_drain': 20},
    'EStorage': {'m': 260, 'e': 1700, 'bp': 4110, 'E': 0, 'M': 0, 'BP': 0, 'storage': 6000, 'is_structure': True, 'e_drain': 0},
    'DefMex':   {'m': 190, 'e': 0, 'bp': 2400, 'E': 0, 'M': 0, 'BP': 0, 'is_structure': True, 'e_drain': 0}, # Dummy entry per costi

    # Factories
    'AirLab':    {'m': 600, 'e': 1200, 'bp': 7000, 'BP': 100, 'is_structure': True, 'e_drain': 0, 'builder_name': 'ConAir', 'unit_options': ['ConAir', 'Offensive']},
    'BotLab':    {'m': 650, 'e': 1200, 'bp': 6500, 'BP': 100, 'is_structure': True, 'e_drain': 0, 'builder_name': 'ConBot', 'unit_options': ['ConBot', 'Offensive']},
    'VehLab':    {'m': 590, 'e': 1550, 'bp': 5700, 'BP': 100, 'is_structure': True, 'e_drain': 0, 'builder_name': 'ConVeh', 'unit_options': ['ConVeh', 'Offensive']},
    'Adv_VehLab':{'m': 2900, 'e': 14000, 'bp': 18000, 'BP': 300, 'is_structure': True, 'e_drain': 0, 'builder_name': 'Adv_ConVeh', 'unit_options': ['Adv_ConVeh', 'Adv_Offensive']},

    # Builders
    'ConAir':    {'m': 100, 'e': 2400, 'bp': 5000, 'BP': 50, 'is_structure': False, 'e_drain': 0, 'build_options': ['Mex', 'Solar', 'Wind', 'EStorage', 'Adv_Solar', 'Adv_VehLab']},
    'ConBot':    {'m': 110, 'e': 2200, 'bp': 4500, 'BP': 80, 'is_structure': False, 'e_drain': 0, 'build_options': ['Mex', 'Solar', 'Wind', 'EStorage', 'Adv_Solar', 'Adv_VehLab']},
    'ConVeh':    {'m': 135, 'e': 1950, 'bp': 4100, 'BP': 90, 'is_structure': False, 'e_drain': 0, 'build_options': ['Mex', 'Solar', 'Wind', 'EStorage', 'Adv_Solar', 'Adv_VehLab']},
    'Adv_ConVeh':{'m': 550, 'e': 6800, 'bp': 12400, 'BP': 250, 'is_structure': False, 'e_drain': 0, 'build_options': ['Adv_Mex', 'Reactor', 'Adv_Solar']},

    # Offensive
    'Offensive': {'m': 200, 'e': 2250, 'bp': 3500, 'BP': 0, 'is_structure': False, 'e_drain': 0},
    'Adv_Offensive': {'m': 950, 'e': 13000, 'bp': 17200, 'BP': 0, 'is_structure': False, 'e_drain': 0}
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

# ==========================================
# 2. MOTORE DI SIMULAZIONE (Class)
# ==========================================

class BarOptimizer:
    def __init__(self):
        # Stato Globale: [Metal, Energy, MaxE, Time, ...Inventory(N_UNITS), Mex_Count_Tracker]
        # Inventory starts at index 4.
        self.sim_states = None 
        
        # Squadre: [BP, Status, TaskUnitID, Progress, WaitTime, MissionID, LocID]
        self.squads = None 
        
        self.histories = [] # Lista di liste per tracciare gli eventi (CPU side)
        self.n_sims = 0

    def initialize(self, initial_build_queue=None):
        """Inizializza con 1 simulazione base"""
        # Dimensione Stato Globale: 4 (res) + N_UNITS + 1 (MexTracker)
        global_cols = 4 + N_UNITS + 1
        
        self.sim_states = cp.zeros((1, global_cols), dtype=cp.float32)
        self.sim_states[0, 0] = START_M
        self.sim_states[0, 1] = START_E
        self.sim_states[0, 2] = 1000 # Max E iniziale (Commander)
        self.sim_states[0, 3] = 0    # Time
        self.sim_states[0, 4 + UNIT_MAP['Commander']] = 1 # Inventory Commander
        
        # Inizializza Squadre
        # 7 Props: BP, Status, TaskID, Progress, WaitTime, MissionID, LocID
        self.squads = cp.zeros((1, N_SQUADS, 7), dtype=cp.float32)
        
        # Setup Commander (Squad 0)
        self.squads[0, 0, 0] = DB['Commander']['BP'] # BP
        self.squads[0, 0, 1] = ST_IDLE               # Status
        self.squads[0, 0, 2] = -1                    # Task (-1 none)
        self.squads[0, 0, 5] = -1                    # Mission (-1 none)
        
        # Setup predefinito build queue (Opzionale)
        self.histories = [[]] # Una history vuota per la sim 0
        self.n_sims = 1

    def run_gpu_physics(self, duration_sec):
        """
        Esegue la simulazione fisica sulla GPU per N secondi.
        Calcola drain, income, stalli e progressi.
        """
        dt = 1.0 # step di 1 secondo
        steps = int(duration_sec)
        
        # Alias per leggibilità (viste, non copie)
        metal = self.sim_states[:, 0]
        energy = self.sim_states[:, 1]
        max_e = self.sim_states[:, 2]
        time_sim = self.sim_states[:, 3]
        inventory = self.sim_states[:, 4:4+N_UNITS]
        
        # Squad aliases
        sq_bp = self.squads[:, :, 0]
        sq_status = self.squads[:, :, 1]
        sq_task = self.squads[:, :, 2].astype(cp.int32)
        sq_progress = self.squads[:, :, 3]
        sq_wait = self.squads[:, :, 4]
        
        for _ in range(steps):
            # 1. Calcolo Income
            # Income statico da unità costruite (Solar, Wind, Mex)
            # Mex Yield dipende da mappa base (semplificato qui: yield costante * numero mex)
            # Nota: Per simulazione avanzata Mex income dipende da E-Stall, lo calcoliamo dopo
            
            base_m_income = cp.sum(inventory * G_M_YIELD, axis=1) + DB['Commander']['M']
            base_e_income = cp.sum(inventory * G_E_YIELD, axis=1) + DB['Commander']['E']
            
            # 2. Calcolo Demand (Drain)
            # Solo squadre in ST_BUILDING (3) consumano
            is_building = (sq_status == ST_BUILDING)
            
            # Costi del task corrente
            # Usiamo take_along_axis per estrarre i costi basati su sq_task
            task_m_cost = cp.take(G_M_COST, sq_task)
            task_e_cost = cp.take(G_E_COST, sq_task)
            task_bp_cost = cp.take(G_BP_COST, sq_task)
            
            # Evitiamo divisione per zero se task_bp_cost è 0 (non dovrebbe accadere in building)
            task_bp_cost = cp.maximum(task_bp_cost, 1.0)
            
            # Consumo per secondo = (TotalCost / (TotalCost / BP)) = BP * (Cost/BP_cost) ??
            # No. Formula standard Spring: Drain = BP * (Cost / TotalBuildTimeBP)
            # Semplificato: Drain/sec = BP_squad * (Cost_unit / Cost_BP_unit)
            
            drain_ratio_m = task_m_cost / task_bp_cost
            drain_ratio_e = task_e_cost / task_bp_cost
            
            current_drain_m = cp.sum(cp.where(is_building, sq_bp * drain_ratio_m, 0), axis=1)
            current_drain_e = cp.sum(cp.where(is_building, sq_bp * drain_ratio_e, 0), axis=1)
            
            # Aggiungi Drain passivo (Mex attivi, Radar, etc)
            passive_e_drain = cp.sum(inventory * G_E_DRAIN, axis=1)
            total_drain_e = current_drain_e + passive_e_drain
            
            # 3. Gestione Stalli (Logic Cascade)
            # E Stalling calculation
            # Se Energy < 0 dopo drain, calcoliamo efficienza
            
            net_e = energy + base_e_income - total_drain_e
            
            # Efficienza E: se net_e < 0, efficiency = available / demand
            # Available include riserva.
            # Se ho 100 in bank, income 10, demand 200. Posso pagare 110. Eff = 110/200.
            
            available_e = energy + base_e_income
            # Evitiamo div by zero
            safe_demand_e = cp.maximum(total_drain_e, 0.001)
            e_efficiency = cp.clip(available_e / safe_demand_e, 0.0, 1.0)
            
            # Se e_efficiency < 1, i Mex producono meno metallo
            # Formula semplificata: Mex output scales linearly (o quasi) with E
            actual_m_income = base_m_income * ((e_efficiency + 1.0) / 2.0) # Soft penalty
            # Nota: In BAR Mex si spengono a 0, qui facciamo penalty lineare smussata
            
            # M Stalling calculation
            available_m = metal + actual_m_income
            safe_demand_m = cp.maximum(current_drain_m, 0.001)
            m_efficiency = cp.clip(available_m / safe_demand_m, 0.0, 1.0)
            
            # Global Simulation Efficiency (min of M and E for construction)
            # Se manca E, costruttori rallentano. Se manca M, rallentano.
            global_efficiency = cp.minimum(e_efficiency, m_efficiency)
            
            # 4. Update Risorse
            # Spesa reale
            spent_m = current_drain_m * global_efficiency
            spent_e = total_drain_e * e_efficiency # E drain scales with E availability logic usually
            
            # Se siamo in overflow di E, non possiamo spendere più di quello che abbiamo
            # In realtà: resources = prev + income - spent
            # Se stallo, spent è limitato da available. Quindi resources vanno a 0.
            
            new_m = metal + actual_m_income - spent_m
            new_e = energy + base_e_income - spent_e # Qui c'è semplificazione, ma ok per sim
            
            # Clamp limits
            self.sim_states[:, 0] = cp.clip(new_m, 0, 100000) # No max M limit usually
            self.sim_states[:, 1] = cp.clip(new_e, 0, max_e)
            
            # 5. Avanzamento Costruzione e Tempo
            # Wait time reduction
            self.squads[:, :, 4] = cp.maximum(sq_wait - 1, 0)
            
            # Build Progress update
            # Solo se BUILDING e wait time è 0 (should be already)
            global_eff_reshaped = global_efficiency[:, None] 
            bp_applied = sq_bp * global_eff_reshaped
            active_builds = (sq_status == ST_BUILDING) & (sq_wait <= 0)
            
            self.squads[:, :, 3] = cp.where(active_builds, sq_progress + bp_applied, sq_progress)
            
            # Check Completion
            # Se Progress >= Costo BP Unit
            completed = active_builds & (self.squads[:, :, 3] >= task_bp_cost)
            
            # Se completato:
            # 1. Reset Squad a IDLE (fatto via mask dopo loop o qui)
            # 2. Incremento Inventory (se struttura)
            # Questa parte complessa la facciamo nel branching CPU per pulizia, 
            # MA per performance GPU dovremmo aggiornare l'inventory qui per l'income del prossimo tick.
            
            # Update Inventory in GPU (per income calculation next tick)
            # Otteniamo ID unità completate
            done_indices = cp.where(completed) # (sim_idx, squad_idx)
            if done_indices[0].size > 0:
                 # Otteniamo task ID per questi completamenti
                completed_tasks = sq_task[completed]
                
                # Aggiungiamo 1 all'inventario per ogni task completato
                # cp.add.at è necessario per race conditions (più builders finiscono stessa cosa stesso tick)
                # Offset 4 è inventory start
                # sim_idx = done_indices[0], col_idx = 4 + completed_tasks
                
                # Hacky: cp.add.at funziona su flat o slices.
                # Per ogni completamento: sim_states[s_idx, 4+u_id] += 1
                
                # Ma attenzione: Se è un'unità mobile (es. Costruttore), 
                # non va solo in inventory, va attivata una nuova squadra.
                # Questo richiede logica complessa.
                # PER SEMPLIFICARE:
                # Lasciamo che lo status rimanga BUILDING ma progress > 100%.
                # Income non si aggiorna subito.
                # Il ciclo CPU dopo il run_physics rileverà il completamento e gestirà la logica.
                pass

            self.sim_states[:, 3] += 1 # Time increment

    def cpu_branching_and_logic(self):
        # ... (Inizio funzione standard come prima) ...
        # Scarica dati CPU
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
                        # COMPLETATO
                        unit_name = REV_UNIT_MAP[task_id]
                        
                        # Aggiorna Inventario
                        parent_st[4 + task_id] += 1
                        
                        # === FIX: ATTIVAZIONE SQUADRE ===
                        if unit_name == FACTORY_TYPE:
                            # Attiva la squadra Factory T1 (Slot 1)
                            # Se non lo facciamo, resta INACTIVE per sempre
                            parent_sq[1][0] = DB[FACTORY_TYPE]['BP']
                            parent_sq[1][1] = ST_IDLE
                            
                        elif unit_name == 'Adv_VehLab':
                             # Attiva Factory T2 (Slot 5)
                             parent_sq[5][0] = DB['Adv_VehLab']['BP']
                             parent_sq[5][1] = ST_IDLE
                             
                        elif unit_name == 'Adv_ConVeh':
                             # Attiva Costruttore T2 (Slot 6)
                             parent_sq[6][0] = DB['Adv_ConVeh']['BP']
                             parent_sq[6][1] = ST_IDLE

                        elif unit_name in ['ConVeh', 'ConBot', 'ConAir']:
                             # Attiva Costruttore T1 (Slot 2, 3 o 4)
                             # Cerca il primo slot inattivo tra 2 e 4
                             for slot in range(2, 5): 
                                 if parent_sq[slot][1] == ST_INACTIVE:
                                     parent_sq[slot][0] = DB[unit_name]['BP']
                                     parent_sq[slot][1] = ST_IDLE 
                                     break
                                     
                        # Effetti Risorse speciali
                        if unit_name == 'Mex':
                            parent_st[4 + N_UNITS] += 1
                        elif unit_name == 'EStorage':
                            parent_st[2] += DB['EStorage']['storage']

                        # Reset Squadra costruttrice a IDLE
                        s_props[1] = ST_IDLE
                        s_props[2] = -1
                        s_props[3] = 0
                        s_props[4] = 0
                        
                        parent_hist.append(f"{int(current_time)}: Completed {unit_name} by {SQUAD_NAMES[s_idx]}")

            # ... (Il resto: "2. CASCATA DI BRANCHING" rimane identico alla mia risposta precedente) ...

            # 2. CASCATA DI BRANCHING (Cartesian Product)
            # Invece di scegliere solo una squadra, iteriamo su tutte le squadre.
            # Se la squadra 0 ramifica in A e B, e la squadra 1 ramifica in X e Y,
            # generiamo 4 universi: AX, AY, BX, BY.
            
            # Lista di universi parziali per questo tick. Iniziamo con l'universo corrente modificato.
            # Ogni elemento è una tupla (state, squads, history)
            step_universes = [(parent_st, parent_sq, parent_hist)]
            
            # Iteriamo su ogni squadra per vedere se deve prendere una decisione
            for s_idx in range(N_SQUADS):
                next_step_universes = []
                
                for p_st, p_sq, p_hist in step_universes:
                    
                    # Se la squadra non è IDLE, passa l'universo così com'è
                    if p_sq[s_idx][1] != ST_IDLE:
                        next_step_universes.append((p_st, p_sq, p_hist))
                        continue
                        
                    # --- GENERAZIONE OPZIONI PER SQUADRA IDLE ---
                    options = []
                    squad_name = SQUAD_NAMES[s_idx]
                    
                    # Identifica tipo costruttore
                    builder_type = None
                    if s_idx == 0: builder_type = 'Commander'
                    elif 1 <= s_idx <= 4: builder_type = 'ConVeh' if s_idx > 1 else 'VehLab' # Slot 1 is always T1 Factory
                    elif s_idx == 5: builder_type = 'Adv_VehLab'
                    elif s_idx == 6: builder_type = 'Adv_ConVeh'
                    
                    # === LOGICA FABBRICHE (Devono produrre sempre) ===
                    if 'Lab' in builder_type:
                        # Recupera opzioni dal DB
                        raw_opts = DB[builder_type].get('unit_options', [])
                        # Nessun filtro "limitante": la fabbrica DEVE produrre qualcosa
                        # Non mettiamo "continue" se non ci sono risorse, la fabbrica accoda (queue)
                        for opt in raw_opts:
                            # Filtro: Adv Factory costruisce solo se T1 Factory ha finito? No, indipendenti.
                            options.append(opt)
                            
                    # === LOGICA COSTRUTTORI ===
                    else:
                        raw_opts = DB[builder_type].get('build_options', [])
                        
                        for opt in raw_opts:
                            # 1. VINCOLO E-STORAGE UNICO
                            if opt == 'EStorage':
                                if p_st[4 + UNIT_MAP['EStorage']] > 0: continue
                                # Extra: Non costruire EStorage se max_e è già alto (evita spam inutile anche se 0)
                                if p_st[2] > 5000: continue 

                            # 2. VINCOLO FABBRICA UNICA (Specificata da Config)
                            if opt.endswith('Lab'):
                                # Se non è il tipo prescelto, scarta
                                if opt != FACTORY_TYPE and opt != ADV_FACTORY_TYPE: continue
                                # Se ne esiste già una (anche in costruzione conta come 1 nell'inventario logico 
                                # o controlliamo lo stato della squadra Factory associata)
                                # Controllo inventario (finito) + Controllo se la squadra Factory è già attiva/costruita
                                
                                # Se T1 Factory: controlla se squadra 1 è attiva o se inventario > 0
                                if opt == FACTORY_TYPE and (p_sq[1][0] > 0 or p_st[4+UNIT_MAP[opt]] > 0): continue
                                
                                # Se T2 Factory: controlla se squadra 5 è attiva
                                if opt == ADV_FACTORY_TYPE:
                                     # Richiede T1 factory esistente? Di solito si fa upgrade. 
                                     # Qui assumiamo sia una struttura separata per semplicità sim
                                     if p_sq[5][0] > 0: continue

                            # 3. VINCOLO MEX
                            if opt == 'Mex':
                                # Se finiti i mex sulla mappa
                                if int(p_st[4 + N_UNITS]) >= TOTAL_MEX_ON_MAP: continue
                                
                            # 4. VINCOLO WIND
                            if opt == 'Wind' and WIND_LEVEL < 6: continue
                            
                            # 5. VINCOLO ADV_MEX
                            if opt == 'Adv_Mex':
                                # Serve almeno un Mex base da convertire
                                if p_st[4+UNIT_MAP['Mex']] == 0: continue

                            options.append(opt)

                    # --- APPLICAZIONE OPZIONI (Branching) ---
                    if not options:
                        # Se non ha opzioni valide (improbabile per Commander, possibile per builder se mappa mex finita)
                        # La squadra rimane IDLE in questo ramo
                        next_step_universes.append((p_st, p_sq, p_hist))
                    else:
                        # Per ogni opzione, crea un nuovo ramo dall'universo parziale corrente
                        for opt in options:
                            # Deep Copy per il nuovo ramo
                            branch_st = p_st.copy()
                            branch_sq = p_sq.copy()
                            branch_hist = copy.copy(p_hist)
                            
                            t_idx = UNIT_MAP[opt]
                            wait_time = 2 # Default delay
                            
                            # Calcolo Travel Time per Mex
                            if opt == 'Mex':
                                curr_m = int(branch_st[4 + N_UNITS])
                                if curr_m < TOTAL_MEX_ON_MAP:
                                    wait_time += MEX_DISTANCES[curr_m]
                            
                            # Imposta la squadra al lavoro
                            branch_sq[s_idx][1] = ST_BUILDING
                            branch_sq[s_idx][2] = t_idx
                            branch_sq[s_idx][3] = 0
                            branch_sq[s_idx][4] = wait_time
                            
                            branch_hist.append(f"{int(current_time)}: {squad_name} starts {opt}")
                            
                            next_step_universes.append((branch_st, branch_sq, branch_hist))
                
                # Aggiorna la lista degli universi per il prossimo ciclo di squadra
                step_universes = next_step_universes
            
            # Alla fine del loop sulle squadre, step_universes contiene tutti i rami
            # generati dall'universo "parent" originale i.
            # Li aggiungiamo alla lista globale dei nuovi stati.
            for final_st, final_sq, final_hist in step_universes:
                new_states_list.append(final_st)
                new_squads_list.append(final_sq)
                new_histories_list.append(final_hist)

        # Ricostruzione Tensori GPU
        if not new_states_list: return

        self.sim_states = cp.array(new_states_list, dtype=cp.float32)
        self.squads = cp.array(new_squads_list, dtype=cp.float32)
        self.histories = new_histories_list
        self.n_sims = len(new_states_list)

    def prune(self):
        """Mantiene solo i BEAM_WIDTH migliori universi"""
        if self.n_sims <= BEAM_WIDTH:
            return

        scores = cp.zeros(self.n_sims, dtype=cp.float32)
        
        # Estrai dati necessari
        m_curr = self.sim_states[:, 0]
        e_curr = self.sim_states[:, 1]
        max_e = self.sim_states[:, 2]
        inventory = self.sim_states[:, 4:4+N_UNITS]
        
        # Calcola Income (Approx)
        m_income = cp.sum(inventory * G_M_YIELD, axis=1) + 2.0
        
        # Calcola Total BP
        total_bp = cp.sum(self.squads[:, :, 0], axis=1) # Sum BP of all squads (attive o meno)
        
        # Formula Score (dal prompt)
        # estimated_metal_consumption = total_bp * 0.025
        est_cons = total_bp * 0.025
        eff_throughput = cp.minimum(m_income, est_cons)
        
        # Metal Invested
        m_invested = cp.sum(inventory * G_M_COST, axis=1)
        
        # Penalty
        penalty = cp.zeros(self.n_sims, dtype=cp.float32)
        penalty += cp.where(e_curr < 50, 500, 0)
        
        has_storage = inventory[:, UNIT_MAP['EStorage']] > 0
        penalty += cp.where((e_curr > max_e * 0.9) & has_storage, 100, 0)
        penalty += cp.where((e_curr > 1000) & (~has_storage), 200, 0)
        
        scores = (eff_throughput * 100) + m_invested - penalty + total_bp
        
        # Sort e Slice
        # Argsort in CuPy
        sorted_indices = cp.argsort(scores)[::-1] # Descending
        keep_indices = sorted_indices[:BEAM_WIDTH]
        
        self.sim_states = self.sim_states[keep_indices]
        self.squads = self.squads[keep_indices]
        
        # Sync Python lists (History)
        # Convert to numpy for list indexing
        keep_indices_np = cp.asnumpy(keep_indices)
        self.histories = [self.histories[i] for i in keep_indices_np]
        
        self.n_sims = len(self.histories)

    def run(self):
        self.initialize()
        
        # Simulazione a blocchi
        for t in tqdm(range(0, TIME_LIMIT, EVAL_INTERVAL), desc="Simulating"):
            # 1. GPU Physics
            self.run_gpu_physics(EVAL_INTERVAL)
            
            # 2. Branching
            self.cpu_branching_and_logic()
            
            # 3. Pruning
            if self.n_sims > BEAM_WIDTH:
                self.prune()
                
            # Early exit se tutti morti (non dovrebbe succedere)
            if self.n_sims == 0:
                print("Extinction Event.")
                break

        # Final Ranking
        self.prune() # Sort final
        return self.histories[0], self.sim_states[0], self.squads[0]

# ==========================================
# 3. OUTPUT E GRAFICI
# ==========================================

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