# BAR-sim

Scrivi uno script python per windows:

Il software è un **simulatore economico** progettato specificamente per ottimizzare l'apertura (Early Game) di *Beyond All Reason*. Il suo scopo è trovare la sequenza di costruzione ("Build Order") che massimizza la crescita economica senza incappare in blocchi di risorse fatali.

Utilizza le librerie numpy e cupy (e anche tqdm)

Funziona così: in base all'ordine degli eventi, creo realtà parallele e alla fine del tempo scelgo la migliore.

La CPU gestisce gli eventi, le logiche, le gerarchie e il punteggio delle varie simulazioni, mentre la GPU gestisce il motore fisico delle simulazioni ed il loro avanzamento nel tempo.

### 1. I Dati logici - CPU
Come funziona il gioco è noto, ma noi useremo un pool ristretto di unità per calcolare i primi eventi, lasciamo il database ampliabile per il futuro.
*   **Unità:** Commander, Mex, Adv_Mex, DefMex , Solar, Adv_Solar, Reactor, Wind, E-Storage, Fabbriche T1 (BotLab, VehLab, AirLab) Fabbriche T2 (per ora solo Vehicle Lab avanzata) Builder T1 (ConAir, ConBot, ConVeh), Builder T2 (per ora solo Adv_ConVeh) e unità offensive T1 e T2 (Offensive e Adv_Offensive).
*   **Evoluzioni:** alcune strutture possono essere costruite solo da unità particolari e seguono una gerarchia: 
          1. Commander può costruire: Mex, DefMex, Solar, Wind, E-Storage e Fabbriche T1,
          2. Builder T1 possono costruire: Mex, DefMex, Solar, Adv_Solar, Wind, E-Storage e Fabbrica T2,
          3. Builder T2 (Adv_ConVeh) possono costruire: Adv_Mex e Reactor,
          4. la costruzione di Adv_Mex sostituisce Mex,
          5. DefMex devono essere costruite automaticamente dalla stessa unità che ha costruito il Mex, ma solo dopo la numero n (es.N=2): sono le difese intorno a Mex che restano anche se queso diventa AdvMex,
*   **Parametri di gioco:** Vento (es. 14), Risorse Iniziali (1000 M/E), Tempo Limite (es. 5 minuti), capacità estrattiva media miniera ferro e scelta dell'unico tipo di fabbrica (es. VehLab che può evolvere in Adv_VehLab).
* Permetti di poter cominciare con una lista di scelte predefinita facilmente modificabile (es. [(0, 'Mex'), (0, 'Solar'), (0, FACTORY_TYPE)] #[0] = Commander)
*   **Parametri di mappa:** 
          1. Quante Mex totali ci sono nella mappa, 
          2. Distanza in secondi tra i Mex a partire dal precedente (Il primo valore è 0 (start), poi distanze relative es. MEX_DISTANCES = [0, 5, 5, 10]), la distanza si applica anche ai builder T2 quando devono trasformare Mex in AdvMex,
          3. I primi x Mex (x=2) sono semplici, successivi hanno fortificazioni di costo DefMex impostabile (es MEX_FORT_COST = {'m': 170, 'e': 1360, 'bp': 4800})
*   **Parametri di simulazione:** BEAM_WIDTH (quante simulazioni tenere), ogni quanto tempo valutare le simulazioni.

### 2. Regole di Branching - CPU
Ogni azione possibile crea ramificazioni negli universi.
- Ogni volta che Commander o AdvBuilder sono liberi (IDLE) cominciano a costruire qualcosa, viene creato un universo per ogni tipo di costruzione tra quelle che possono costruire.
- Ogni volta che la fabbrica è in IDLE si crea un universo dove comincia a costruire un Builder, ed uno in cui comincia a costruire un Offensive (o AdvOffensive/Adv_ConVeh se fabbrica T2),
- Logica dei Costruttori T1 (Builder Squads)
      - Un builder T1 in IDLE ha davanti molte possibilità:
            - Assist Commander: aumenta il BP del commander o di una squadra di costruttori (come un unico super-costruttore), creando un universo per ogni squadra possibile,
            - Assist Factory: aumenta il BP di una fabbrica, se ci sono due tipi si crea un universo per ciascuno, 
            - Dedicated Task: Diventa una squadra autonoma con un compito fisso (es. "Costruisci solo pannelli solari" o "Espandi solo Mex"), creando un universo per ciascun compito possibile,
            - Temporary Task: il builder T1 che costruisce Energy storage o una fabbrica non può continuare a costruirle perchè sono edifici unici, quindi alla fine della costruzione torna in IDLE.
      - Travel Time: Simula il tempo perso per spostarsi, c'è una lista di tempi in successione quando si costruiscono i Mex, ed un tempo fisso (t=5) quando un commander o un Adv_VehCon creano una struttura di tipo diverso rispetto alla precedente
      - Wind non viene preso in considerazione come scelta se il vento è < 6,


### 3. GPU

Per rappresentare le BEAM_WIDTH simulazioni, ora useremo due strutture dati principali sulla GPU: una matrice 2D per lo stato globale e un tensor 3D per lo stato di tutte le squadre.

#### La Matrice sim_states (2D) - Lo Stato Globale

Questa matrice tiene traccia delle risorse e dell'inventario di strutture che non sono squadre. La sua forma è (BEAM_WIDTH, N_GLOBAL_PROPERTIES).

Dimensioni: (8192, 4 + N_UNITS)

Colonne:
    [0] Metallo: Riserve attuali di metallo.
    [1] Energia: Riserve attuali di energia.
    [2] Max Energia: La capacità massima di immagazzinamento di energia.
    [3] Tempo Simulazione: I secondi trascorsi per questa simulazione.
    [4:] Inventario Strutture: Una colonna per ogni tipo di unità nel gioco (N_UNITS). Conterrà il numero di strutture passive costruite (es. Mex, Solar, EStorage). Le unità che sono squadre (come VehLab) avranno qui 1 se la struttura è costruita, ma i loro dettagli operativi saranno nel tensor squads.`

#### Il Tensor squads (3D) - Il Cuore della Simulazione

È un "cubo" di dati che rappresenta lo stato di ogni squadra, per ogni simulazione. La sua forma è (BEAM_WIDTH, N_SQUADS, N_SQUAD_PROPERTIES).

Dimensioni: (8192, 7, 7)
Prima Dimensione (8192): L'indice della simulazione.
Seconda Dimensione (7): L'indice della squadra (il "posto fisso").
    [0]: Commander
    [1]: VehLab
    [2]: ConVeh_1
    [3]: ConVeh_2
    [4]: ConVeh_3
    [5]: Adv_VehLab
    [6]: Adv_ConVeh
Terza Dimensione (7): Le proprietà di quella specifica squadra.
    [0] build_power: Il BP attuale della squadra. È 0 se la squadra non è ancora stata costruita. Aumenta se è composta da più unità che si aggiungono.
    [1] status: Lo stato operativo attuale, codificato numericamente:
        0: INACTIVE (non ancora costruito)
        1: IDLE (costruito ma senza un compito)
        2: TRAVELING (in attesa/spostamento)
        3: BUILDING (attivamente in costruzione/produzione)
    [2] task_unit_idx: L'indice dell'unità che sta costruendo/producendo (-1 se idle).
    [3] task_progress: I punti costruzione accumulati per il task attuale.
    [4] wait_time_remaining: I secondi rimanenti di attesa/spostamento (gestiti dalla CPU). La GPU si limita a scalarli.
    [5] mission_idx: L'indice dell'unità che questa squadra deve costruire in modo dedicato (es. UNIT_MAP['Solar']). Vale -1 per le squadre senza missione fissa (Commander, Fabbriche, Builder T2).
    [6] location_idx: L'indice della posizione sulla mappa (per calcolare i tempi di viaggio).

Quando un task è completato:
       La squadra costruttrice viene impostata su status = 1 (IDLE).
       Se l'unità costruita è una struttura passiva (es. Solar), il suo contatore viene incrementato nella matrice sim_states.
       Se l'unità costruita è una nuova unità ci pensa la CPU perchè c'è una ramificazione (aumenta BP di squadra esistente, "crea" squadre nuova).


### 4. Il Ciclo di Simulazione

Simula la logica reale del motore di gioco (*Spring RTS*):
*   **Costruzione Dinamica:** Non sottrae il costo tutto subito. Il costo viene spalmato secondo per secondo in base al **Build Power (BP)** applicato.
*   **Formula del Drain:**
    $$Consumo_{Energia/sec} = \frac{CostoTotaleEnergia \times BP_{Applicato}}{BuildCost}$$
    *   *Esempio:* Costruire un Mex (500E) col Commander (300BP) richiede ~6 secondi e consuma **83 Energia/sec**. Dato che il Commander ne produce solo 30, questo crea un deficit immediato (il crollo nei grafici).
*   Gestione degli "Stalli" (Cascata di Inefficienza):
      - Stallo Energia: Se l'Energia finisce, i Mex riducono l'estrazione di Metallo (simulazione dello spegnimento) e i costruttori rallentano.
      - Stallo Metallo: Se manca Metallo, la velocità di costruzione (Efficiency) cala sotto il 100%.
      - Questo meccanismo punisce severamente le strategie che consumano più di quanto producono

Lo script crea tutte le sequenze possibili, le valuta ogni N secondi e tiene le più promettenti.
*   per ogni simulazione/universo, calcola tick per tick (secondo per secondo) le risorse.
*   Branching (Ramificazione): Ogni volta che un costruttore o una fabbrica finisce un lavoro, la simulazione si "sdoppia" in tante varianti quante sono le azioni possibili (es. "Costruisco Solar?", "Costruisco Mex?", "Faccio un Costruttore?").
*   Pruning (Potatura): Poiché le combinazioni sarebbero infinite, ogni 15 secondi (EVAL_INTERVAL) il sistema valuta tutte le simulazioni attive.
*   Survival of the Fittest: Mantiene vive solo le N migliori simulazioni (BEAM_WIDTH) basandosi sul punteggio. Le strategie inefficienti vengono cancellate dalla memoria.
*   Il punteggio per valutare le simulazioni e la vincitrice deve essere facilemente editabile nello script, per ora osserva questa regola:

```python
def get_score(self):
    m_prod, e_prod, _ = self.calculate_income()
    
    # Stima di quanto metallo possiamo consumare al massimo con il BP attuale
    # Assumiamo che 1 BP consumi circa 0.02 Metallo (media tra Mex e Solar)
    # Esempio: Commander (300 BP) consuma circa 5-6 metallo/sec sui Mex.
    estimated_metal_consumption_capacity = self.total_bp * 0.025
    
    # Il "Throughput Reale" è il minimo tra quanto produciamo e quanto possiamo spendere.
    # Se produco 20 ma posso spenderne 10 -> Score basato su 10 (sprecato income)
    # Se produco 10 ma posso spenderne 20 -> Score basato su 10 (sprecato BP)
    effective_throughput = min(m_prod, estimated_metal_consumption_capacity)
    
    # Bonus per il metallo già speso (per premiare l'aver costruito cose)
    m_invested = sum(self.inventory[u] * DB[u].get('m', 0) for u in self.inventory)

    # Penalità per stallo energetico (Inventory Check)
    # Se abbiamo accumulato troppa energia sprecata (storage pieno) o siamo a secco
    penalty = 0
    if self.energy < 50: penalty += 500
    if self.energy > self.max_e * 0.9 and self.inventory['EStorage'] > 0: penalty += 100
    if self.energy > 1000 and self.inventory['EStorage'] == 0: penalty += 200

    return (effective_throughput * 100) + m_invested - penalty + self.total_bp
```

### 5. L'Output (Risultati)
Il programma restituisce tre cose:
1.  **Il "Build Order" Perfetto:** La lista esatta delle azioni della simulazione vincente. (tempo di inizo, tempo di fine attività, quale squadra la compie, BP della squadra)
2.  **Statistiche Finali:** Quali strutture, quali unità e in che numero hai alla fine del tempo. Mostra le componenti dello Score (Metal_Spent, Metal_Income, Enegy_income e Build_Power_totale).
3.  **Grafici Diagnostici:**
    *   *Grafico Risorse:* Mostra le riserve di Metallo ed Energia. Permette di vedere i picchi negativi causati dalla costruzione di strutture ad alto consumo (Lab/Mex).
    *   *Grafico BP (Verde):* Mostra l'efficienza. Se la linea verde crolla, significa che quella strategia ha causato uno "Stallo" economico, rallentando tutto.

```python
DB = {
    'Commander': {
        'bp': 0, 'm': 0, 'e': 0, 'BP': 300, 'M': 2, 'E': 25, 
        'is_structure': False, 
        'build_options': ['Mex', 'Solar', 'Wind', 'EStorage', 'BotLab', 'VehLab', 'AirLab']
    },
    
    # STRUTTURE
    'Solar':    {'m': 155, 'e': 0, 'bp': 2600, 'E': 20, 'M': 0, 'BP': 0, 'is_structure': True, 'e_drain': 0},
    'Adv_Solar':{'m': 350, 'e': 5000, 'bp': 7950, 'E': 75, 'M': 0, 'BP': 0, 'is_structure': True, 'e_drain': 0},
    'Wind':     {'m': 40, 'e': 175, 'bp': 1600, 'E': WIND_LEVEL, 'M': 0, 'BP': 0, 'is_structure': True, 'e_drain': 0},
    'Reactor':  {'m': 4300, 'e': 21000, 'bp': 70000, 'E': 1000, 'M': 0, 'BP': 0, 'is_structure': True, 'e_drain': 0},
    'Mex':      {'m': 50, 'e': 500, 'bp': 1800, 'E': 0, 'M': MEX_METAL_YIELD, 'BP': 0, 'is_structure': True, 'e_drain': 3},
    'Adv_Mex':  {'m': 620, 'e': 7700, 'bp': 14900, 'E': 0, 'M': MEX_METAL_YIELD*4, 'BP': 0, 'is_structure': True, 'e_drain': 20},
    'EStorage': {'m': 260, 'e': 1700, 'bp': 4110, 'E': 0, 'M': 0, 'BP': 0, 'storage': 6000, 'is_structure': True, 'e_drain': 0},
    
    # DIFESA MEX (Automatica)
    'DefMex':   {'m': 190, 'e': 0, 'bp': 2400, 'E': 0, 'M': 0, 'BP': 0, 'is_structure': True, 'e_drain': 0},

    # FABBRICHE T1
    'AirLab':    {'m': 600, 'e': 1200, 'bp': 7000, 'BP': 100, 'is_structure': True, 'e_drain': 0, 'builder_name': 'ConAir', 'unit_options': ['ConAir', 'Offensive']},
    'BotLab':    {'m': 650, 'e': 1200, 'bp': 6500, 'BP': 100, 'is_structure': True, 'e_drain': 0, 'builder_name': 'ConBot', 'unit_options': ['ConBot', 'Offensive']},
    'VehLab':    {'m': 590, 'e': 1550, 'bp': 5700, 'BP': 100, 'is_structure': True, 'e_drain': 0, 'builder_name': 'ConVeh', 'unit_options': ['ConVeh', 'Offensive']},
    
    # FABBRICHE T2
    'Adv_VehLab':{'m': 2900, 'e': 14000, 'bp': 18000, 'BP': 300, 'is_structure': True, 'e_drain': 0, 'builder_name': 'Adv_ConVeh', 'unit_options': ['Adv_ConVeh', 'Adv_Offensive']},

    # COSTRUTTORI T1 (Rimossa opzione DefMex manuale)
    'ConAir':    {'m': 100, 'e': 2400, 'bp': 5000, 'BP': 50, 'is_structure': False, 'e_drain': 0, 'build_options': ['Mex', 'Solar', 'Wind', 'EStorage', 'Adv_Solar', 'Adv_VehLab']},
    'ConBot':    {'m': 110, 'e': 2200, 'bp': 4500, 'BP': 80, 'is_structure': False, 'e_drain': 0, 'build_options': ['Mex', 'Solar', 'Wind', 'EStorage', 'Adv_Solar', 'Adv_VehLab']},
    'ConVeh':    {'m': 135, 'e': 1950, 'bp': 4100, 'BP': 90, 'is_structure': False, 'e_drain': 0, 'build_options': ['Mex', 'Solar', 'Wind', 'EStorage', 'Adv_Solar', 'Adv_VehLab']},

    # COSTRUTTORI T2
    'Adv_ConVeh':{'m': 550, 'e': 6800, 'bp': 12400, 'BP': 250, 'is_structure': False, 'e_drain': 0, 'build_options': ['Adv_Mex', 'Reactor', 'Adv_Solar']},

    # UNITÀ
    'Offensive': {'m': 200, 'e': 2250, 'bp': 3500, 'BP': 0, 'is_structure': False, 'e_drain': 0},
    'Adv_Offensive': {'m': 950, 'e': 13000, 'bp': 17200, 'BP': 0, 'is_structure': False, 'e_drain': 0}
}

# Costi fortificazione per Mex > 2 (Indici 2,3,4...)
MEX_FORT_COST = {'m': 170, 'e': 1360, 'bp': 4800} 
```
