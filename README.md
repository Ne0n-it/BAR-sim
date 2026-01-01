# BAR-sim

Il software è un **simulatore economico** progettato specificamente per ottimizzare l'apertura (Early Game) di *Beyond All Reason*. Il suo scopo è trovare la sequenza di costruzione ("Build Order") che massimizza la crescita economica senza incappare in blocchi di risorse fatali.

Utilizza le librerie numpy, cupy multiprocessing e tqdm.

Funziona così: in base all'ordine degli eventi, creo realtà parallele e alla fine del tempo scelgo la migliore.

la struttura del software è divisa in due parti principali: CPU e GPU.
I compiti sono la gestione degli eventi, le logiche, le gerarchie, il punteggio delle varie simulazioni, il motore fisico delle simulazioni ed il loro avanzamento nel tempo.

### 1. I Dati logici
Come funziona il gioco è noto, ma noi useremo un pool ristretto di unità per calcolare i primi eventi, lasciamo il database ampliabile per il futuro.
*   **Evoluzioni:** alcune strutture possono essere costruite solo da unità particolari e seguono una gerarchia: 
          1. Chi cosstruisce cosa è visionabile nel database, 
          2. DefMex devono essere costruite automaticamente dalla stessa unità che ha costruito il Mex, ma solo dopo la numero n (es.N=2): sono le difese intorno a Mex che restano anche se queso diventa AdvMex,
*   **Parametri di gioco:** Vento (es. 14), Risorse Iniziali (1000 M/E), Tempo alla fine della quale trovare la simulazione (es. 5 minuti), capacità estrattiva media miniera ferro e scelta dell'unico tipo di fabbrica (es. VehLab che può evolvere in Adv_VehLab).
* initial_build_queue: Permetti di poter cominciare con una lista di scelte 'initial_build_queue' con la lista delle scelte che devono prendere le squadre (es. [(0, 0, 'Mex'), (0, 0, 'Solar'), (0, 0, FACTORY_TYPE)] #0 = squadra del Commander, 1 build, 2 unità da costruire)
*   **Parametri di mappa:** 
          1. Quante Mex totali ci sono nella mappa, 
          2. Distanza in secondi tra i Mex a partire dal precedente (Il primo valore è 0 (start), poi distanze relative es. MEX_DISTANCES = [0, 5, 5, 10]), la distanza si applica anche ai builder T2 quando devono trasformare Mex in AdvMex,
          3. I primi x Mex (x=2) sono semplici, successivi hanno fortificazioni di costo DefMex impostabile (es MEX_FORT_COST = {'m': 170, 'e': 1360, 'bp': 4800})
*   **Parametri di simulazione:** BEAM_WIDTH (quante simulazioni tenere), ogni quanto tempo valutare le simulazioni.

### 2. Regole di Branching
Unità costruttrici della SQUAD_NAMES hanno comportamenti specifici, ed ogni azione possibile crea ramificazioni negli universi.
- Commander: quando libero (IDLE) costruisce esmpre qualcosa tra le opzioni possibili,
- Factory T1 T2 e T3 (Experimental_Gantry_T3): quando sono liberi (IDLE) possono 1. costruire un Builder (ConVeh o Adv_ConVeh), 2. costruire un Offensive (AdvOffensive se T2 od un Titan se T3), o 3. aspettare T secondi (T=20),
- C'è differenza tra un Builder e una Builder Squads: Un builder è un'unità singola (es. ConVeh, ConBot o ConTurrent) che deve confluire in altre entità, mentre una Builder Squad è un gruppo di builder (SQUAD_NAMES) che lavorano insieme come un'unica entità con BP cumulativo e possono fare azioni diverse.
- Logica dei Builder: Un builder appena costruito puo:
    - Confluire in una Builder Squad (solo se già inizializzata) in SQUAD_NAMES, ovver Commander, FactoryT1, ConSquad1, FactoryT2, Adv_ConSquad o Experimental_Gantry_T3, ConTur possono solo confluire in fabbriche) aumentando il BP della squadra,
    - Creare una Builder Squad nuova se c'è uno slot libero e solo del tipo corretto (Builder T1 possono creare ConSquad1,2,3, Builder T2 possono creare Adv_ConSquad, ConTur non possono creare squadre), creando un universo per ciascuna possibilità, 
- Logica delle Builder Squads:
      - Una ConSquadX in IDLE può:
            - Dedicated Task: Diventa una squadra autonoma con un compito fisso (es. "Costruisci solo pannelli solari" o "Espandi solo Mex"),
            - Temporary Task: il builder T1 che costruisce Energy storage o una fabbrica non può continuare a costruirle perchè sono edifici unici, quindi alla fine della costruzione torna in IDLE, 
            - Upgrade: unirsi ad una fabbrica oppure ad una Adv_ConSquad,
- Wind non viene preso in considerazione come scelta se il vento è < 6,
- Travel Time: Simula il tempo perso per spostarsi, c'è una lista di tempi in successione quando si costruiscono Mex o GEO, ed un tempo fisso (t=5) quando una Builder Squad (tranne le fabbriche) creano una struttura di tipo diverso rispetto alla precedente.
      


### 3. GPU

Per rappresentare le BEAM_WIDTH simulazioni, ora useremo due strutture dati principali sulla GPU: una matrice 2D per lo stato globale e un tensor 3D per lo stato di tutte le squadre.

#### La Matrice sim_states (2D) - Lo Stato Globale

Questa matrice tiene traccia delle risorse, dell'inventario di strutture, e delle strutture in costruzione. La sua forma è (BEAM_WIDTH, N_GLOBAL_PROPERTIES).

Dimensioni: (BEAM_WIDTH(es. 8192), 4 + strutture costruite e in costruzione)

Colonne:
    [0] Metallo: Riserve attuali di metallo.
    [1] Energia: Riserve attuali di energia.
    [2] Max Energia: La capacità massima di immagazzinamento di energia.
    [3] Tempo Simulazione: I secondi trascorsi per questa simulazione.
    [4:] Inventario Strutture costruite e in costruzione: Una colonna per ogni tipo di unità nel gioco (N_UNITS). Conterrà il numero di strutture passive (es. Mex, Solar, EStorage). Le unità che sono squadre (come VehLab) avranno qui 1 se la struttura è costruita, ma i loro dettagli operativi saranno nel tensor squads.`

#### Il Tensor squads (3D) - Il Cuore della Simulazione

È un "cubo" di dati che rappresenta lo stato di ogni squadra, per ogni simulazione. La sua forma è (BEAM_WIDTH, N_SQUADS, N_SQUAD_PROPERTIES).

Dimensioni: (8192, 8, 7)
Prima Dimensione (8192): L'indice della simulazione.
Seconda Dimensione (7): L'indice della squadra (il "posto fisso").
    [0]: Commander
    [1]: FactoryT1
    [2]: ConSquad1
    [3]: ConSquad2
    [4]: ConSquad3
    [5]: FactoryT2
    [6]: Adv_ConSquad
    [7]: Experimental_Gantry_T3

Terza Dimensione (7): Le proprietà di quella specifica squadra.
    [0] build_power: Il BP attuale della squadra. È 0 se la squadra non è ancora stata costruita. Aumenta se è composta da più unità che si aggiungono.
    [1] status: Lo stato operativo attuale, codificato numericamente:
        0: INACTIVE (non ancora costruito)
        1: IDLE (costruito ma senza un compito)
        2: TRAVELING/WAITING (per quadre in spostamento o che devono aspettare)
        3: BUILDING (attivamente in costruzione/produzione)
    [2] task_unit_idx: L'indice dell'unità che sta costruendo/producendo (-1 se idle).
    [3] task_progress: I punti costruzione accumulati per il task attuale.
    [4] wait_time_remaining: I secondi rimanenti di attesa/spostamento.
    [5] mission_idx: L'indice dell'unità che questa squadra deve costruire in modo dedicato (es. UNIT_MAP['Solar']). Vale -1 per le squadre senza missione fissa (Commander, Fabbriche, Builder T2).
    [6] location_idx: L'indice della posizione sulla mappa (per calcolare i tempi di viaggio).

Quando un task è completato:
       La squadra costruttrice viene impostata su status = 1 (IDLE) a meno che non sia una squadra dedicata.
       Se l'unità costruita è una struttura passiva (es. Solar), il suo contatore viene incrementato nella matrice sim_states.

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

Lo script crea tutte le sequenze possibili, le valuta ogni 'EVAL_INTERVAL' secondi e tiene le più promettenti.
*   Cerca di fare il calcolo dello score con la GPU. 
*   Branching: universo diverso per ogni possobile scelta, per ogni simulazione/universo calcola tick per tick (secondo per secondo) le risorse.
*   Pruning (Potatura): Poiché le combinazioni sarebbero infinite, ogni 15 secondi (EVAL_INTERVAL) il sistema valuta tutte le simulazioni attive.
*   Survival of the Fittest: Mantiene vive solo le N migliori simulazioni (BEAM_WIDTH) basandosi sul punteggio. Le strategie inefficienti vengono cancellate dalla memoria.
*   Il punteggio per valutare le simulazioni e la vincitrice deve essere facilemente editabile nello script, per ora osserva questa regola:

```python
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
```

### 5. L'Output (Risultati)
Il programma restituisce tre cose:
1.  **Il "Build Order" Perfetto** delle prime X simulazioni (per ora solo la migliore): 
    - linguaggio naturale con la lista esatta delle azioni della simulazione vincente (tempo di inizo, tempo di fine attività, quale squadra la compie, BP della squadra),
    - una lista di tuple (squad_name, action, unit_built) per poter essere riutilizzata come initial_build_queue.
2.  **Statistiche Finali:** Quali strutture, quali unità e in che numero hai alla fine del tempo. Mostra le componenti dello Score (Metal_Spent, Metal_Income, Enegy_income e Build_Power_totale).
3.  **Grafici Diagnostici:**
    *   *Grafico Risorse:* Mostra le riserve di Metallo ed Energia. Permette di vedere i picchi negativi causati dalla costruzione di strutture ad alto consumo (Lab/Mex).
    *   *Grafico BP (Verde):* Mostra l'efficienza. Se la linea verde crolla, significa che quella strategia ha causato uno "Stallo" economico, rallentando tutto.
