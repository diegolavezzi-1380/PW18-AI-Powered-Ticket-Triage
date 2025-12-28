# AI-Powered Ticket Classification System

Progetto di Machine Learning per il triage automatico dei ticket di assistenza, finalizzato alla classificazione per **categoria** e **priorità** a supporto di un service desk.

Il progetto è stato sviluppato come **Project Work universitario** con l’obiettivo di valutare approcci ML tradizionali, la qualità di un dataset sintetico realistico e la costruzione di una pipeline riproducibile e utilizzabile in contesto applicativo.

---

## Obiettivo

L’obiettivo del progetto è verificare la fattibilità di un sistema di **ticket triage automatico** basato su Machine Learning supervisionato, in grado di:

- classificare i ticket per categoria funzionale: *Amministrazione*, *Commerciale*, *Tecnico*;
- assegnare una priorità: *bassa*, *media*, *alta*;
- fornire un supporto decisionale agli operatori umani, senza sostituirne il giudizio.

Il focus è sulla **solidità metodologica**, sulla **riproducibilità** e sulla **chiarezza del processo**, non sull’uso di modelli complessi o LLM.

---

## Scope del progetto

**Incluso**
- Generazione di dataset sintetico realistico tramite template e regole deterministiche;
- Analisi esplorativa (EDA) del dataset;
- Addestramento e valutazione di modelli ML classici (SVM);
- Pipeline di training minimale e riproducibile;
- Prototipo applicativo tramite Streamlit.

**Fuori scope**
- Utilizzo di Large Language Models;
- Integrazione con sistemi di ticketing reali;
- Retraining automatico e gestione del concept drift;
- Deployment enterprise o MLOps avanzato.

---

## Struttura del repository

```
project-work/
│
├── data/
│   ├── raw/                 # Dataset grezzi e sintetici
│   ├── splits/              # Dataset preprocessati e suddivisi train/test
│   ├── results_baseline/    # Risultati baseline e metriche post EDA e confronto modelli
│   └── results_pipeline/    # Risultati pipeline finale
│
├── notebooks/               # EDA, analisi e confronto modelli
│
├── runs/                    # Archivio delle run di verifica effettuate con notebooks
│
├── src/
│   ├── app/                 # Applicazione Streamlit
│   │   └── models/          # Modelli addestrati utilizzati dall’applicazione
│   ├── config/              # Setting di progetto e Template di generazione
│   ├── data_generation/     # Generazione dataset sintetico
│   ├── pipelines/           # Pipeline ML (training e salvataggio modelli)
│   └── test_env.py          # Verifica ambiente
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Tecnologie utilizzate

- Python 3.11
- scikit-learn
- pandas, numpy
- Streamlit
- Faker (generazione dati sintetici)

---

## Setup dell’ambiente

1. Clonare il repository
2. Creare e attivare una virtual environment
3. Installare le dipendenze

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Esecuzione del progetto

### 1. Generazione del dataset

```bash
python -m src.data_generation.generate_realistic_tickets
```

Il dataset viene salvato nella cartella `data/raw/`.

---

### 2. Training della pipeline ML

```bash
python -m src.pipelines.train_svm_pipeline --input data/raw/tickets_realistic.csv
```

La pipeline addestra due modelli distinti:
- classificazione della categoria;
- classificazione della priorità.

I modelli e le metriche vengono salvati su disco per uso applicativo.

---

### 3. Analisi e notebook

I notebook nella cartella `notebooks/` includono:
- esplorazione e validazione del dataset sintetico;
- analisi delle distribuzioni e dei bias;
- confronto tra modelli ML;
- valutazione delle metriche (accuracy, F1, confusion matrix).

Il notebook 00_orchestrator_run_all funge da orchestratore per eseguire in pipeline tutti i notebook di analisi e verifica, i risultati delle run orchestrate sono archiviati in apposita cartella

I notebook sono parte integrante della valutazione scientifica, ma **non sono necessari per l’esecuzione della pipeline applicativa**.

---

## Risultati

I risultati quantitativi (metriche, confusion matrix) sono salvati nelle cartelle:
- `data/results_baseline/`
- `data/results_pipeline/`

L’analisi dettagliata dei risultati è documentata nei notebook e nell'archivio delle runs
- `runs/aaaammgg_HHmmss/`

---

## Dashboard applicativa

```bash
streamlit run src/app/TK_Triage.py
```

Dashboard applicativa con semplice interfaccia grafica utile a verificare i modelli addestrati:
- classificazione singola per titolo e testo : categoria prevista, priorità suggerita e parole più influenti.
- classificazione bacth : upload CSV e restituzione CSV aggiornato conn predizioni.

---

## Limiti del progetto

- Dataset sintetico non completamente rappresentativo di ticket reali;
- Pattern testuali parzialmente deterministici;
- Rischio di overfitting legato alla natura artificiale dei dati;
- Mancanza di rumore linguistico tipico di ambienti reali.

Questi limiti sono considerati parte integrante dell’analisi critica del progetto.

---

## Contesto

Project Work universitario per il corso L-31 di Laurea in Informatica.

