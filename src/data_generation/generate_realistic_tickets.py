"""
Script di generazione di un dataset sintetico realistico per il progetto di Triage Automatico dei Ticket.

Questo script genera un dataset sintetico di ticket di assistenza, simulando in modo 
controllato e riproducibile il comportamento reale degli utenti nella creazione di richieste di supporto.

Il dataset risultante viene utilizzato esclusivamente a fini sperimentali per l’addestramento e la valutazione 
di modelli di Machine Learning per la classificazione automatica di categoria e priorità.

In particolare, lo script:
- genera ticket appartenenti a più categorie funzionali (es. Amministrazione, Commerciale, Tecnico), secondo una distribuzione predefinita;
- costruisce titoli e descrizioni testuali a partire da template parametrizzati definiti in un file YAML esterno;
- assegna la priorità del ticket (bassa, media, alta) sulla base della presenza di parole chiave semantiche nel testo;
- introduce rumore controllato per aumentare il realismo del linguaggio, includendo:
    - errori di battitura;
    - abbreviazioni comuni;
    - mescolanza linguistica italiano/inglese;
    - inconsistenze di maiuscole/minuscole;
    - contaminazione semantica tra categorie;
- applica una logica di riequilibrio delle priorità per avvicinarsi a una distribuzione target, evitando dataset eccessivamente sbilanciati.

Scelte progettuali:
- separazione tra logica di generazione e configurazione, demandando a file esterni (settings.py e YAML) la definizione di template, keyword, distribuzioni e parametri di rumore;
- introduzione intenzionale di ambiguità e contaminazione semantica per ridurre il rischio di modelli trivialmente separabili;
- assenza di qualsiasi informazione reale o sensibile: tutti i dati generati sono completamente fittizi.

Output:
- generazione di un file CSV contenente il dataset sintetico, con le colonne principali:
    - id
    - title
    - body
    - category
    - priority
- stampa a console di statistiche descrittive sul dataset generato (distribuzione categorie, priorità, lunghezza testi).

Esecuzione:
Lo script può essere eseguito direttamente per rigenerare il dataset:
    python src/data/generate_realistic_tickets.py

Il file CSV viene salvato di default in: data/raw/tickets_realistic.csv
"""


import re
import yaml
import random
import pandas as pd
from faker import Faker
fake = Faker("it_IT")

from datetime import datetime
from src.config.settings import (
    TOTAL_TICKETS,
    CATEGORY_DISTRIBUTION,
    PRIORITY_TARGET,
    USE_TITLE,
    MODULES,
    AMBIENTS,
    CODICE_ERRORE,
    NOME_UTENTE,
    PAGINA_APPLICATIVA,
    VERSIONE_SOFTWARE,
    TIPO_DATO,
    FORMATO_FILE,
    PRODOTTO,
    EMAIL_UTENTE,
    TEMPO_MASSIMO,
    GIORNI_RICHIESTA,
    PRODUCT_LINES,
    HIGH_KEYWORDS,
    MEDIUM_KEYWORDS,
    LOW_KEYWORDS,
    # Parametri di controllo per il "rumore"
    NOISE_CONFIG,
    ABBREVIATIONS,
    CATEGORY_CONTAMINATION_KEYWORDS,
    PRIORITY_CONTAMINATION_KEYWORDS,
    SHARED_KEYWORDS,
    LANGUAGE_MIXING
)


# ==================================================
# Carica la configurazione YAML
# ==================================================
def load_config(path="src/config/tickets_templates.yaml"):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# ==================================================
# Normalizza il testo per il matching: porta tutti in minuscolo e rimuove spazzi doppi
# ==================================================
def normalize_text(text_in: str) -> str:
    text_out = text_in.lower().strip()
    
    while "  " in text_out:
        text_out = text_out.replace("  ", " ")
    return text_out


# ==================================================
# Ritorna True se almeno una keyword è presente nel testo normalizzato
# ==================================================
def contains_keyword(text_in: str, keywords: list[str]) -> bool:
    text_out = normalize_text(text_in)
    return any(kw in text_out for kw in keywords)


# ==================================================
# Assegna la priorità in base alla presenza di parole chiave nel testo
# ==================================================
def assign_priority(text: str) -> str:
    """
    Ordine di precedenza:
    - HIGH_KEYWORDS → 'alta'
    - MEDIUM_KEYWORDS → 'media'
    - LOW_KEYWORDS → 'bassa'
    - default → 'media'
    """
    if contains_keyword(text, HIGH_KEYWORDS):
        return "alta"
    
    if contains_keyword(text, MEDIUM_KEYWORDS):
        return "media"

    if contains_keyword(text, LOW_KEYWORDS):
        return "bassa"

    # Default neutro: media
    return "media"


# ==================================================
# Introduce errori di ortografia realistici
# ==================================================
def add_typos(text: str, typo_probability: float = 0.15) -> str:
    if len(text) < 5: return text
    
    words = text.split()
    for i in range(len(words)):
        word = words[i]
        # Applica typo solo se supera la probabilità E la parola è abbastanza lunga
        if len(word) > 3 and random.random() < typo_probability:
            # Scambia due caratteri casuali adiacenti
            idx = random.randint(0, len(word) - 2)
            word = word[:idx] + word[idx+1] + word[idx] + word[idx+2:]
            words[i] = word
    
    return ' '.join(words)


# ==================================================
# Sostituisce termini con abbreviazioni comuni
# ==================================================
def replace_with_abbreviations(text: str, replacement_probability: float = 0.2) -> str:
    for term, abbrevs in ABBREVIATIONS.items():
        if random.random() < replacement_probability:
            # Usa regex case-insensitive senza modificare tutto il testo
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            if pattern.search(text):
                abbrev = random.choice(abbrevs)
                text = pattern.sub(abbrev, text, count=1)  # Sostituisci solo una occorrenza
    
    return text


# ==================================================
# Aggiunge mescolanza linguistica (italiano-inglese)
# ==================================================
def add_language_mixing(text: str, mix_probability: float = 0.1) -> str:
    for it_term, en_term in LANGUAGE_MIXING.items():
        if random.random() < mix_probability:
            pattern = re.compile(re.escape(it_term), re.IGNORECASE)
            if pattern.search(text):
                text = pattern.sub(en_term, text, count=1)
    
    return text


# ==================================================
# Aggiunge informazioni extra in modo più naturale
# ==================================================
def add_extra_info_noise(text: str, noise_probability: float = 0.15) -> str:
    if random.random() < noise_probability and SHARED_KEYWORDS:
        noise_templates = [
            f" {random.choice(SHARED_KEYWORDS)}",
            f". Segnalato anche {random.choice(SHARED_KEYWORDS)}",
            f" - relativo a {random.choice(SHARED_KEYWORDS)}",
            f", problema su {random.choice(SHARED_KEYWORDS)}",
        ]
        text += random.choice(noise_templates)
    
    return text


# ==================================================
# Aggiunge contaminazione di keyword da altre categorie
# ==================================================
def create_contaminated_category(text: str, category: str, contamination_probability_category: float = 0.15) -> str:
    if random.random() < contamination_probability_category and CATEGORY_CONTAMINATION_KEYWORDS:
        other_keywords = []
        for other_cat, keywords in CATEGORY_CONTAMINATION_KEYWORDS.items():
            if other_cat != category:
                other_keywords.extend(keywords)
        
        if other_keywords:
            # Aggiungi 1-3 keyword da altre categorie in modo naturale
            num_contaminations = random.randint(1, 3)
            for _ in range(num_contaminations):
                contamination_templates = [
                    f" - {random.choice(other_keywords)}",
                    f" {random.choice(other_keywords)}",  
                    f" con {random.choice(other_keywords)}",
                    f" per {random.choice(other_keywords)}",
                ]
                text += random.choice(contamination_templates)
    
    return text

# ==================================================
# Aggiunge contaminazione di keyword da altre priorità
# ==================================================
def create_contaminated_priority(text: str, category: str, contamination_probability_priority: float = 0.15) -> str:
    if random.random() < contamination_probability_priority and PRIORITY_CONTAMINATION_KEYWORDS:
        other_keywords = []
        for other_cat, keywords in PRIORITY_CONTAMINATION_KEYWORDS.items():
            if other_cat != category:
                other_keywords.extend(keywords)
        
        if other_keywords:
            text += random.choice(random.choice(other_keywords))
    
    return text

# ==================================================
# Applica una serie di trasformazioni per aumentare il realismo
# ==================================================
def apply_realistic_transformations(title: str, body: str, category: str, noise_config: dict) -> tuple[str, str]:
   
    # TITLE TRANSFORMATIONS
    if USE_TITLE: 
        title = replace_with_abbreviations(title, replacement_probability=noise_config.get('abbreviation_probability', 0.30))
        title = create_contaminated_category(title, category, contamination_probability_category=noise_config.get('contamination_probability_category', 0.45))
    
    # BODY TRANSFORMATIONS
    body = add_typos(body, typo_probability=noise_config.get('typo_probability', 0.05))
    body = replace_with_abbreviations(body, replacement_probability=noise_config.get('abbreviation_probability', 0.30))
    body = add_language_mixing(body, mix_probability=noise_config.get('language_mixing_probability', 0.25))
    body = add_extra_info_noise(body, noise_probability=noise_config.get('extra_noise_probability', 0.25))
    body = create_contaminated_category(body, category, contamination_probability_category=noise_config.get('contamination_probability_category', 0.45))
    body = create_contaminated_priority(body, category, contamination_probability_priority=noise_config.get('contamination_probability_priority', 0.20))

    return title, body


# ==================================================
# Generazione ticket per categoria
# ==================================================
def generate_ticket(ticket_id: int, category: str, noise_config: dict) -> dict:
    # Scelta di titoli e body casuali
    config = load_config("src/config/tickets_templates.yaml")
    cat_key = category.lower() 
    if cat_key not in config:
        raise ValueError(f"Categoria {category} non trovata nel file YAML")

    title_template = ""
    if USE_TITLE:
        title_template = random.choice(config[cat_key]['titles'])
    
    body_template = random.choice(config[cat_key]['bodies'])

    # Setting delle variabili di supporto
    context = {
        "numero_fattura": f"{random.randint(1000, 9999)}/{random.randint(20, 24)}",
        "numero_ordine": f"ORD-{random.randint(10000, 99999)}",
        "importo": round(random.uniform(50, 1500), 2),
        "cliente": fake.company(),
        "email_cliente": fake.company_email(),
        "data_pagamento": fake.date(),
        "anno_fiscale": fake.year(),
        "quantita": random.randint(1, 10),
        "modulo": random.choice(MODULES),
        "ambiente":  random.choice(AMBIENTS),
        "linea_prodotto": random.choice(PRODUCT_LINES),
        "nome_utente": random.choice(NOME_UTENTE),
        "codice_errore": random.choice(CODICE_ERRORE),
        "pagina_applicativa": random.choice(PAGINA_APPLICATIVA),
        "versione_software": random.choice(VERSIONE_SOFTWARE),
        "tipo_dato": random.choice(TIPO_DATO),
        "formato_file": random.choice(FORMATO_FILE),
        "prodotto": random.choice(PRODOTTO), 
        "email_utente": random.choice(EMAIL_UTENTE), 
        "tempo_massimo": random.choice(TEMPO_MASSIMO), 
        "giorni_richiesta": random.choice(GIORNI_RICHIESTA)     
    }

    try:
        # Definizione del titolo e corpo del ticket
        title = title_template.format(**context)
        body = body_template.format(**context)
    except KeyError as e:
        print(f"Warning: Template mancante variabile {e} in categoria {category}")
        return None
    
    # Applicazione del rumore per trasformazioni realistiche
    title, body = apply_realistic_transformations(title, body, category, noise_config)

    # Definizione della priorità
    priority = assign_priority(title + " " + body)

    return {
        "id": ticket_id,
        "title": title,
        "body": body,
        "category": category,
        "priority": priority
    }


# ==================================================
# Generazione del dataset considerando la distribuzione 
# delle categorie e applicando il rumore configurato
# ==================================================
def generate_dataset(num_tickets: int = 500, noise_config: dict = None) -> pd.DataFrame:
    if noise_config is None: noise_config = NOISE_CONFIG
  
    rows = []
    ticket_id = 1
    priority_counts = {"alta": 0, "media": 0, "bassa": 0}
    priorities = ["alta", "media", "bassa"]

    for category, num_ticket in CATEGORY_DISTRIBUTION.items():
        ticket_count = int(TOTAL_TICKETS * num_ticket)
        for _ in range(ticket_count):

            # 1) Calcola distribuzione corrente
            current_ratios = {
                p: priority_counts[p] / max(1, ticket_id - 1)
                for p in priorities
            }

            # 2) Calcola deficit rispetto al target
            priority_deficit = {
                p: PRIORITY_TARGET[p] - current_ratios[p]
                for p in priorities
            }

            # 3) Qual è la priorità più sotto target?
            desired_priority = max(priority_deficit, key=priority_deficit.get)

            # 4) Decidi se vale la pena "spingere" verso desired_priority
            #    Se non c'è un vero deficit oltre il 3%, accetta qualsiasi priorità.
            must_match = priority_deficit[desired_priority] > 0.03      # soglia 3%

            # 5) Genera (e se serve rigenera) il ticket
            attempt = 0
            ticket = None

            while attempt < 30: # tentativi massimi prima di accettare qualsiasi ticket (evita loop infiniti)
                attempt += 1
                candidate = generate_ticket(ticket_id, category, noise_config)

                if (not must_match) or (candidate["priority"] == desired_priority):
                    ticket = candidate
                    break

            if ticket is None:
                ticket = candidate

            priority_counts[ticket["priority"]] += 1
            rows.append(ticket)
            ticket_id += 1
    
    df = pd.DataFrame(rows)
    return df


# ==================================================
# Funzioni di reporting
# ==================================================
def print_dataset_statistics(df: pd.DataFrame):
    print("\n" + "="*60)
    print("STATISTICHE DATASET GENERATO")
    print("="*60)
    
    print("\n=== DISTRIBUZIONE CATEGORIE ===")
    counts = df["category"].value_counts()
    percent = df["category"].value_counts(normalize=True) * 100
    summary = pd.DataFrame({
        "count": counts,
        "percent": percent.round(1)
    })
    print(summary)
    
    print("\n=== DISTRIBUZIONE PRIORITÀ ===")
    counts = df["priority"].value_counts()
    percent = df["priority"].value_counts(normalize=True) * 100
    summary = pd.DataFrame({
        "count": counts,
        "percent": percent.round(1)
    })
    print(summary)
    
    print("\n=== STATISTICHE TESTUALI ===")
    df['title_length'] = df['title'].str.len()
    df['body_length'] = df['body'].str.len()
    
    print(f"Lunghezza media titoli: {df['title_length'].mean():.1f} caratteri")
    print(f"  Min: {df['title_length'].min()}, Max: {df['title_length'].max()}")
    print(f"Lunghezza media body: {df['body_length'].mean():.1f} caratteri")
    print(f"  Min: {df['body_length'].min()}, Max: {df['body_length'].max()}")
    
    print("\n=== VARIABILITÀ REALISMO ===")
    print(f"Total tickets: {len(df)}")
    
    print("\n" + "="*60)
      

# ==================================================
# Analizza la presenza di parole chiave cross-category
# ==================================================
def dataset_cross_category_analisys(df: pd.DataFrame):
    for category in df['category'].unique():
        cat_data = df[df['category'] == category]
        
        print(f"\n{category}:")
        print(f"  Tickets totali: {len(cat_data)}")
        
        # Conta quanti contengono keyword di altre categorie
        other_categories = [c for c in df['category'].unique() if c != category]
        for other_cat in other_categories:
            other_keywords = CATEGORY_CONTAMINATION_KEYWORDS[other_cat]
            count = cat_data['body'].apply(
                lambda x: any(kw in x.lower() for kw in other_keywords)
            ).sum()
            print(f"  Contains {other_cat} keywords: {count} ({count/len(cat_data)*100:.1f}%)")


# ==================================================
# Funzione di base
# Genera il dataset e lo salva in CSV
# ==================================================
def main():
    print("\n" + "="*60)
    print("GENERAZIONE DATASET TICKET REALISTICI")
    print("="*60)
    print(f"\n" + f"Ticket toali: {TOTAL_TICKETS}")
    print(f"\n" + "="*60)
    print(f"\n" + f"Distribuzione categorie: {CATEGORY_DISTRIBUTION}")
    print(f"Distribuzione priorità: {PRIORITY_TARGET}")
    print(f"\n" + "="*60)
    print(f"\n" + f"Configurazione rumore: {NOISE_CONFIG}")
    
    df = generate_dataset(noise_config=NOISE_CONFIG)
    print_dataset_statistics(df)
    dataset_cross_category_analisys(df)

    # Salva il dataset
    output_path = "data/raw/tickets_realistic.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    archive_path = f"data/raw/tickets_realistic_{timestamp}.csv"
    df.to_csv(archive_path, index=False, encoding="utf-8")

    print("\n✓ DATASET GENERATO")
    print(f"✓ DATASET SALVATO IN: {output_path}")
    print(f"✓ DATASET ARCHIVIATO IN: {archive_path}\n")

if __name__ == "__main__":
    main()
