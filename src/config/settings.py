# ==================================================
# CONFIGURAZIONE GENERALE DEL DATASET
# ==================================================

TOTAL_TICKETS = 480

CATEGORIES = ["Amministrazione", "Tecnico", "Commerciale"]

# DISTRIBUZIONE TARGET CATEGORIE (distibuzione realistica)
CATEGORY_DISTRIBUTION = {
    "Amministrazione": 144,     # 30%
    "Tecnico": 216,             # 45%
    "Commerciale": 120          # 25%
}

# DISTRIBUZIONE TARGET PRIORITÀ (distibuzione realistica)
PRIORITY_TARGET = {
    "alta": 0.28,    # 28%
    "media": 0.40,   # 40%
    "bassa": 0.32    # 32%
}

#ABILITA L'UTILIZZO DEL TITOLO TEMPLATE NELLA GENERAZIONE DEI TICKET
USE_TITLE = True

# ==================================================
# CONFIGURAZIONE DEL RUMORE E REALISMO DEL DATASET
# ==================================================
NOISE_CONFIG = {
    # Probabilità di introdurre errori di ortografia (typo)
    "typo_probability": 0.05,                   # 5% di ticket con typo 
    
    # Probabilità di sostituire termini con abbreviazioni
    "abbreviation_probability": 0.30,           # 30% di sostituzione abbreviazioni
    
    # Probabilità di miscelare IT/EN
    "language_mixing_probability": 0.25,        # 25% di mixing linguistico nel body
    
    # Probabilità di aggiungere info extra incoerente
    "extra_noise_probability": 0.25,            # 25% di info "rumorosa" extra
    
    # Probabilità di contaminazione tra categorie
    "contamination_probability_category": 0.45,          # 45% di keyword da altre categorie

    # Probabilità di contaminazione tra priorità
    "contamination_probability_priority": 0.20, # 20% di keyword da altre priorità
}


# ==================================================
# LISTE VARIABILI TESTUALI 
# ==================================================
MODULES = ["fatturazione", "ordini", "magazzino", "assistenza","reportistica", "vendite", "catalogo", "clienti"]
AMBIENTS = ["produzione", "pre-produzione", "test", "sviluppo", "demo", "staging"]
PRODUCT_LINES = ["Linea Basic", "Linea Pro", "Linea Enterprise","Linea Accessori", "Linea Premium"]
CODICE_ERRORE = ["500", "502", "503", "504", "ERR_CONNECTION_REFUSED", "ERR_TIMEOUT", "SQL_ERROR_1045", "NullPointerException", "OutOfMemoryError"]
NOME_UTENTE = ["mario.rossi", "giulia.verdi", "admin_01", "operatore_23"]               
PAGINA_APPLICATIVA = ["dashboard principale", "schermata ordini", "report vendite", "gestione utenti"]            
VERSIONE_SOFTWARE = ["v2.3.1", "v3.0.0", "v2.5.4"]
TIPO_DATO = ["anagrafici", "transazioni", "inventario", "storico ordini"]
FORMATO_FILE = ["CSV", "Excel", "PDF", "XML"]
PRODOTTO = ["ERP", "CRM", "Sistema Gestionale"] 
EMAIL_UTENTE = ["acquisti@acmecorp.com", "procurement@techsolutions.it", "info@globaltrade.com", "commerciale@innovatech.it"]
TEMPO_MASSIMO = ["entro oggi", "entro 2 ore", "in giornata"]
GIORNI_RICHIESTA = ["5", "7", "10", "15"]


# ==================================================
# KEYWORD PER ASSEGNAZIONE PRIORITÀ AUTOMATICA
# ==================================================
HIGH_KEYWORDS = [
    "urgente", "bloccato", "critico", "immediato", "emergenza",
    "urgentissimo", "emergenza", "critico grave",
    "bloccato totale", "down completo", "crash critico",
    "perdita dati", "sistema fermo", "produzione ferma",
    "entro oggi", "entro 2 ore", "immediato assoluto"
]

MEDIUM_KEYWORDS = [
    "impossibile", "non funziona", "problema", "errore", "timeout",
    "anomalia", "malfunzionamento", "bug", "discrepanza",
    "lento", "rallentamento", "prestazioni",
    "necessario", "richiesto", "entro settimana",
    "verifica", "controllo", "lentezza", "performance degradate",
    "incongruenza", "mancato allineamento",
    "parziale", "sporadico", "intermittente", "funzionalità",
    "errore validazione", "non visualizza",
    "entro 3 giorni", "entro venerdì", "odierna",
    "scaduta", "necessaria rettifica", 
    "supporto operativo", "permessi insufficienti"
]

LOW_KEYWORDS = [
    "quando possibile", "senza urgenza", "non urgente",
    "prossimo mese", "prossime settimane", "pianificare",
    "archiviazione", "storico", "documentazione",
    "miglioramento", "suggerimento",
    "informativo", 
    "chiarimento", "informazioni generali",
    "materiale informativo", 
    "analisi storica", "report",
    "pianificazione", "gradualmente",
    "pianificabile", "quando riuscite",
    "pianificata", "pianificato", "schedulare", 
    "ordinaria", "uso interno", "per conoscenza", 
    "non bloccante", "domanda generica", 
    "monitoraggio", "manuale", "esplorativa", 
    "lieve", "minore", "ignorare", 
    "archivio", "data entry", "statistiche", "futura",
    "tracciamento interno", "statistico", 
    "manutenzione ordinaria", "manualistica", 
    "soddisfazione", "data cleaning"
]


# ==================================================
# ABBREVIAZIONI NON STANDARDIZZATE
# ==================================================
ABBREVIATIONS = {
    "amministrativo": ["admin", "amm.", "amministr.", "adm", "amm"],
    "tecnico": ["tech", "tec.", "problemi tecnici", "issue tec"],
    "commerciale": ["comm.", "com.", "sales", "comm"],
    "password": ["pwd", "psw", "pass", "pw"],
    "errore": ["err", "err.", "problem", "bug"],
    "sistema": ["sys", "sys.", "applicativo"],
    "accesso": ["acc.", "login", "access"],
    "fattura": ["fatt.", "fat", "doc"],
    "urgente": ["urg.", "urgentissimo", "ASAP"],
    "problema": ["pb", "pb.", "issue"],
    "contatto": ["cont.", "email"],
}

# ==================================================
# LANGUAGE MIXING IT/EN
# ==================================================
LANGUAGE_MIXING = {
    "errore": "error",
    "problema": "issue",
    "accesso": "access",
    "sistema": "system",
    "non funziona": "not working",
    "urgente": "urgent",
    "critico": "critical",
    "verifica": "check",
    "aggiornamento": "update",
    "report": "report",
}

# ==================================================
# KEYWORD DI CONTAMINAZIONE TRA CATEGORIE 
# ==================================================
CATEGORY_CONTAMINATION_KEYWORDS = {
    "Amministrazione": [
        "pagamento", "fattura", "importo", "bonifico", "scadenza",
        "pianificare", "suggerimento", "lieve"
    ],
    "Tecnico": [
        "errore critico", "login fallito", "performance", "crash",
        "sincronizzazione", "bug", "memory leak", "timeout"
    ],
    "Commerciale": [
        "preventivo", "sconto", "disponibilità", "ordine urgente",
        "condizioni commerciali", "listino", "negoziazione"
    ],
}

# ==================================================
# KEYWORD DI CONTAMINAZIONE TRA PRIORITÀ 
# ==================================================
PRIORITY_CONTAMINATION_KEYWORDS = {
    "Alta": [
        "anomalia", "bug", "lento", "scaduta",
        "suggerimento", "informativo", "chiarimento"
    ],
    "Media": [
        "critico", "bloccato", "fermo", "storico",
        "lieve", "statistico", "proposta"
    ],
    "Bassa": [
        "perdita dati", "sistema fermo", "produzione ferma",
        "impossibile", "non funziona", "problema", "errore", "timeout"
    ],
}

# ==================================================
# KEYWORD COMUNI TRA CATEGORIE 
# ==================================================
SHARED_KEYWORDS = [
    "cliente", "sistema", "utente", "dati", "modulo", 
    "verifica", "problema", "accesso", "richiesta", 
    "aggiornamento", "gestione", "controllo"," [vedi nota allegata]",
    "contattare supporto", " richiesta esterna",
    " cc: direttore", " ringraziamo", " grazie mille", " help!",
    "urgente", "errore", "blocco", "non funziona", "timeout",
    "configurazione", "report", "documento", "informazioni",
    "email", "account", "ambiente", "versione", "file",
    "ordine", "fattura", "preventivo", "contratto", "pagamento",
    "assistenza", "analisi", "IMPORTANTE"
]

