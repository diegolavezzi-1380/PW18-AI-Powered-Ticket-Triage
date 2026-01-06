"""
Pipeline di addestramento per il progetto di Triage Automatico dei Ticket.

Questo script implementa una pipeline di addestramento minimale e riproducibile
per la creazione dei modelli di classificazione dei ticket.

In particolare, lo script:
- carica un dataset in formato CSV contenente le colonne: title, body, category, priority;
- costruisce un unico campo testuale combinando titolo e corpo del ticket;
- applica una pulizia del testo semplice e deterministica;
- addestra due modelli di classificazione indipendenti:
    1) classificazione della categoria del ticket (Amministrazione / Commerciale / Tecnico);
    2) classificazione della priorità del ticket (bassa / media / alta);
- salva su disco le pipeline addestrate in formato .joblib.

Scelte progettuali:
- utilizzo di una pipeline scikit-learn per ciascun target, composta da TF-IDF Vectorizer e classificatore LinearSVC;
- suddivisione del dataset in training e test:
  - tramite una colonna 'split' già presente nel dataset (valori: train/test),
    se disponibile;
  - in alternativa, tramite uno split stratificato 80/20 sulla combinazione delle variabili 'category' e 'priority'.

Esecuzione (dalla root del progetto):
    python -m src.pipelines.train_svm_pipeline --input data/raw/tickets_realistic.csv

Esecuzione con split predefinito:
    python -m src.pipelines.train_svm_pipeline --input data/splits/tickets_preprocessed_split.csv --use-existing-split

Artefatti generati:
- app/models/category_model.joblib
- app/models/priority_model.joblib
- data/results_pipeline/svm_metrics.json (opzionale)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import joblib


# ==================================================
# Configurazione di training
# ==================================================
@dataclass(frozen=True)
class TrainConfig:
    title_col: str = "title"
    body_col: str = "body"
    text_col: str = "text"
    category_col: str = "category"
    priority_col: str = "priority"
    split_col: str = "split"

    test_size: float = 0.2
    random_state: int = 42

    # Parametri TF-IDF (valori definiti durante sperimentazione preliminare)
    max_features: Optional[int] = 200
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.8

    # Parametri LinearSVC
    C: float = 1.0


# ==================================================
# Preprocessing del testo (pulizia semplice)
# ==================================================
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    text = text.lower()
    text = re.sub(r"[^\w\sàèéìòù]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def try_get_italian_stopwords() -> Optional[Sequence[str]]:
    try:
        from nltk.corpus import stopwords
        return stopwords.words("italian")
    except Exception:
        return None


# ==================================================
# Caricamento e preparazione dei dati
# ==================================================
def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File CSV di input non trovato: {path}")

    df = pd.read_csv(path)

    colonne_richieste = {"title", "body", "category", "priority"}
    colonne_mancanti = colonne_richieste - set(df.columns)
    if colonne_mancanti:
        raise ValueError(
            f"Colonne obbligatorie mancanti {sorted(colonne_mancanti)} nel file {path}. "
            f"Colonne trovate: {df.columns.tolist()}"
        )

    return df


def build_text_column(df: pd.DataFrame, cfg: TrainConfig) -> pd.DataFrame:
    # Costruisce il campo testuale unico a partire da titolo e corpo
    df = df.copy()
    df[cfg.text_col] = (
        df[cfg.title_col].fillna("").astype(str)
        + " "
        + df[cfg.body_col].fillna("").astype(str)
    ).str.strip()

    # Rimozione difensiva di righe con testo vuoto e pulizia del testo
    df = df[df[cfg.text_col].str.len() > 0].copy()
    df[cfg.text_col] = df[cfg.text_col].map(clean_text)
    return df


def make_train_test_split(df: pd.DataFrame, cfg: TrainConfig,use_existing_split: bool = False,):
    X = df[cfg.text_col]
    y_cat = df[cfg.category_col]
    y_pri = df[cfg.priority_col]

    if use_existing_split:
        if cfg.split_col not in df.columns:
            raise ValueError("Opzione --use-existing-split attivata, ma la colonna "f"'{cfg.split_col}' non è presente nel dataset."
            )

        train_mask = df[cfg.split_col].astype(str).str.lower().eq("train")
        test_mask = df[cfg.split_col].astype(str).str.lower().eq("test")

        if not train_mask.any() or not test_mask.any():
            raise ValueError("La colonna 'split' deve contenere almeno un record 'train' e uno 'test'."
            )

        return (
            X[train_mask],
            X[test_mask],
            y_cat[train_mask],
            y_cat[test_mask],
            y_pri[train_mask],
            y_pri[test_mask],
        )

    # Split automatico stratificato sul abbinamento categoria + priorità
    strat = df["category"].astype(str) + "||" + df["priority"].astype(str)

    X_train, X_test, y_cat_train, y_cat_test = train_test_split(
        X,
        y_cat,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=strat,
    )

    y_pri_train = y_pri.loc[X_train.index]
    y_pri_test = y_pri.loc[X_test.index]

    return X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test


# ==================================================
# Costruzione e addestramento dei modelli
# ==================================================
def build_svm_pipeline(cfg: TrainConfig) -> Pipeline:
    stopwords_it = try_get_italian_stopwords()

    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        stop_words=stopwords_it,
        lowercase=True,
        ngram_range=cfg.ngram_range,
        max_features=cfg.max_features,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        strip_accents="unicode"
    )

    classifier = LinearSVC(
        class_weight="balanced",
        C=cfg.C, 
        random_state=cfg.random_state
    )

    return Pipeline(
        steps=[
            ("tfidf", vectorizer),
            ("classifier", classifier),
        ]
    )


def train_and_evaluate(pipeline: Pipeline, X_train: pd.Series, y_train: pd.Series, X_test: pd.Series, y_test: pd.Series,) -> dict:
    # Addestra la pipeline e restituisce metriche di valutazione essenziali.
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    }


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ==================================================
# Funzione principale
# ==================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Addestramento modelli TF-IDF + LinearSVC per il triage dei ticket."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Percorso del file CSV di input.",
    )
    parser.add_argument(
        "--use-existing-split",
        action="store_true",
        help="Utilizza una colonna 'split' (train/test) già presente nel dataset.",
    )
    parser.add_argument(
        "--out-dir",
        default="src/app/models",
        help="Directory di output per i modelli addestrati.",
    )
    parser.add_argument(
        "--metrics-path",
        default="data/results_pipeline/svm_metrics.json",
        help="Percorso del file JSON delle metriche (vuoto per disabilitare).",
    )

    args = parser.parse_args()
    cfg = TrainConfig()

    print("\n" + "=" * 60)
    print("AVVIO PIPELINE DI ADDESTRAMENTO")
    print("=" * 60)

    df = load_dataset(Path(args.input))
    df = build_text_column(df, cfg)

    X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test = make_train_test_split(
        df, cfg, use_existing_split=args.use_existing_split
    )

    print(f"\nNumero record training: {len(X_train)}")
    print(f"Numero record test:     {len(X_test)}")

    print("\nAddestramento modello di classificazione CATEGORIA...")
    cat_pipeline = build_svm_pipeline(cfg)
    cat_metrics = train_and_evaluate(
        cat_pipeline, X_train, y_cat_train, X_test, y_cat_test
    )

    print("Addestramento modello di classificazione PRIORITÀ...")
    pri_pipeline = build_svm_pipeline(cfg)
    pri_metrics = train_and_evaluate(
        pri_pipeline, X_train, y_pri_train, X_test, y_pri_test
    )

    out_dir = Path(args.out_dir)
    category_model_path = out_dir / "category_model.joblib"
    priority_model_path = out_dir / "priority_model.joblib"

    ensure_parent_dir(category_model_path)
    joblib.dump(cat_pipeline, category_model_path)
    joblib.dump(pri_pipeline, priority_model_path)

    print("\nModelli salvati:")
    print(f"- Categoria: {category_model_path}")
    print(f"- Priorità:  {priority_model_path}")

    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        ensure_parent_dir(metrics_path)
        metrics_payload = {
            "modello": "tfidf_linear_svc",
            "random_state": cfg.random_state,
            "test_size": cfg.test_size,
            "categoria": cat_metrics,
            "priorita": pri_metrics,
        }
        metrics_path.write_text(
            json.dumps(metrics_payload, indent=2),
            encoding="utf-8",
        )
        print(f"\nMetriche salvate in: {metrics_path}")

    print("\nRisultati:")
    print(
        f"- Categoria -> Accuracy: {cat_metrics['accuracy']:.3f}, "
        f"F1-macro: {cat_metrics['f1_macro']:.3f}"
    )
    print(
        f"- Priorità -> Accuracy: {pri_metrics['accuracy']:.3f}, "
        f"F1-macro: {pri_metrics['f1_macro']:.3f}"
    )

    print("\nPipeline completata con successo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
