import io
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path


# Path base = cartella in cui si trova app.py (cioè src/)
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

def clean_text(text: str) -> str:
    """Pulizia semplice e riproducibile del testo dei ticket."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
        
    text = text.lower()
    # sostituisco numeri con token generico
    text = re.sub(r"\d+", " num ", text)
    # rimuovo punteggiatura non alfanumerica (mantengo caratteri accentati)
    text = re.sub(r"[^\w\sàèéìòù]", " ", text)
    # normalizzo spazi multipli
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

# =========================
# Caricamento modelli
# =========================
@st.cache_resource
def load_models():
    category_model_path = MODELS_DIR / "category_model.joblib"
    priority_model_path = MODELS_DIR / "priority_model.joblib"

    if not category_model_path.exists():
        raise FileNotFoundError(f"Modello categoria non trovato: {category_model_path}")
    if not priority_model_path.exists():
        raise FileNotFoundError(f"Modello priorità non trovato: {priority_model_path}")

    category_model = joblib.load(category_model_path)
    priority_model = joblib.load(priority_model_path)
    return category_model, priority_model


def combine_title_body(title: str, body: str) -> str:
    title = title or ""
    body = body or ""
    return (title.strip() + " " + body.strip()).strip()


# ==========================================
# Parole più influenti per una predizione
# ==========================================
def get_top_influential_words(text: str, pipeline, predicted_label: str, top_n: int = 5):
     # 1) recupera gli step
    try:
        vectorizer = pipeline.named_steps["tfidf"]
    except KeyError:
        raise ValueError("La pipeline non ha uno step 'tfidf' (controlla i nomi degli step).")

    clf = pipeline.named_steps[list(pipeline.named_steps.keys())[-1]]  # ultimo step = classifier

    # 2) vettorizza il testo
    X = vectorizer.transform([text])  # shape (1, n_features)

    # 3) trova l'indice della classe predetta
    classes = clf.classes_
    if isinstance(predicted_label, str):
        # label è stringa, cerco nell'array delle classi
        class_idx = np.where(classes == predicted_label)[0][0]
    else:
        # label numerica
        class_idx = int(predicted_label)

    # 4) prendi i coefficienti per quella classe
    # LogisticRegression: clf.coef_[class_idx, :]
    # LinearSVC: idem
    coefs = clf.coef_[class_idx]  # shape (n_features,)

    # 5) contributo = tfidf_value * coef
    # X è sparsa, uso multiply
    contributions = X.multiply(coefs).toarray().ravel()

    # 6) feature names
    feature_names = np.array(vectorizer.get_feature_names_out())

    # 7) tieni solo parole effettivamente presenti nel testo (tfidf > 0)
    nonzero_indices = np.where(X.toarray().ravel() > 0)[0]
    if len(nonzero_indices) == 0:
        return []

    contrib_nonzero = contributions[nonzero_indices]
    feat_nonzero = feature_names[nonzero_indices]

    # 8) ordina per contributo decrescente
    top_indices = np.argsort(contrib_nonzero)[::-1][:top_n]
    top_words = feat_nonzero[top_indices]

    return list(top_words)


# ========================
# Predizione singola
# ========================
def predict_single_ticket(title: str, body: str, cat_model, prio_model):
    text = combine_title_body(title, body)

    # predizione categoria
    cat_pred = cat_model.predict([text])[0]

    # predizione priorità
    prio_pred = prio_model.predict([text])[0]

    # parole più influenti
    top_words = get_top_influential_words(text, cat_model, cat_pred, top_n=5)

    return cat_pred, prio_pred, top_words


# ===========================
# Predizione su CSV batch
# ===========================
def predict_batch_csv(df: pd.DataFrame, cat_model, prio_model):
    if "title" not in df.columns or "body" not in df.columns:
        raise ValueError("Il CSV deve contenere le colonne 'title' e 'body'.")

    texts = df["title"].fillna("") + " " + df["body"].fillna("")

    cat_preds = cat_model.predict(texts)
    prio_preds = prio_model.predict(texts)

    df_out = df.copy()
    df_out["pred_category"] = cat_preds
    df_out["pred_priority"] = prio_preds

    return df_out


# ===========================
# Interfaccia Streamlit
# ===========================
def main():
    st.set_page_config(
        page_title="Ticket Triage Dashboard",
        page_icon="🎫",
        layout="centered",
    )

    st.title("Dashboard di Triage Automatico dei Ticket")
    st.write(
        "Prototipo di classificazione automatica dei ticket: "
        "data in input un titolo e un corpo di testo, il sistema predice la **categoria** "
        "(Amministrazione / Tecnico / Commerciale) e la **priorità** (bassa / media / alta)."
    )

    # carica modelli
    try:
        cat_model, prio_model = load_models()
    except Exception as e:
        st.error(
            "Errore nel caricamento dei modelli. "
            "Verifica che i file 'models/category_model.joblib' e "
            "'models/priority_model.joblib' esistano e siano stati salvati correttamente."
        )
        st.exception(e)
        return

    tab_single, tab_batch = st.tabs(["Singolo ticket", "Batch CSV"])

    # -------------------------
    # TAB 1: Singolo ticket
    # -------------------------
    with tab_single:
        st.subheader("Classificazione di un singolo ticket")

        title = st.text_input("Titolo del ticket", value="")
        body = st.text_area("Descrizione del ticket", value="", height=150)

        if st.button("Classifica ticket", type="primary"):
            if not title and not body:
                st.warning("Inserisci almeno il titolo o il corpo del ticket.")
            else:
                cat_pred, prio_pred, top_words = predict_single_ticket(
                    title, body, cat_model, prio_model
                )

                st.markdown("### Risultati")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Categoria prevista", str(cat_pred))
                with col2:
                    st.metric("Priorità suggerita", str(prio_pred))

                st.markdown("#### Parole più influenti (categoria)")
                if top_words:
                    st.write(", ".join(top_words))
                else:
                    st.write("Nessuna parola significativa individuata (testo troppo breve o fuori vocabolario).")

    # -------------------------
    # TAB 2: Batch CSV
    # -------------------------
    with tab_batch:
        st.subheader("Predizione su batch di ticket (CSV)")

        st.write(
            "Carica un file CSV con almeno le colonne **title** e **body**. "
            "Se presente, la colonna **id** sarà mantenuta."
        )

        uploaded_file = st.file_uploader("Carica CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error("Errore nella lettura del CSV. Controlla il formato del file.")
                st.exception(e)
                return

            st.write("Anteprima dati (prime 5 righe):")
            st.dataframe(df.head())

            if st.button("Esegui predizione batch"):
                try:
                    df_pred = predict_batch_csv(df, cat_model, prio_model)
                except Exception as e:
                    st.error("Errore durante la predizione batch.")
                    st.exception(e)
                    return

                st.write("Risultati (prime 10 righe):")
                st.dataframe(df_pred.head(10))

                # prepara CSV per download
                csv_buf = io.StringIO()
                df_pred.to_csv(csv_buf, index=False)
                csv_bytes = csv_buf.getvalue().encode("utf-8")

                st.download_button(
                    label="Scarica CSV con predizioni",
                    data=csv_bytes,
                    file_name="ticket_predictions.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
