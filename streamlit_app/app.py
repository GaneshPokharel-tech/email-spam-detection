import joblib
from pathlib import Path
import streamlit as st

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "spam_pipeline.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.txt"

# --- Load model only (cache ok) ---
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

pipeline = load_model()

def read_threshold():
    return float(THRESHOLD_PATH.read_text().strip())

# --- UI ---
st.title("Email Spam Detection (SpamAssassin)")
st.write("Paste an email text and click Predict.")

email_text = st.text_area("Email text", height=200)

if st.button("Predict"):
    if not email_text.strip():
        st.warning("Please enter some email text.")
    else:
        threshold = read_threshold()  # read latest value

        # Probability of spam (class 1)
        spam_proba = pipeline.predict_proba([email_text])[0][1]
        prediction = 1 if spam_proba >= threshold else 0

        if prediction == 1:
            st.error("Prediction: SPAM ✅")
        else:
            st.success("Prediction: HAM ✅")

        st.write(f"Spam probability: **{spam_proba:.3f}**")
        st.write(f"Threshold used: **{threshold:.2f}**")
