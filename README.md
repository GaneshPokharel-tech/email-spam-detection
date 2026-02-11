# Email Spam Detection (SpamAssassin)

![CI](https://github.com/GaneshPokharel-tech/email-spam-detection/actions/workflows/ci.yml/badge.svg)


A simple end-to-end email spam detection project using the SpamAssassin public dataset.
Baseline model: TF-IDF + Logistic Regression (scikit-learn).
Includes a Streamlit app for quick testing (text input → prediction + confidence).

## Project Structure
- notebooks/ : Jupyter notebooks (training + evaluation)
- data/raw/ : raw dataset (ignored via .gitignore)
- models/ : saved pipeline + threshold
- streamlit_app/ : Streamlit app (app.py)
- .github/workflows/ : CI workflow (GitHub Actions)

## Local Setup

    git clone https://github.com/GaneshPokharel-tech/email-spam-detection.git
    cd email-spam-detection
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

## Download Dataset (SpamAssassin)

    cd data/raw
    curl -L -o spam_2.tar.bz2 https://spamassassin.apache.org/old/publiccorpus/20030228_spam_2.tar.bz2
    curl -L -o easy_ham.tar.bz2 https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2
    tar -xjf spam_2.tar.bz2
    tar -xjf easy_ham.tar.bz2

Expected folders:
- data/raw/spam_2/
- data/raw/easy_ham/

## Train + Evaluate (Notebook)

    jupyter notebook

Run: notebooks/01_data_load_and_split.ipynb

## Run Streamlit App

    streamlit run streamlit_app/app.py

## Threshold tuning
Edit: models/threshold.txt

Example:

    echo 0.60 > models/threshold.txt

## CI
GitHub Actions runs basic checks on push/PR.
