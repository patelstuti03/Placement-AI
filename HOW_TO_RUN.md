# How to run PlacementAI on your computer

## 1. Install the requirements (one time)

Windows:

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

macOS / Linux:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## 2. (Optional but recommended) Rebuild the saved models

The saved model files were created with an older version of scikit-learn.
Run this once so they match the version on your machine:

    python retrain_models.py

The app also repairs any unreadable model automatically the first time you
make a prediction, so this step is only a shortcut.

## 3. Start the app

    python app.py

Open http://127.0.0.1:5000 in your browser, register an account, log in, and
everything (dashboard, prediction, history, EDA charts, upload, convert,
train & evaluate) works.

## What was fixed

- "Gradient Boosting: Error: No module named '_loss'" — the old model files
  could not be opened by newer scikit-learn. The app now rebuilds any model
  that fails to load and saves the working version back to `models/`.
- Added `retrain_models.py` to regenerate all nine models at once.
