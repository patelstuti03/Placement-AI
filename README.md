# Placement Prediction Web App

A **Flask-based machine learning web application** that predicts student placement outcomes using an ensemble of **9 trained classification models**.

---

## 🚀 Features

- 🔐 Secure session-based authentication (register / login / logout)
- 🤖 Multi-model prediction — 9 ML models run simultaneously:
  - Logistic Regression, Random Forest, Decision Tree
  - Naive Bayes, SVM, KNN, Gradient Boosting, LDA
  - Custom SVM+KNN Hybrid Classifier
- 📊 Interactive EDA visualizations (placement status, gender distribution, histograms, pairplots)
- 📁 Dataset upload, preprocessing, and int-encoding pipeline
- 🕓 Prediction history dashboard per user
- 🧠 In-app model training & evaluation with accuracy + confusion matrices

---

## 📁 Project Structure

```
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── .gitignore
│
├── models/                 # 9 pre-trained .pkl model files
├── data/                   # Cleaned CSV datasets
│   ├── cleaned_data_cleaned.csv
│   └── cleaned_data_int0.csv
│
├── templates/              # Jinja2 HTML templates
├── static/                 # CSS & JavaScript
│   ├── style.css
│   └── script.js
│
├── scripts/                # Utility / analysis scripts (not part of web app)
│   ├── eda_analysis.py     # EDA + model training CLI script
│   └── data_cleaning.py    # Raw dataset cleaning script
│
└── instance/               # SQLite database (auto-created, gitignored)
```

---

## ⚙️ Setup & Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/placement-prediction.git
cd placement-prediction

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
python app.py
```

Open your browser at **http://127.0.0.1:5000**

---

## 🧪 Models & Accuracy

The app ships with pre-trained models stored in the `models/` folder.  
You can also retrain them directly from the dashboard via the **Train & Evaluate** page.

| Model | File |
|---|---|
| Logistic Regression | `logisticregression_model.pkl` |
| Random Forest | `randomforestclassifier_model.pkl` |
| Decision Tree | `decisiontreeclassifier_model.pkl` |
| Naive Bayes | `gaussiannb_model.pkl` |
| SVM | `svc_model.pkl` |
| KNN | `kneighborsclassifier_model.pkl` |
| Gradient Boosting | `gradientboostingclassifier_model.pkl` |
| LDA | `lineardiscriminantanalysis_model.pkl` |
| SVM+KNN Hybrid | `svmknnhybridclassifier_model.pkl` |

---

## 🛠 Tech Stack

- **Backend:** Python, Flask, Flask-SQLAlchemy, SQLite
- **ML:** scikit-learn, joblib, NumPy, pandas
- **Visualization:** Matplotlib, Seaborn
- **Frontend:** Vanilla HTML/CSS/JS (glassmorphic dark theme)
