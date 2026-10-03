from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from functools import wraps
import os, json, io, base64, csv, pickle, warnings
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

warnings.filterwarnings("ignore")
plt.switch_backend('Agg')

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(ROOT_DIR, 'data')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')

# ── Custom classifier (needed to load the pkl) ─────────────────────────────────
def _unique_labels(*args):
    return np.unique(np.hstack(args))

class SVMKNNHybridClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.classes_ = None
        self.svm = SVC()
        self.knn = KNeighborsClassifier()
        self.hybrid_model = VotingClassifier(
            estimators=[('svm', self.svm), ('knn', self.knn)], voting='hard')

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = _unique_labels(y)
        self.svm.fit(X, y)
        self.knn.fit(X, y)
        self.hybrid_model.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        svm_pred = self.svm.predict(X)
        knn_pred = self.knn.predict(X)
        hybrid_pred = []
        for i in range(len(X)):
            if svm_pred[i] == knn_pred[i]:
                hybrid_pred.append(svm_pred[i])
            else:
                hybrid_pred.append(self.hybrid_model.predict(X[i].reshape(1, -1))[0])
        return hybrid_pred

class combine(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, svm_kernel='rbf'):
        self.n_neighbors = n_neighbors
        self.svm_kernel = svm_kernel
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.svm = SVC(kernel=svm_kernel)

    def fit(self, X, y):
        self.knn.fit(X, y)
        self.svm.fit(X, y)
        return self

    def predict(self, X):
        knn_pred = self.knn.predict(X)
        svm_pred = self.svm.predict(X)
        return np.where(knn_pred == svm_pred, knn_pred,
                        np.round((knn_pred + svm_pred) / 2))

# ── App ────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'placement_prediction_secret_2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

UPLOAD_FOLDER        = 'static/upload_dataset'
PREPROCESSED_FOLDER  = 'static/preprocessed_data'
INT_DATASET_FOLDER   = 'static/int_dataset'
MODEL_FOLDER         = 'static/trained_models'
TRANSFORMED_DATA_FOLDER = 'static/transformed_data'

for folder in [UPLOAD_FOLDER, PREPROCESSED_FOLDER, INT_DATASET_FOLDER,
               MODEL_FOLDER, TRANSFORMED_DATA_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER']          = UPLOAD_FOLDER
app.config['PREPROCESSED_FOLDER']    = PREPROCESSED_FOLDER
app.config['INT_DATASET_FOLDER']     = INT_DATASET_FOLDER
app.config['MODEL_FOLDER']           = MODEL_FOLDER
app.config['TRANSFORMED_DATA_FOLDER'] = TRANSFORMED_DATA_FOLDER

db = SQLAlchemy(app)

# ── Models ─────────────────────────────────────────────────────────────────────
class User(db.Model):
    __tablename__ = 'users'
    id       = db.Column(db.Integer, primary_key=True)
    email    = db.Column(db.String(100), unique=True, nullable=False)
    username = db.Column(db.String(50),  unique=True, nullable=False)
    password = db.Column(db.String(100), nullable=False)

class Prediction(db.Model):
    __tablename__ = 'predictions'
    id        = db.Column(db.Integer, primary_key=True)
    user_id   = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    inputs    = db.Column(db.Text, nullable=False)   # JSON string
    results   = db.Column(db.Text, nullable=False)   # JSON string
    overall   = db.Column(db.String(20), nullable=False)

with app.app_context():
    db.create_all()

# ── Auth helper ────────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ── Feature config ─────────────────────────────────────────────────────────────
FEATURES = ['backlog', 'course', 'branch', 'cohabit', 'scholarship', 'hsc_branch',
            'caste', 'weekly_studytime', 'extra_classes', 'attendance',
            'degree_result', 'hsc_result', 'hsc_medium', 'communication_rating',
            'certification', 'no_of_internship']

CAT_FEATURES = ['course', 'branch', 'cohabit', 'scholarship', 'hsc_branch',
                'caste', 'weekly_studytime', 'extra_classes', 'attendance',
                'hsc_medium', 'certification']

MODEL_FILES = {
    'Logistic Regression':    'logisticregression_model.pkl',
    'Random Forest':          'randomforestclassifier_model.pkl',
    'Decision Tree':          'decisiontreeclassifier_model.pkl',
    'Naive Bayes':            'gaussiannb_model.pkl',
    'SVM':                    'svc_model.pkl',
    'KNN':                    'kneighborsclassifier_model.pkl',
    'Gradient Boosting':      'gradientboostingclassifier_model.pkl',
    'LDA':                    'lineardiscriminantanalysis_model.pkl',
    'SVM+KNN Hybrid':         'svmknnhybridclassifier_model.pkl',
}

def build_estimators():
    """Fresh, unfitted estimators keyed exactly like MODEL_FILES."""
    return {
        'Logistic Regression':  LogisticRegression(max_iter=1000),
        'Random Forest':        RandomForestClassifier(random_state=42),
        'Decision Tree':        DecisionTreeClassifier(random_state=42),
        'Naive Bayes':          GaussianNB(),
        'SVM':                  SVC(),
        'KNN':                  KNeighborsClassifier(),
        'Gradient Boosting':    GradientBoostingClassifier(random_state=42),
        'LDA':                  LinearDiscriminantAnalysis(),
        'SVM+KNN Hybrid':       SVMKNNHybridClassifier(),
    }

# Cache so a rebuilt model is only trained once per process
_MODEL_CACHE = {}

def load_model(name, X_full=None, y_full=None, scaler=None):
    """Load a saved model. If the .pkl was written by a different
    scikit-learn version and cannot be unpickled, retrain it from the
    dataset and overwrite the file so it works from now on."""
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]

    path = os.path.join(MODEL_DIR, MODEL_FILES[name])
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            _MODEL_CACHE[name] = model
            return model
        except Exception as exc:
            print(f'[models] "{name}" could not be loaded ({exc}); retraining.')

    if X_full is None or y_full is None:
        return None

    estimator = build_estimators()[name]
    X_train = scaler.transform(X_full) if scaler is not None else X_full
    estimator.fit(X_train, y_full)
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(estimator, path)
    except Exception:
        pass
    _MODEL_CACHE[name] = estimator
    return estimator



def get_encoding_maps():
    """Build {col: {raw_str: int}} mappings from the two CSV files."""
    raw_path = os.path.join(DATA_DIR, 'cleaned_data_cleaned.csv')
    int_path = os.path.join(DATA_DIR, 'cleaned_data_int0.csv')
    if not os.path.exists(raw_path) or not os.path.exists(int_path):
        return {}
    raw_df = pd.read_csv(raw_path, encoding='latin1')
    int_df = pd.read_csv(int_path, encoding='latin1')
    maps = {}
    for col in CAT_FEATURES:
        if col in raw_df.columns and col in int_df.columns:
            m = {}
            for rv, iv in zip(raw_df[col], int_df[col]):
                if pd.notna(rv):
                    m[str(rv).strip()] = int(iv)
            maps[col] = m
    return maps

# ── Auth routes ────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('home'))
    if request.method == 'POST':
        email    = request.form.get('email', '').strip()
        password = request.form.get('pass', '').strip()
        user = User.query.filter_by(email=email, password=password).first()
        if user:
            session['user_id']  = user.id
            session['username'] = user.username
            flash(f'Welcome back, {user.username}!', 'success')
            return redirect(url_for('home'))
        flash('Invalid email or password.', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'error')
            return render_template('register.html')
        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))

# ── Dashboard / Home ───────────────────────────────────────────────────────────
@app.route('/home')
@login_required
def home():
    upload_files = []
    try:
        upload_files = os.listdir(app.config['UPLOAD_FOLDER'])
        upload_files.sort(
            key=lambda x: os.path.getmtime(
                os.path.join(app.config['UPLOAD_FOLDER'], x)), reverse=True)
    except Exception:
        pass
    last_file = upload_files[0] if upload_files else None

    # Stats
    total_preds = Prediction.query.filter_by(user_id=session['user_id']).count()
    placed_preds = Prediction.query.filter_by(
        user_id=session['user_id'], overall='Placed').count()
    placed_pct = round((placed_preds / total_preds * 100) if total_preds else 0, 1)

    return render_template('dashboard.html',
                           last_file=last_file,
                           username=session.get('username', 'User'),
                           total_preds=total_preds,
                           placed_preds=placed_preds,
                           placed_pct=placed_pct)

# ── Feature options (for dropdowns) ───────────────────────────────────────────
@app.route('/get_feature_options')
@login_required
def get_feature_options():
    raw_path = os.path.join(DATA_DIR, 'cleaned_data_cleaned.csv')
    if not os.path.exists(raw_path):
        return jsonify({})
    raw_df = pd.read_csv(raw_path, encoding='latin1')
    options = {}
    for feat in CAT_FEATURES:
        if feat in raw_df.columns:
            options[feat] = sorted(raw_df[feat].dropna().unique().tolist())
    return jsonify(options)

# ── Predict (all 9 models) ─────────────────────────────────────────────────────
@app.route('/predict_user', methods=['POST'])
@login_required
def predict_user():
    int_path = os.path.join(DATA_DIR, 'cleaned_data_int0.csv')
    if not os.path.exists(int_path):
        return jsonify({'error': 'Dataset not found. Please run the EDA script first.'}), 400

    int_df = pd.read_csv(int_path, encoding='latin1')
    maps   = get_encoding_maps()

    # Build user input vector
    user_input = {}
    for feat in FEATURES:
        raw_val = request.form.get(feat, '0')
        if feat in CAT_FEATURES:
            user_input[feat] = maps.get(feat, {}).get(str(raw_val).strip(), 0)
        else:
            try:
                user_input[feat] = float(raw_val)
            except ValueError:
                user_input[feat] = 0.0

    X_user = pd.DataFrame([user_input], columns=FEATURES)

    # Fit scaler on full int dataset
    X_full = int_df[FEATURES].fillna(0)
    scaler = StandardScaler()
    scaler.fit(X_full)
    X_scaled = scaler.transform(X_user)

    # Predict with each model
    y_full = int_df['placedornot'] if 'placedornot' in int_df.columns else None
    results = {}
    for name in MODEL_FILES:
        try:
            model = load_model(name, X_full, y_full, scaler)
            if model is None:
                continue
            pred  = model.predict(X_scaled)
            val   = pred[0] if not isinstance(pred, list) else pred[0]
            results[name] = 'Placed' if int(val) == 1 else 'Not Placed'
        except Exception as e:
            results[name] = f'Error: {str(e)[:60]}'

    placed_count     = sum(1 for r in results.values() if r == 'Placed')
    not_placed_count = sum(1 for r in results.values() if r == 'Not Placed')
    overall          = 'Placed' if placed_count >= not_placed_count else 'Not Placed'

    # Save to history
    try:
        pred_record = Prediction(
            user_id = session['user_id'],
            inputs  = json.dumps(request.form.to_dict()),
            results = json.dumps(results),
            overall = overall)
        db.session.add(pred_record)
        db.session.commit()
    except Exception:
        pass

    return jsonify({
        'results':          results,
        'overall':          overall,
        'placed_count':     placed_count,
        'not_placed_count': not_placed_count,
        'total_models':     len(results)
    })

# ── History ────────────────────────────────────────────────────────────────────
@app.route('/history')
@login_required
def history():
    records = Prediction.query.filter_by(
        user_id=session['user_id']).order_by(
        Prediction.timestamp.desc()).all()
    history_data = []
    for r in records:
        history_data.append({
            'id':        r.id,
            'timestamp': r.timestamp.strftime('%d %b %Y, %H:%M'),
            'overall':   r.overall,
            'results':   json.loads(r.results),
            'inputs':    json.loads(r.inputs),
        })
    return render_template('history.html',
                           history=history_data,
                           username=session.get('username', 'User'))

# ── Dataset upload ─────────────────────────────────────────────────────────────
@app.route('/upload_dataset', methods=['POST'])
@login_required
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename  = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    df = pd.read_csv(file_path, encoding='latin1')
    df.dropna(inplace=True)
    def clean_string(v):
        if isinstance(v, str):
            return ' '.join(w.capitalize() for w in v.split()).strip()
        return v
    df = df.apply(lambda col: col.map(clean_string) if col.dtype == object else col)
    cleaned_path = os.path.join(app.config['PREPROCESSED_FOLDER'], 'cleaned_' + filename)
    df.to_csv(cleaned_path, index=False)
    return jsonify({'message': 'File uploaded and cleaned successfully'}), 200

# ── Convert to int ─────────────────────────────────────────────────────────────
unique_values_dict = {}

@app.route('/convert_to_int', methods=['POST'])
@login_required
def convert_to_int():
    cleaned_files = os.listdir(app.config['PREPROCESSED_FOLDER'])
    if not cleaned_files:
        return jsonify({'error': 'No cleaned datasets found'}), 400
    latest = max(cleaned_files,
                 key=lambda x: os.path.getmtime(
                     os.path.join(app.config['PREPROCESSED_FOLDER'], x)))
    cleaned_path = os.path.join(app.config['PREPROCESSED_FOLDER'], latest)
    df = pd.read_csv(cleaned_path)
    converted = {}
    for col in df.columns:
        if col == 'cand_name' or df[col].dtype == 'int64':
            continue
        if df[col].dtype == 'float64':
            df[col] = df[col].fillna(0).astype(int)
        elif df[col].dtype == 'object':
            try:
                # fill blanks first, otherwise sorting mixed text/NaN fails and
                # the column would be left as text (breaking model training)
                series = df[col].fillna('Unknown').astype(str).str.strip()
                uv = sorted(series.unique())
                converted[col] = dict(zip(uv, range(len(uv))))
                df[col] = series.map(converted[col]).astype(int)
                unique_values_dict[col] = uv
            except Exception as exc:
                print(f'[convert_to_int] skipped column "{col}": {exc}')
    int_path = os.path.join(app.config['INT_DATASET_FOLDER'], 'int_' + latest)
    df.to_csv(int_path, index=False)
    for col, vals in converted.items():
        csv_path = os.path.join(app.config['TRANSFORMED_DATA_FOLDER'], f'{col}_details.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            for v, iv in sorted(vals.items()):
                w.writerow([v, iv])
    return jsonify({'message': 'Dataset converted to int values and saved successfully'}), 200

# ── EDA routes ─────────────────────────────────────────────────────────────────
def encode_img(plt_obj):
    buf = BytesIO()
    plt_obj.savefig(buf, format='png', bbox_inches='tight')
    plt_obj.close()
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode()

def latest_cleaned():
    files = os.listdir(app.config['PREPROCESSED_FOLDER'])
    if not files:
        return None, None
    f = max(files, key=lambda x: os.path.getmtime(
        os.path.join(app.config['PREPROCESSED_FOLDER'], x)))
    return f, os.path.join(app.config['PREPROCESSED_FOLDER'], f)

@app.route('/eda')
@login_required
def eda():
    _, path = latest_cleaned()
    if not path:
        flash('No cleaned datasets found', 'error')
        return redirect(url_for('home'))
    df = pd.read_csv(path)
    eda_summary = {
        'columns':       df.columns.tolist(),
        'data_types':    [str(t) for t in df.dtypes.tolist()],
        'missing_values': df.isnull().sum().tolist(),
        'unique_values': [df[c].nunique() for c in df.columns],
    }
    return render_template('eda.html', eda_results=[eda_summary],
                           username=session.get('username', 'User'))

@app.route('/visualize_placement_status', endpoint='visualize_placement_status')
@login_required
def visualize_placement_status_route():
    _, path = latest_cleaned()
    if not path:
        flash('No cleaned datasets found', 'error')
        return redirect(url_for('home'))
    df = pd.read_csv(path)
    df['placement_status'] = df['placement_status'].astype('category')
    plt.figure(figsize=(10, 6))
    sns.countplot(x='placement_status', data=df, palette='viridis')
    plt.title('Placement Status Distribution')
    plt.xlabel('Placement Status'); plt.ylabel('Count')
    plt.tight_layout()
    img = encode_img(plt)
    return render_template('visualization.html', image_src=img, histograms=[],
                           pairplot_images=[], title='Placement Status Distribution',
                           username=session.get('username', 'User'))

@app.route('/visualize_gender_distribution', endpoint='visualize_gender_distribution')
@login_required
def visualize_gender_distribution():
    _, path = latest_cleaned()
    if not path:
        flash('No cleaned datasets found', 'error')
        return redirect(url_for('home'))
    df = pd.read_csv(path)
    gc = df['gender'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(gc.values, labels=['Male', 'Female'],
            autopct='%1.1f%%', colors=['lightcoral', 'skyblue'], shadow=True)
    plt.title('Gender Distribution')
    img = encode_img(plt)
    return render_template('visualization.html', image_src=img, histograms=[],
                           pairplot_images=[], title='Gender Distribution',
                           username=session.get('username', 'User'))

@app.route('/visualize_histograms')
@login_required
def visualize_histograms():
    _, path = latest_cleaned()
    if not path:
        flash('No cleaned datasets found', 'error')
        return redirect(url_for('home'))
    df = pd.read_csv(path)
    features = ['study_field', 'course', 'branch', 'cohabit', 'gender', 'scholarship',
                'hsc_branch', 'caste', 'weekly_studytime', 'backlog', 'extra_classes',
                'attendance', 'degree_result', 'hsc_result', 'hsc_medium',
                'communication_rating', 'extracurricular', 'certification', 'no_of_internship']
    placed   = df[df['placement_status'] == 'Placed']
    unplaced = df[df['placement_status'] == 'Unplaced']
    histograms = []
    for feat in features:
        if feat not in df.columns:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Histogram of {feat}', fontsize=14)
        for ax, data, color, label in [
                (axes[0], placed,   'green', 'Placed'),
                (axes[1], unplaced, 'red',   'Unplaced')]:
            vc = data[feat].value_counts()
            if not vc.empty:
                ax.bar(vc.index.astype(str), vc.values, color=color)
                ax.set_title(label); ax.set_xlabel(feat); ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
                for i, v in enumerate(vc.values):
                    ax.text(i, v, str(v), ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        histograms.append({'feature': feat,
                           'image': base64.b64encode(buf.read()).decode()})
        plt.close()
    return render_template('visualization.html', image_src=None,
                           histograms=histograms, pairplot_images=[],
                           title='Feature Histograms',
                           username=session.get('username', 'User'))

@app.route('/visualize_pairplot')
@login_required
def visualize_pairplot():
    _, path = latest_cleaned()
    if not path:
        flash('No cleaned datasets found', 'error')
        return redirect(url_for('home'))
    df = pd.read_csv(path)
    cols = ['gender', 'degree_result', 'communication_rating', 'scholarship',
            'caste', 'backlog', 'no_of_internship', 'hsc_medium', 'certification']
    used = []
    images = []
    for xc in cols:
        used.append(xc)
        for yc in cols:
            if xc == yc or yc in used:
                continue
            plt.figure(figsize=(10, 8))
            sns.scatterplot(data=df, x=xc, y=yc, hue='placement_status',
                            palette={'Placed': 'green', 'Unplaced': 'red'})
            plt.title(f'{xc} vs {yc}')
            plt.legend(loc='upper right')
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            images.append({'image': base64.b64encode(buf.read()).decode()})
            plt.close()
    return render_template('visualization.html', image_src=None,
                           histograms=[], pairplot_images=images,
                           title='Pairplot Analysis',
                           username=session.get('username', 'User'))

# ── Train & Evaluate ───────────────────────────────────────────────────────────
def _train_models():
    folder = app.config['INT_DATASET_FOLDER']
    int_files = [f for f in os.listdir(folder) if f.lower().endswith('.csv')]
    if not int_files:
        return None, None, None, None
    # use the most recently created int dataset, not the alphabetically last one
    int_path = os.path.join(
        folder, max(int_files, key=lambda x: os.path.getmtime(os.path.join(folder, x))))
    df = pd.read_csv(int_path, encoding='latin1')
    feats = ['course', 'branch', 'cohabit', 'scholarship', 'hsc_branch', 'caste',
             'weekly_studytime', 'backlog', 'extra_classes', 'attendance',
             'degree_result', 'hsc_result', 'hsc_medium', 'communication_rating',
             'certification', 'no_of_internship']
    missing = [c for c in feats + ['placedornot'] if c not in df.columns]
    if missing:
        return None, None, None, None
    # any column left as text (e.g. it still had blanks when converted) is
    # label-encoded here so training never crashes on a string value
    X = df[feats].copy()
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.Categorical(X[col].astype(str).str.strip()).codes
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    y = pd.to_numeric(df['placedornot'], errors='coerce').fillna(0).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    models = {
        'LogisticRegression':       LogisticRegression(),
        'RandomForestClassifier':   RandomForestClassifier(),
        'DecisionTreeClassifier':   DecisionTreeClassifier(),
        'GaussianNB':               GaussianNB(),
        'SVC':                      SVC(),
        'KNeighborsClassifier':     KNeighborsClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
        'KNN+SVM Hybrid':           combine(),
    }
    trained = {}
    for name, m in models.items():
        m.fit(X_tr_s, y_tr)
        trained[name] = m
    folder = app.config['MODEL_FOLDER']
    for name, m in trained.items():
        joblib.dump(m, os.path.join(folder, f'{name.lower()}_model.pkl'))
    return trained, scaler, X_te_s, y_te

@app.route('/train_and_evaluate')
@login_required
def train_and_evaluate():
    trained, scaler, X_te_s, y_te = _train_models()
    if trained is None:
        flash('No int datasets found. Please upload & transform first.', 'error')
        return redirect(url_for('home'))
    reports = {}
    for name, m in trained.items():
        preds = m.predict(X_te_s)
        reports[name] = {
            'accuracy':             round(accuracy_score(y_te, preds) * 100, 2),
            'confusion_matrix':     confusion_matrix(y_te, preds).tolist(),
            'classification_report': classification_report(y_te, preds),
        }
    return render_template('model_results.html', classification_reports=reports,
                           username=session.get('username', 'User'))

if __name__ == '__main__':
    app.run(debug=True)
