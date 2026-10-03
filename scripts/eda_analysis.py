import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

warnings.filterwarnings("ignore", message='X does not have valid feature names')
warnings.filterwarnings("ignore",
                        message='Precision is ill-defined and being set to 0.0 in labels with no predicted samples')


def unique_labels(*args):
    """Extract an ordered array of unique labels"""
    return np.unique(np.hstack(args))


class SVMKNNHybridClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.classes_ = None
        self.svm = SVC()
        self.knn = KNeighborsClassifier()
        self.hybrid_model = VotingClassifier(estimators=[('svm', self.svm), ('knn', self.knn)], voting='hard')

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Fit the models
        self.svm.fit(X, y)
        self.knn.fit(X, y)
        self.hybrid_model.fit(X, y)

        # Return the classifier
        return self

    # noinspection PyTypeChecker
    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Predictions from SVM and KNN
        svm_pred = self.svm.predict(X)
        knn_pred = self.knn.predict(X)

        # Hybrid prediction (simple ensemble)
        hybrid_pred = []
        for i in range(len(X)):
            if svm_pred[i] == knn_pred[i]:
                hybrid_pred.append(svm_pred[i])
            else:
                hybrid_pred.append(self.hybrid_model.predict(X[i].reshape(1, -1))[0])

        return hybrid_pred


# Define 'features' globally
features = ['backlog', 'course', 'branch', 'cohabit',
            'scholarship', 'hsc_branch', 'caste', 'weekly_studytime', 'extra_classes', 'attendance',
            'degree_result', 'hsc_result', 'hsc_medium', 'communication_rating', 'certification',
            'no_of_internship']


def display_data_info():
    print("Basic Information about the dataset:")
    print(df.info())

    print("\nFirst few rows of the dataset:")
    print(df.head())

    print("\nSummary statistics of numeric columns:")
    print(df.describe())


def visualize_placement_status():
    plt.figure(figsize=(10, 6))
    df['placement_status'] = df['placement_status'].astype('category')
    sns.countplot(x='placement_status', data=df, hue='placement_status', palette='viridis')
    plt.title('Placement Status Distribution')
    plt.xlabel('Placement Status')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Placed', 'Unplaced'])
    plt.show()


# noinspection PyUnusedLocal
def visualize_gender_distribution():
    gender_counts = df['gender'].value_counts()
    labels = gender_counts.index
    sizes = gender_counts.values
    # Rearrange the labels list
    labels = ['Male', 'Female']
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['lightcoral', 'skyblue'], shadow=True)
    plt.title('Gender Distribution')
    plt.show()


# noinspection PyShadowingNames
def visualize_correlation_matrix():
    # Drop specified columns from the DataFrame
    columns_to_drop = ['sr_no','cand_name', 'degree', 'placement_status', 'passout_year',
                       'college_name', 'reason_college', 'company_name',
                       'campus_placement', 'salary']
    df_filtered = df.drop(columns=columns_to_drop)

    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_filtered.corr(), annot=True, cmap='RdBu_r', fmt=".2f")
    plt.title('Correlation Matrix (Excluding Specified Columns)')
    plt.show()

    print("Columns with positive and negative impact (correlation >= -0.5):")
    # Initialize PrettyTable
    table = PrettyTable()
    table.field_names = ["Column", "Correlation"]

    for col in df_filtered.columns:
        correlation = df_filtered[col].corr(df['placement_status'])
        if correlation >= -0.5:
            table.add_row([f"{col}", f"{correlation:.2f}"])

    print(table)
    print("Columns which might impact on results")
    for col in df_filtered.columns:
        correlation = df_filtered[col].corr(df['placement_status'])
        if abs(correlation) >= -0.5:
            table.add_row([f"{col}", f"{correlation:.2f}"])

    print(table)


# noinspection PyShadowingNames
def visualize_histograms():
    file_path = r"Data\cleaned_data_cleaned.csv"
    df = pd.read_csv(file_path, encoding='latin1')

    # Features to analyze
    features = ['study_field', 'course', 'branch', 'cohabit', 'gender', 'scholarship', 'hsc_branch',
                'caste', 'weekly_studytime', 'backlog',
                'extra_classes', 'attendance', 'degree_result', 'hsc_result', 'hsc_medium',
                'communication_rating', 'extracurricular',
                'certification', 'no_of_internship']

    # Separate data for placed and unplaced students
    placed_data = df[df['placement_status'] == 'Placed']
    unplaced_data = df[df['placement_status'] == 'Unplaced']

    # Plot graphs for each feature
    for feature in features:
        plt.figure(figsize=(10, 6))
        plt.suptitle(f'Histogram of {feature}', fontsize=16)

        # Placed students
        plt.subplot(1, 2, 1)
        plt.title('Placed')
        value_counts_placed = placed_data[feature].value_counts()
        if not value_counts_placed.empty:
            plt.bar(value_counts_placed.index.astype(str), value_counts_placed.values, color='green')
            plt.xlabel(feature)
            plt.ylabel('Count')
            plt.xticks(rotation=90)  # Rotate x-axis labels

            # Write the value of each bar at its top
            for i, v in enumerate(value_counts_placed.values):
                plt.text(i, v, str(v), ha='center', va='bottom')
            plt.tight_layout()
        else:
            plt.text(0.5, 0.5, 'No data for placed students', horizontalalignment='center',
                     verticalalignment='center',
                     transform=plt.gca().transAxes)

        # Unplaced students
        plt.subplot(1, 2, 2)
        plt.title('Unplaced')
        value_counts_unplaced = unplaced_data[feature].value_counts()
        if not value_counts_unplaced.empty:
            plt.bar(value_counts_unplaced.index.astype(str), value_counts_unplaced.values, color='red')
            plt.xlabel(feature)
            plt.ylabel('Count')
            plt.xticks(rotation=90)  # Rotate x-axis labels

            # Write the value of each bar at its top
            for i, v in enumerate(value_counts_unplaced.values):
                plt.text(i, v, str(v), ha='center', va='bottom')

            plt.tight_layout()
        else:
            plt.text(0.5, 0.5, 'No data for unplaced students', horizontalalignment='center',
                     verticalalignment='center', transform=plt.gca().transAxes)

        plt.show()


def visualize_pairplots():
    pairplot_columns = ['backlog', 'gender', 'degree_result', 'communication_rating', 'scholarship',
                        'caste',
                        'no_of_internship', 'hsc_medium', 'certification']

    used_columns = []  # Store the columns already used

    for x_column in pairplot_columns:
        # Add the x_column to the list of used columns
        used_columns.append(x_column)

        for y_column in pairplot_columns:
            # Skip if x_column is the same as y_column or y_column is in used_columns
            if x_column == y_column or y_column in used_columns:
                continue

            plt.figure(figsize=(10, 8))
            sns.scatterplot(data=df, x=x_column, y=y_column, hue='placement_status',
                            palette={0: "green", 1: "red"})
            plt.title(f'Scatter plot of {x_column} vs {y_column} with Placement Status')
            handles, labels = plt.gca().get_legend_handles_labels()
            # Move the legend box to a position outside the plot area
            plt.legend(handles, ['Placed', 'Unplaced'], title='Placement Status', loc='upper center',
                       bbox_to_anchor=(1, 1.1), shadow=True, ncol=2)

            plt.show()


def user_input():
    scaler = None  # Initialize scaler variable
    while True:
        print("\n1. Display basic information, first rows, and summary statistics of the dataset")
        print("2. Visualize the distribution of placement status with a bar plot")
        print("3. Visualize the distribution of gender using a pie chart")
        print("4. Visualize the correlation matrix with a heatmap")
        print("5. Visualize histograms for numeric columns")
        print("6. Visualize pairplots for numeric columns")
        print("7. Train classification models and evaluate")
        print("8. Deploy Model")
        print("9. Predict Placement Status")
        print("10. Exit")

        try:
            choice = int(input("Enter your choice (1-10): "))
            if choice == 1:
                display_data_info()
            elif choice == 2:
                visualize_placement_status()
            elif choice == 3:
                visualize_gender_distribution()
            elif choice == 4:
                visualize_correlation_matrix()
            elif choice == 5:
                visualize_histograms()
            elif choice == 6:
                visualize_pairplots()
            elif choice == 7:
                trained_models, scaler = train_and_evaluate_models()  # Update to capture both trained models and scaler
            elif choice == 8:
                if scaler is None:
                    print("Please train the models first.")
                else:
                    deploy_model()
            elif choice == 9:
                if scaler is None:
                    print("Please train the models first.")
                else:
                    predict_placement_status(scaler)  # Pass scaler to predict_placement_status
            elif choice == 10:
                print("Good Bye!!! Exiting...")
                break
            else:
                print("Invalid input! Please enter a number between 1 and 10.")
        except ValueError:
            print("Invalid input! Please enter a number.")


def load_models():
    # Load all trained models from pickle files
    models = {}
    # noinspection SpellCheckingInspection
    model_files = [
        'logisticregression_model.pkl',
        'randomforestclassifier_model.pkl',
        'decisiontreeclassifier_model.pkl',
        'gaussiannb_model.pkl',
        'svc_model.pkl',
        'kneighborsclassifier_model.pkl',
        'gradientboostingclassifier_model.pkl',
        'lineardiscriminantanalysis_model.pkl',
        'svmknnhybridclassifier_model.pkl'  # Add the hybrid classifier pickle file here
    ]
    for model_file in model_files:
        model_name = model_file.split('_model.pkl')[0]
        models[model_name] = joblib.load(model_file)
    return models


# noinspection PyUnusedLocal,PyShadowingNames
def train_and_evaluate_models():
    # Assuming 'placement_status' is the target variable and others are features
    features = ['backlog', 'course', 'branch', 'cohabit',
                'scholarship', 'hsc_branch', 'caste', 'weekly_studytime', 'extra_classes', 'attendance',
                'degree_result', 'hsc_result', 'hsc_medium', 'communication_rating', 'certification',
                'no_of_internship']
    X = df[features]
    y = df['placedornot']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Convert categorical columns to dummy variables if needed
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    # Standardize features if needed (for some algorithms)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize dictionary to store models
    trained_models = {}

    # Logistic Regression
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train_scaled, y_train)
    logistic_pred = logistic_model.predict(X_test_scaled)
    trained_models['LogisticRegression'] = logistic_model

    # Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    trained_models['RandomForestClassifier'] = rf_model

    # Decision Tree
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    trained_models['DecisionTreeClassifier'] = dt_model

    # Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    trained_models['GaussianNB'] = nb_model

    # Support Vector Machine (SVM)
    svm_model = SVC()
    svm_model.fit(X_train_scaled, y_train)
    svm_pred = svm_model.predict(X_test_scaled)
    trained_models['SVC'] = svm_model

    # K-Nearest Neighbors (KNN)
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train_scaled, y_train)
    knn_pred = knn_model.predict(X_test_scaled)
    trained_models['KNeighborsClassifier'] = knn_model

    # Gradient Boosting
    gb_model = GradientBoostingClassifier()
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    trained_models['GradientBoostingClassifier'] = gb_model

    # Linear Discriminant Analysis (LDA)
    lda_model = LinearDiscriminantAnalysis()
    lda_model.fit(X_train_scaled, y_train)
    lda_pred = lda_model.predict(X_test_scaled)
    trained_models['LinearDiscriminantAnalysis'] = lda_model

    # Hybrid Classifier (SVM and KNN)
    hybrid_model = SVMKNNHybridClassifier()
    hybrid_model.fit(X_train_scaled, y_train)
    hybrid_pred = hybrid_model.predict(X_test_scaled)
    trained_models['SVMKNNHybridClassifier'] = hybrid_model

    # Evaluate models
    # noinspection PyShadowingNames
    def evaluate_model(model, predictions):
        accuracy = accuracy_score(y_test, predictions)
        print(f"\nAccuracy for {type(model).__name__}: {accuracy:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, predictions)  # Use predictions instead of y_pred
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
        plt.title(f'Confusion Matrix - {type(model).__name__}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        print("Classification Report:")
        print(classification_report(y_test, predictions))

    # Save trained models to pickle files
    for model_name, model in trained_models.items():
        model_filename = f'{model_name.lower()}_model.pkl'
        joblib.dump(model, model_filename)
        print(f"Model trained and saved as {model_filename}")

        # Evaluate each model
        evaluate_model(model, model.predict(X_test_scaled))

    return trained_models, scaler  # Return both trained models and scaler object


# noinspection PyShadowingNames
def predict_placement_status(scaler):
    # Load models
    models = load_models()

    print("Please select how you want to provide the data:")
    print("1. Enter data manually")
    print("2. Upload a CSV file")
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        # Input new dataset values
        new_data = {}
        for feature in features:
            while True:
                try:
                    value = float(input(f"Enter value for {feature}: "))
                    break  # Exit the loop if input is successfully converted to float
                except ValueError:
                    print("Invalid input! Please enter a number.")
            new_data[feature] = [value]

        # Create DataFrame from new data
        new_df = pd.DataFrame(new_data)

    elif choice == '2':
        print("Please prepare a file containing the following features in CSV or Excel format:")
        print(", ".join(features))
        print("Make sure the file has the same columns as mentioned above.")

        while True:
            file_path = input("Enter the file path of your CSV or Excel file: ")
            file_extension = file_path.split('.')[-1]  # Extract file extension
            if file_extension.lower() == 'csv':
                try:
                    new_df = pd.read_csv(file_path)
                    break
                except FileNotFoundError:
                    print("File not found. Please enter a valid file path.")
            elif file_extension.lower() in ['xlsx', 'xls']:
                try:
                    new_df = pd.read_excel(file_path)
                    break
                except FileNotFoundError:
                    print("File not found. Please enter a valid file path.")
            else:
                print("Unsupported file format. Please provide a CSV or Excel file.")

    else:
        print("Invalid choice! Please select 1 or 2.")

    # Convert categorical columns to dummy variables if needed
    # noinspection PyUnboundLocalVariable
    new_df = pd.get_dummies(new_df)

    # Standardize features using the provided scaler
    new_df_scaled = scaler.transform(new_df)

    # Initialize PrettyTable
    table = PrettyTable()
    table.field_names = ["Model", "Prediction"]

    # Predict placement status using each model for all rows
    for model_name, model in models.items():
        if model_name == 'SVMKNNHybridClassifier':  # Special handling for hybrid classifier
            predictions = model.predict(new_df_scaled)
        else:
            predictions = model.predict(new_df_scaled)
        prediction_labels = ['Placed' if pred == 1 else 'Unplaced' for pred in predictions]
        table.add_row([model_name, ", ".join(prediction_labels)])

    # Print the table
    print("Predictions using different models:")
    print(table)


# noinspection PyShadowingNames
def deploy_model():
    # Assuming 'placement_status' is the target variable and others are features
    features = ['backlog', 'course', 'branch', 'cohabit',
                'scholarship', 'hsc_branch', 'caste', 'weekly_studytime', 'extra_classes', 'attendance',
                'degree_result', 'hsc_result', 'hsc_medium', 'communication_rating', 'certification',
                'no_of_internship']
    X = df[features]
    y = df['placedornot']

    # Convert categorical columns to dummy variables if needed
    X = pd.get_dummies(X)

    # Standardize features if needed (for some algorithms)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the selected model
    selected_model = RandomForestClassifier()  # You can replace this with the model of your choice
    selected_model.fit(X_scaled, y)

    # Save the trained model
    model_filename = 'placement_prediction_model.pkl'
    joblib.dump(selected_model, model_filename)
    print(f"Model trained and saved as {model_filename}")

# Load the dataset and convert column names to lowercase
file_path = r"Data\cleaned_data_cleaned.csv"
df = pd.read_csv(file_path, encoding='latin1')

# Iterate through each column for preprocessing
# noinspection GrazieInspection
for col in df.columns:
    # Check if the column is 'cand_name'
    if col == 'cand_name':
        continue  # Skip 'cand_name' column
    # Check if the column dtype is string/object (handles both legacy 'object' and newer pandas StringDtype)
    if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
        try:
            # Convert the column to int via factorize
            df[col] = pd.factorize(df[col])[0]
            df[col] = df[col].astype(int)
        except ValueError:
            # Handle non-numeric or missing values
            print(f"Cannot convert column {col} to int.")
    elif df[col].dtype == 'float64' or str(df[col].dtype) == 'Float64':
        # Convert float64 columns to int
        df[col] = df[col].fillna(0).astype(int)

# Convert all columns to int64 dtype except 'cand_name'
for col in df.columns:
    if col != 'cand_name':
        try:
            df[col] = df[col].astype('int64')
        except (ValueError, TypeError):
            print(f"Skipping int64 conversion for column: {col}")


def custom_round(value):
    if isinstance(value, float):
        if value >= 0.5:
            return round(value + 0.1)
        else:
            return round(value - 0.1)
    else:
        return value


decimal_columns = df.select_dtypes(include=['float64']).columns
df[decimal_columns] = df[decimal_columns].apply(custom_round)

# Define the file path for the new CSV file
new_csv_file_path = r"Data\cleaned_data_int0.csv"

# Save the DataFrame to a new CSV file
df.to_csv(new_csv_file_path, index=False)
print(f"Data has been saved to {new_csv_file_path}")

# Load the dataset again for further processing
file_path = r"Data\cleaned_data_int0.csv"
df = pd.read_csv(file_path, encoding='latin1')
# Call the user input function
user_input()
