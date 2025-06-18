import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from sklearn.feature_selection import mutual_info_classif

# Load dataset
df = pd.read_csv("laptops_cleaned.csv")  

# Convert "Price" into meaningful bins
df['Price_Category'] = pd.cut(df['Price'], bins=[0, 30000, 60000, 90000, 120000, np.inf], labels=[0, 1, 2, 3, 4])  
y = df['Price_Category']

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  

# Drop original price column
X = df.drop(columns=['Price', 'Price_Category'])

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Feature selection using mutual information
mi_scores = mutual_info_classif(X, y)
selected_features = np.argsort(mi_scores)[-10:]
X = X[:, selected_features]

# Apply SMOTETomek to balance dataset
smote_tomek = SMOTETomek(random_state=42)
X, y = smote_tomek.fit_resample(X, y)

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naïve Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "ANN": MLPClassifier(max_iter=500, random_state=42)
}

# Hyperparameter tuning using GridSearchCV
param_grid = {
    "Decision Tree": {'max_depth': [5, 10, 15], 'criterion': ['gini', 'entropy']},
    "KNN": {'n_neighbors': [3, 5, 7], 'metric': ['euclidean', 'manhattan']},
    "ANN": {'hidden_layer_sizes': [(50,), (100,), (100, 50)], 'solver': ['adam', 'sgd']}
}

for name, model in models.items():
    if name in param_grid:
        grid_search = GridSearchCV(model, param_grid[name], cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        models[name] = grid_search.best_estimator_

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Compute errors
    train_error = 1 - accuracy_score(y_train, y_pred_train)
    test_error = 1 - accuracy_score(y_test, y_pred_test)

    # Evaluation metrics
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred_test, average='weighted')
    f1 = f1_score(y_test, y_pred_test, average='weighted')
    cm = confusion_matrix(y_test, y_pred_test)

    # Store results
    results[name] = {
        "Train Accuracy": accuracy_train,
        "Test Accuracy": accuracy_test,
        "Train Error": train_error,
        "Test Error": test_error,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Confusion Matrix": cm,
        "Parameters": model.get_params()
    }

# Print results
for name, res in results.items():
    print(f"\n===== {name} Results =====")
    print(f"🔹 Parameters: {res['Parameters']}")
    print(f"✅ Train Accuracy: {res['Train Accuracy']:.2f}")
    print(f"✅ Test Accuracy: {res['Test Accuracy']:.2f}")
    print(f"❌ Train Error: {res['Train Error']:.2f}")
    print(f"❌ Test Error: {res['Test Error']:.2f}")
    print(f"📌 Precision: {res['Precision']:.2f}")
    print(f"📌 Recall: {res['Recall']:.2f}")
    print(f"📌 F1 Score: {res['F1 Score']:.2f}")
    print("\nConfusion Matrix:")
    print(res["Confusion Matrix"])

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(res["Confusion Matrix"], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()