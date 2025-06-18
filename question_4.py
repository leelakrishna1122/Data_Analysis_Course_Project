import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# Load original dataset
df = pd.read_csv("laptops_cleaned.csv")

# Convert "Price" into bins (categorization)
df['Price_Category'] = pd.qcut(df['Price'], q=5, labels=[0, 1, 2, 3, 4])  
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

# Apply SMOTE for balancing
smote = SMOTE(sampling_strategy='auto', k_neighbors=2, random_state=42)
X, y = smote.fit_resample(X, y)

# Create datasets D1 (25%), D2 (50%), D3 (75%), and D (100%)
df_full = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])
df_full['Price_Category'] = y

df_d1 = df_full.sample(frac=0.25, random_state=42)
df_d2 = df_full.sample(frac=0.50, random_state=42)
df_d3 = df_full.sample(frac=0.75, random_state=42)

datasets = {"D1": df_d1, "D2": df_d2, "D3": df_d3, "D": df_full}

# Initialize classifiers
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=10, criterion="gini", random_state=42),
    "Naïve Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
    "ANN": MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500, random_state=42)
}

# Train and evaluate models on each dataset
results = {}

for dataset_name, dataset in datasets.items():
    print(f"\n===== Processing {dataset_name} Dataset =====")
    
    X_data = dataset.drop(columns=['Price_Category'])
    y_data = dataset['Price_Category']
    
    # Split data into train-test (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    
    dataset_results = {}

    for model_name, model in models.items():
        print(f"Training {model_name} on {dataset_name}...")

        # Train model
        model.fit(X_train, y_train)

        # Predictions
        y_pred_test = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=1)
        recall = recall_score(y_test, y_pred_test, average='weighted')
        f1 = f1_score(y_test, y_pred_test, average='weighted')

        # Store results
        dataset_results[model_name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }

    results[dataset_name] = dataset_results

# Print comparison results
for dataset_name, dataset_results in results.items():
    print(f"\n===== {dataset_name} Results =====")
    for model_name, metrics in dataset_results.items():
        print(f"\n🔹 {model_name}")
        print(f"✅ Accuracy: {metrics['Accuracy']:.2f}")
        print(f"📌 Precision: {metrics['Precision']:.2f}")
        print(f"📌 Recall: {metrics['Recall']:.2f}")
        print(f"📌 F1 Score: {metrics['F1 Score']:.2f}")

# Compare Accuracy of Different Datasets
accuracy_values = {dataset_name: [metrics["Accuracy"] for model, metrics in dataset_results.items()]
                   for dataset_name, dataset_results in results.items()}

labels = list(models.keys())  # Classifier names
x = np.arange(len(labels))  # X-axis positions

plt.figure(figsize=(10, 6))

for i, (dataset_name, acc_values) in enumerate(accuracy_values.items()):
    plt.bar(x + i * 0.2, acc_values, width=0.2, label=dataset_name)

plt.xlabel("Classifiers")
plt.ylabel("Accuracy")
plt.title("Classifier Accuracy Across Different Dataset Sizes")
plt.xticks(x + 0.3, labels)
plt.legend()
plt.show()
