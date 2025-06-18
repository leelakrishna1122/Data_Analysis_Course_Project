import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE  
from sklearn.neural_network import MLPClassifier

# Load dataset
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

# Apply SMOTE to balance dataset
smote = SMOTE(sampling_strategy='auto', k_neighbors=2, random_state=42)
X, y = smote.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Different activation functions to compare
activations = ["relu", "tanh", "logistic"]
results = {}

# Train ANNs with different activation functions
plt.figure(figsize=(10, 6))
for activation in activations:
    print(f"\nTraining ANN with activation function: {activation}")

    # Define the model
    model = MLPClassifier(hidden_layer_sizes=(100,), activation=activation, solver='adam', 
                          max_iter=500, random_state=42, verbose=True)

    # Train the model
    history = model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Store results
    results[activation] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=1),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1 Score": f1_score(y_test, y_pred, average='weighted'),
        "Loss Curve": model.loss_curve_
    }

    # Plot the loss curve
    plt.plot(model.loss_curve_, label=f"Activation: {activation}")

# Format the loss curve plot
plt.title("Loss Function vs. Epochs for Different Activation Functions")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

# Print performance metrics
for activation, res in results.items():
    print(f"\n===== ANN Results (Activation: {activation}) =====")
    print(f"✅ Accuracy: {res['Accuracy']:.2f}")
    print(f"📌 Precision: {res['Precision']:.2f}")
    print(f"📌 Recall: {res['Recall']:.2f}")
    print(f"📌 F1 Score: {res['F1 Score']:.2f}")

