import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

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

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define different numbers of hidden nodes
num_features = X.shape[1]
hidden_nodes_list = [1, 2, 3, int(np.sqrt(num_features)), int(num_features / 2)]

# Store results
test_errors = []

# Train and evaluate ANN models with different hidden nodes
for hidden_nodes in hidden_nodes_list:
    print(f"\nTraining ANN with {hidden_nodes} hidden nodes...")

    # Define MLP model
    ann = MLPClassifier(hidden_layer_sizes=(hidden_nodes,), activation='relu', solver='adam',
                        max_iter=500, random_state=42)
    
    # Train model
    ann.fit(X_train, y_train)
    
    # Predict on test data
    y_pred_test = ann.predict(X_test)
    
    # Calculate test error
    test_error = 1 - accuracy_score(y_test, y_pred_test)
    test_errors.append(test_error)

# Plot test errors vs. hidden nodes
plt.figure(figsize=(8, 5))
plt.plot(hidden_nodes_list, test_errors, marker='o', linestyle='--', color='b')
plt.xlabel("Number of Hidden Nodes")
plt.ylabel("Test Set Error (1 - Accuracy)")
plt.title("Effect of Hidden Nodes on ANN Test Error")
plt.xticks(hidden_nodes_list)
plt.grid()
plt.show()
