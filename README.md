##  Data Cleaning Process Using OpenRefine

### 1. Project Initialization
- Imported the raw dataset into OpenRefine.
- Initiated the data cleaning workflow.

---

### 2. Handling Missing and Unnecessary Data
- Removed **30 blank rows** to ensure data consistency.
- Deleted the **"Unnamed: 0"** column as it was redundant.

---

### 3.  Standardizing Text Fields
- **Trimmed extra spaces** in the `Company` and `TypeName` columns using GREL (`value.trim()`).
- Performed **mass edits** in:
  - `TypeName` (e.g., standardizing variations like "Notebook", "Ultrabook", etc.)
  - `ScreenResolution` (e.g., fixing inconsistent formats)
- Used GREL to **remove redundant terms** from `ScreenResolution`, such as:
  - `"Full HD"`, `"Touchscreen"`, `"4K Ultra HD"`, `"IPS Panel"`, etc.

---

### 4. Cleaning and Formatting CPU Data
- Standardized CPU names (e.g., removed duplicate prefixes like `"A10-"`, `"A9-"`).
- Normalized case formatting:
  - `"x5"` ➝ `"X5"`, `"HZ"` ➝ `"Hz"`, `"m"` ➝ `"M"`, `"v"` ➝ `"V"`.
- Used **mass edits** to correct inconsistent CPU values.

---

### 5. Formatting Numerical Columns
- Converted the following columns to **numeric format**:
  - `Weight`, `Price`, `Inches`
- Applied GREL to:
  - Remove `"kg"` from `Weight`, then round to **2 decimal places**.
  - Replace `"gb"` with `"GB"` and `"flash storage"` with `"SSD"` in memory data.
- Edited specific incorrect entries in memory-related fields.

---

### 6. Handling Duplicates and Clustering
- Created a new column: `Cluster Key` by concatenating:
  - `Company`, `TypeName`, `Inches`, `ScreenResolution`, `Cpu`, `Ram`, `Memory`
- Used **mass edits** to normalize `Cluster Key` values.
- Removed `Cluster Key` after duplicate resolution.

---

### 7.  Final Adjustments and Renaming
- Renamed `Weight` ➝ `Weight(kg)` for clarity.
- Rounded values in `Price` to whole numbers.
- Performed final **mass edits and GREL transformations** for consistency.

---

### 8. Exporting Cleaned Data
- Exported the cleaned dataset as: `laptops_cleaned.csv`.





##### Data Splitting into train and test
## Method Used
The dataset was split into training and test sets using **scikit-learn's `train_test_split` function** from `sklearn.model_selection`.

## Splitting Details
   Splitting Ratio: 80% training, 20% testing (`test_size=0.2`)
   Randomization: Controlled using `random_state=42` to ensure reproducibility

## Code Implementation
```python
from sklearn.model_selection import train_test_split

# Assuming df is the cleaned dataset
df: Your cleaned dataset (after OpenRefine processing)

# Splitting the dataset into training and test sets
train, test = train_test_split(df, test_size=0.2, random_state=42)
```

### Explanation of Parameters
- `df`: The cleaned dataset after processing in OpenRefine
- `test_size=0.2`: 20% of the data is allocated to the test set, while 80% is retained for training
- `random_state=42`: Ensures that the train-test split is consistent and reproducible across multiple runs



### QUESTION 1 
# Machine Learning Model Training on Laptop Price Dataset

## Overview
This project applies multiple machine learning models to predict laptop price categories based on various features. It involves data preprocessing, feature selection, dataset balancing, training classifiers, hyperparameter tuning, and evaluating model performance.

## Dependencies
The following Python libraries are used in this project:
- `pandas`: For data manipulation
- `numpy`: For numerical operations
- `matplotlib.pyplot` & `seaborn`: For visualization
- `sklearn`: For machine learning tasks (preprocessing, model training, evaluation)
- `imblearn`: For handling imbalanced datasets
- `xgboost`: For XGBoost classifier

## Workflow

### 1. Load Dataset
The dataset (`laptops_cleaned.csv`) is loaded using pandas:
```python
 df = pd.read_csv("laptops_cleaned.csv")
```

### 2. Categorizing Price
The continuous `Price` column is converted into categorical bins:
```python
df['Price_Category'] = pd.cut(df['Price'], bins=[0, 30000, 60000, 90000, 120000, np.inf], labels=[0, 1, 2, 3, 4])
y = df['Price_Category']
```

### 3. Encoding Categorical Variables
Label encoding is applied to categorical columns:
```python
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  
```

### 4. Feature Selection
- Drop `Price` and `Price_Category` columns.
- Standardize features using `StandardScaler`.
- Select top 10 features using mutual information scores:
```python
scaler = StandardScaler()
X = scaler.fit_transform(X)
mi_scores = mutual_info_classif(X, y)
selected_features = np.argsort(mi_scores)[-10:]
X = X[:, selected_features]
```

### 5. Handling Imbalanced Data
The dataset is balanced using SMOTETomek:
```python
smote_tomek = SMOTETomek(random_state=42)
X, y = smote_tomek.fit_resample(X, y)
```

### 6. Splitting Data
Splitting the dataset into training (80%) and testing (20%):
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 7. Model Training
The following models are initialized:
- Decision Tree
- Naïve Bayes
- K-Nearest Neighbors (KNN)
- Artificial Neural Network (ANN)

```python
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naïve Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "ANN": MLPClassifier(max_iter=500, random_state=42)
}
```

### 8. Hyperparameter Tuning
GridSearchCV is used to tune hyperparameters for Decision Tree, KNN, and ANN:
```python
param_grid = {
    "Decision Tree": {'max_depth': [5, 10, 15], 'criterion': ['gini', 'entropy']},
    "KNN": {'n_neighbors': [3, 5, 7], 'metric': ['euclidean', 'manhattan']},
    "ANN": {'hidden_layer_sizes': [(50,), (100,), (100, 50)], 'solver': ['adam', 'sgd']}
}
```
The best model is selected based on GridSearchCV results.

### 9. Model Evaluation
Each model is trained and evaluated using accuracy, precision, recall, F1-score, and a confusion matrix:
```python
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred_test, average='weighted')
f1 = f1_score(y_test, y_pred_test, average='weighted')
cm = confusion_matrix(y_test, y_pred_test)
```

### 10. Results Visualization
Each model's confusion matrix is displayed using a heatmap:
```python
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix for {name}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

## Conclusion
This project successfully classifies laptop prices using machine learning models. The best-performing model can be identified based on evaluation metrics.




#### QUESTION 2
# Workflow

1. Importing Required Libraries
The script imports necessary libraries for data handling, preprocessing, model training, and evaluation.

2. Loading the Dataset
Reads the laptops_cleaned.csv file into a Pandas DataFrame.
Converts the "Price" column into five price categories using quantile-based discretization (qcut).
Defines y as the target variable (Price_Category).

3. Splitting Data into Training and Testing Sets
Splits the dataset into 80% training and 20% testing using train_test_split.

4. Training ANN Models with Different Activation Functions
Defines three activation functions: ReLU, Tanh, and Logistic.
Trains separate Multi-Layer Perceptron Classifiers (MLPClassifier) for each activation function with:
Hidden Layer Size: (100,)
Solver: adam
Maximum Iterations: 500
Random State: 42
Stores evaluation metrics for each model.

5. Evaluating Model Performance
Predictions: Generates predictions on the test set.
Metrics Computed:
Accuracy: Overall correctness of predictions.
Precision: Fraction of correctly identified price categories.
Recall: Model’s ability to identify all instances of each class.
F1 Score: Harmonic mean of precision and recall.
Loss Curve Visualization:
Plots training loss over epochs for each activation function to analyze convergence behavior.

# Dependencies

pandas,numpy,matplotlib,seaborn,scikit-learn,imbalanced-learn



#### QUESTION 3
just running the code and you will get the comparision graph


### QUESTION 4
# workflow

6. Creating Dataset Variants (D1, D2, D3, D)
D1 (25% of data)
D2 (50% of data)
D3 (75% of data)
D (100% of data)
Each dataset is randomly sampled from the full dataset.

7. Splitting Data into Training and Testing Sets
Splits each dataset into 80% training and 20% testing using train_test_split.

8. Training and Evaluating Multiple Models
The following models are trained and evaluated on each dataset:
Decision Tree (max depth = 10, Gini criterion)
Naïve Bayes (GaussianNB)
KNN (k=5, Minkowski distance, p=2)
ANN (MLPClassifier with 100 hidden neurons, ReLU activation, Adam optimizer, max 500 iterations)
For each model, the following metrics are calculated:
Accuracy
Precision (weighted average)
Recall
F1 Score

9. Evaluating and Comparing Results
The classification performance of each model is printed for each dataset.
A bar chart comparison is generated to visualize the accuracy of different models across dataset sizes.





