import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# This file focuses on Balanced accuracy scoring GridSearch


# Load preprocessed dataset into a pandas DataFrame
df = pd.read_csv("preprocessed_diabetes.csv")

# Split into features (X) and target (y)
X = df.drop("Diabetes_binary", axis=1)
y = df["Diabetes_binary"]

# Train/test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Define parameter grid
param_grid = {
    "n_estimators": [100, 200],  # Number of trees in the forest
    "max_features": ["sqrt", "log2"], # how many features are considered for best split at each node
    "oob_score": [True, False], # use out of bag samples for validation
    "class_weight": [None, "balanced"], #weights for imbalanced class
    "max_depth": [10, 20] 
}

# Create a Random Forest with bootstapping 
rf = RandomForestClassifier(random_state=42, bootstrap=True)

# Balanced accuracy scoring GridSearch
grid_search_bal = GridSearchCV(
    RandomForestClassifier(random_state=42, bootstrap=True),
    param_grid,
    cv=5,
    scoring='balanced_accuracy',
    n_jobs=-1,
    return_train_score=True,
)
grid_search_bal.fit(X_train, y_train)

# Save results to CSV
results_bal = pd.DataFrame(grid_search_bal.cv_results_)
results_bal["param_class_weight"] = results_bal["param_class_weight"].astype(str).replace("None", "None")
results_bal.to_csv("grid2_search_results_balanced_accuracy.csv", index=False)

#Classification Reports for trainging and test
print("\nBalanced Accuracy Model:")
print("Train Report:")
print(classification_report(y_train, grid_search_bal.predict(X_train)))
print("Test Report:")
print(classification_report(y_test, grid_search_bal.predict(X_test)))