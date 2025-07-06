import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# Train Logistic Regression
def train_logistic(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    print("Logistic Regression training complete.")
    return model

# Train Random Forest
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Random Forest training complete.")
    return model

# Train XGBoost
def train_xgboost(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    print("XGBoost training complete.")
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

# Save model
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}.")

if __name__ == "__main__":
    from data_preprocessing import load_data, remove_duplicates, split_data, balance_data

    data_path = './data/creditcard.csv'
    df = load_data(data_path)
    df = remove_duplicates(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_bal, y_train_bal = balance_data(X_train, y_train)

    # Train models
    lr_model = train_logistic(X_train_bal, y_train_bal)
    rf_model = train_random_forest(X_train_bal, y_train_bal)
    xgb_model = train_xgboost(X_train_bal, y_train_bal)

    # Evaluate models
    print("\n--- Logistic Regression Evaluation ---")
    evaluate_model(lr_model, X_test, y_test)

    print("\n--- Random Forest Evaluation ---")
    evaluate_model(rf_model, X_test, y_test)

    print("\n--- XGBoost Evaluation ---")
    evaluate_model(xgb_model, X_test, y_test)

    # Save best model
    save_model(xgb_model, './models/xgb_model.pkl')