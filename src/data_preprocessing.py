import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    print(f"Data loaded with shape: {df.shape}")
    return df

# Remove duplicates
def remove_duplicates(df):
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Removed {duplicates} duplicate records.")
    else:
        print("No duplicate records found.")
    return df

# Handle class imbalance using SMOTE
def balance_data(X, y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"Balanced dataset shape: {X_res.shape}, Positive class count: {sum(y_res)}")
    return X_res, y_res

# Split dataset
def split_data(df):
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    data_path = './data/creditcard.csv'  # Adjust path as needed
    df = load_data(data_path)
    df = remove_duplicates(df)

    # Further preprocessing steps can be added here

    X_train, X_test, y_train, y_test = split_data(df)

    # Handle imbalance on training data only
    X_train_bal, y_train_bal = balance_data(X_train, y_train)
    print(X_train_bal.columns)