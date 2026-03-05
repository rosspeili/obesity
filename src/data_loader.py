import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath: str, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads raw CSV data and preprocesses it.
    Returns (X_train, X_test, y_train, y_test)
    """
    df = pd.read_csv(filepath)
    
    # 1. Drop Height and Weight as they are not part of the 18 specified predictors
    if 'Height' in df.columns and 'Weight' in df.columns:
        df = df.drop(columns=['Height', 'Weight'])

    # 2. Map binary categorical variables
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['family_history_with_overweight'] = df['family_history_with_overweight'].map({'yes': 1, 'no': 0})
    df['FAVC'] = df['FAVC'].map({'yes': 1, 'no': 0})
    df['SMOKE'] = df['SMOKE'].map({'yes': 1, 'no': 0})
    df['SCC'] = df['SCC'].map({'yes': 1, 'no': 0})

    # 3. Map ordinal categorical variables (0 to 3 scale)
    ordinal_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    df['CAEC'] = df['CAEC'].map(ordinal_map)
    df['CALC'] = df['CALC'].map(ordinal_map)

    # 4. One-hot encode MTRANS
    transport_dummies = pd.get_dummies(df['MTRANS'], dtype=int)
    expected_mtrans = ['Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking']
    for col in expected_mtrans:
        if col not in transport_dummies.columns:
            transport_dummies[col] = 0
            
    df = pd.concat([df.drop(columns=['MTRANS']), transport_dummies[expected_mtrans]], axis=1)

    # 5. Outcome variable `NObeyesdad` mapped to 1 (obese) or 0 (not obese)
    df['NObeyesdad'] = df['NObeyesdad'].apply(lambda x: 1 if 'Obesity' in x else 0)

    X = df.drop(columns=['NObeyesdad'])
    y = df['NObeyesdad']
    
    # 6. Perform Train/Test Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

