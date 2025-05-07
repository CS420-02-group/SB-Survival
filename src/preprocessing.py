# preprocessing.py
# handles data prepossessing for buisness survial prediction models

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath="../processed/filtered_bds_data.csv", test_size=0.3, random_state=42):
    """
    load and preprosess the buisness data for modeling
    
    parameters:
    -----------
    filepath : str
        path to the processed data file
    test_size : float
        propotion of data to use for testing
    random_state : int
        random seed for reproduicibility
        
    returns:
    --------
    X_train, X_test, y_train, y_test, feature_names : tuple
        preprocessed training and testing data with feature names
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"error: file not fount at {filepath}")
        return None, None, None, None, None
    
    # check wich dataset we're working with based on columns
    if "dataclass_name" in df.columns:
        # working with filtered bds data
        print(f"preprocessing filtered bds data from {filepath}")
        
        # create target varible
        df["label"] = df["dataclass_name"].apply(lambda x: 1 if x == "Establishment Births" else 0)
        
        # handle misssing values
        df = df.replace("-", np.nan)
        df = df.dropna(subset=["year", "value", "industry_name", "sizeclass_name"])
        
        # conver numeric fields
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        
        # drop remaining nans
        df = df.dropna(subset=["year", "value"])
        
        # featur engineering
        df["year_since_2005"] = df["year"] - 2005  # normalize year
        
        # one-hot encode categorical varibles
        df = pd.get_dummies(df, columns=["industry_name", "sizeclass_name"])
        
        # select fetures
        feature_cols = ["year", "year_since_2005", "value"] + [
            col for col in df.columns if col.startswith("industry_name_") or col.startswith("sizeclass_name_")
        ]
        
    else:
        # working with aggregated bds data
        print(f"preprocessing aggregated bds data from {filepath}")
        
        # create target variable
        df["label"] = (df["births"] > df["deaths"]).astype(int)
        
        # feature engineering
        if "net_jobs" not in df.columns:
            df["net_jobs"] = df["births"] - df["deaths"]
        
        if "survival_rate" not in df.columns:
            # avoid divison by zero
            df["survival_rate"] = df["births"] / (df["births"] + df["deaths"])
        
        # one-hot encode catagorical variables
        df = pd.get_dummies(df, columns=["industry_name"])
        
        # select features (exclude births and deaths to avoid data leakage)
        feature_cols = ["year", "net_jobs", "survival_rate"] + [
            col for col in df.columns if col.startswith("industry_name_")
        ]
    
    # create feature matrx and target vector
    X = df[feature_cols]
    y = df["label"]
    
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # scale numerial features
    scaler = StandardScaler()
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    print(f"data preprocessed succesfully: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    print(f"features: {len(feature_cols)}")
    print(f"class distrubution - training: {y_train.value_counts().to_dict()}")
    print(f"class distrubution - testing: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test, feature_cols

if __name__ == "__main__":
    # test the preprossecing function
    try:
        X_train, X_test, y_train, y_test, features = load_and_preprocess_data()
        if X_train is not None:
            # save preprocessed data for reus
            output_dir = "../processed"
            os.makedirs(output_dir, exist_ok=True)
            
            X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
            X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
            y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
            y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
            
            print(f" preprocessed data saved to: {output_dir}")
    except Exception as e:
        print(f"error during preprossecing: {e}")
