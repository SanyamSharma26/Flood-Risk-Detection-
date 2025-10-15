
# Flood Risk Prediction Project – Final Documentation

## 📖 1. Project Overview

The objective of this project was to build a **flood detection and risk classification system** using data.  
flood masks were all `0`, we shifted to a **proxy-label approach** (rainfall & flow accumulation). This gave us a practical way to train models without ground-truth flood masks.

The final system delivers:

-   A **machine learning ensemble model** (Random Forest + XGBoost + Logistic Regression) for flood risk classification.
    
-   Optional **GAN augmentation** module for generating synthetic samples.
    
-   Preprocessing pipeline (imputation + scaling).
    
-   Predict script for deployment.
    

----------

## 2. Data Pipeline

### 2.1 Feature Extraction

-   Input rasters (rainfall, NDVI, flow accumulation, etc.) were aligned and converted into a **feature matrix**.
    
-   Final `features.parquet` had **238,640 samples × 609 features**.
    

### 2.2 Labeling (Flood Proxy Labels)

Since masks were empty (`0` everywhere), we used **proxy labeling**:

-   `FlowAccumulation + Rainfall` → proxy indicator for flooding.
    
-   Thresholds derived via quantiles → flood/not flood.
    
-   Additional risk levels → **low, medium, high** using `pd.qcut`.
    

----------

## 3. Preprocessing

1.  **Handling NaNs**
    
    -   Found ~7.3M NaNs in dataset.
        
    -   Used `SimpleImputer(strategy='mean')` to fill missing values.
        
    -   Columns with no non-missing values were skipped automatically.
        
2.  **Scaling**
    
    -   Standardized features using `StandardScaler`.
        
    -   Produced `X_scaled` → ready for ML.
        
3.  **Balancing Classes**
    
    -   Used `SMOTE` to oversample minority class (flooded points).
        
    -   Result: perfectly balanced dataset with ~386k rows.
        

----------

##  4. Model Training

    

### 4.1 Ensemble Model (Final)

To improve robustness:

-   **RandomForestClassifier**
    
-   **XGBoostClassifier**
    
-   **LogisticRegression**
    
-   Combined via **VotingClassifier (hard voting)**
    
Achieved: **~99% accuracy**
    

----------

##  5. GAN Module (Optional)


-   Implemented `train_gan.py` (PyTorch).
    
-   Trains Generator + Discriminator on balanced dataset.
    
-   Generates synthetic samples (`synthetic_samples.parquet`).
    
-   These can be merged with real data to retrain ensemble.
    


----------


## 🚀 8. Usage Guide

1.  **Train Model**
    
    `python scripts/train_ensemble.py` 
    
    → Saves ensemble model in `models/`
    
2.  **Predict on New Data**
    
    `python scripts/predict.py --input new_features.parquet --output predictions.csv` 
    
3.  **GAN Augmentation (Optional)**
    
    `python scripts/train_gan.py
    python scripts/merge_with_synthetic.py` 
    

----------


##  9. Final Outcome

-   Ensemble flood risk model (robust, balanced).
    
-   Proxy labeling strategy → practical workaround for missing masks.
    
-   High accuracy (~99%).
    
