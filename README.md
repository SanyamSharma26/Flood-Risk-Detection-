# Flood Risk Prediction Using Machine Learning

A data-driven flood detection and risk classification system built using machine learning and deep learning techniques to predict flood risks in different geographical regions.

## 📋 Table of Contents

- [Abstract](#abstract)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Data Pipeline](#data-pipeline)
- [Model Details](#model-details)
- [Usage](#usage)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)
- [License](#license)

## 🌊 Abstract

Floods are among the most devastating natural disasters, causing significant loss of life and property. Accurate flood risk prediction helps authorities take preventive actions in time.

This project presents a **data-driven flood detection and risk classification system** built using:
- **Machine Learning**: Random Forest, XGBoost, Logistic Regression
- **Deep Learning**: CNN and GAN modules

Due to the absence of true flood masks, a **proxy labeling approach** based on rainfall and flow accumulation data was used to simulate flood risk levels.

**The system achieves ~99% accuracy** and is capable of classifying regions into **low, medium, and high-risk categories**.

## ✨ Features

- **Proxy-labeled flood risk dataset** derived from rainfall and flow accumulation metrics
- **Ensemble modeling** combining Random Forest, XGBoost, and Logistic Regression
- **CNN module** for spatial raster data classification (experimental)
- **GAN module** for synthetic data generation to improve model generalization
- **High accuracy** (~99%) flood risk classification
- **Three-tier risk assessment**: Low, Medium, and High risk categories
- **Scalable pipeline** for real-time flood monitoring applications

## 🏗️ System Architecture

```
┌──────────────────────────┐
│ Raw Geospatial Data      │
│ (Rainfall, NDVI, FlowAcc)│
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Feature Extraction       │
│ (Raster → Tabular Data)  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Proxy Labeling Module    │
│ (Rainfall + Flow Accum.) │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Preprocessing & Scaling  │
│ (Imputer + StandardScaler)│
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Model Training (Ensemble)│
│ RF + XGB + LR Voting Model│
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Flood Risk Prediction    │
│ (Low / Medium / High)    │
└──────────────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/flood-risk-prediction.git
cd flood-risk-prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Required Libraries

```
pandas
numpy
scikit-learn
xgboost
pytorch
imbalanced-learn
matplotlib
seaborn
pyarrow  # for parquet support
```

## 📊 Data Pipeline

### 4.1 Feature Extraction

- **Input sources**: Rainfall, NDVI, slope, elevation, and flow accumulation rasters
- **Output format**: Tabular data (`features.parquet`)
- **Dataset size**: 238,640 samples × 609 features

### 4.2 Proxy Labeling

Since ground-truth flood masks were unavailable (all zeros), proxy labels were created using:

```
Flood Likelihood = FlowAccumulation + Rainfall
```

- **Quantile thresholds** used to derive risk levels:
  - **Low risk**: Below 33rd percentile
  - **Medium risk**: 33rd to 66th percentile
  - **High risk**: Above 66th percentile

### 4.3 Preprocessing

1. **Handling Missing Values**
   - Detected ~7.3 million NaN values
   - Applied `SimpleImputer(strategy='mean')`

2. **Feature Scaling**
   - Applied `StandardScaler` for normalization
   - Ensures all features contribute equally to model training

3. **Class Balancing**
   - Used **SMOTE** (Synthetic Minority Over-sampling Technique)
   - Balanced flooded and non-flooded samples
   - Result: ~386,000 balanced samples

## 🤖 Model Details

### Ensemble Model

The final model integrates three classifiers using **hard voting**:

1. **Random Forest Classifier**
   - Robust against overfitting
   - Handles non-linear relationships well

2. **XGBoost Classifier**
   - Gradient boosting for high performance
   - Excellent handling of imbalanced data

3. **Logistic Regression**
   - Fast and interpretable baseline
   - Works well for linearly separable classes

### Performance Metrics

| Metric     | Score |
|------------|-------|
| Accuracy   | 0.99  |
| Precision  | 0.98  |
| Recall     | 0.99  |
| F1-Score   | 0.985 |

### Deep Learning Modules (Optional)

#### CNN Module
- Used for image-like raster classification
- Experimental feature for spatial pattern recognition

#### GAN Module
- Generates synthetic tabular samples using PyTorch
- Improves model generalization and data diversity
- Helps stabilize training on imbalanced datasets

## 🚀 Usage

### Making Predictions

Use the prediction script to classify flood risk on new data:

```bash
python scripts/predict.py --input new_features.parquet --output predictions.csv
```

### Script Parameters

| Parameter  | Description                          | Required |
|------------|--------------------------------------|----------|
| `--input`  | Path to input feature file (parquet) | Yes      |
| `--output` | Path to save predictions (CSV)       | Yes      |

### Example Output

```csv
region_id,latitude,longitude,risk_level,confidence
1,28.6139,77.2090,High,0.95
2,19.0760,72.8777,Medium,0.87
3,13.0827,80.2707,Low,0.92
```

## 📈 Results

### Key Achievements

✅ **99% classification accuracy** on test dataset  
✅ **Proxy labeling** successfully replaced missing ground-truth flood masks  
✅ **GAN-based augmentation** improved data diversity and model stability  
✅ **Robust ensemble** handles varying geographical conditions effectively  

### Model Performance Visualization

The ensemble model demonstrates:
- High precision in identifying high-risk zones
- Excellent recall for detecting potential flood areas
- Balanced performance across all risk categories

## 🛠️ Tech Stack

| Category          | Tools / Libraries                    |
|-------------------|--------------------------------------|
| **Language**      | Python                               |
| **Data Processing** | Pandas, NumPy                      |
| **Modeling**      | Scikit-learn, XGBoost, PyTorch       |
| **Balancing**     | Imbalanced-learn (SMOTE)             |
| **Visualization** | Matplotlib, Seaborn                  |
| **Data Format**   | Parquet, CSV                         |
| **Deep Learning** | PyTorch (GAN implementation)         |

## 🎯 Future Enhancements

- [ ] Real-time data integration from weather APIs
- [ ] Web-based dashboard for visualization
- [ ] Mobile app for emergency alerts
- [ ] Integration with satellite imagery for live monitoring
- [ ] Multi-region comparative analysis
- [ ] Temporal prediction (flood forecasting)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact the maintainers.

---

**⚠️ Disclaimer**: This system is designed for research and early warning purposes. Always consult official meteorological and disaster management authorities for critical decisions.

**Made with ❤️ for disaster prevention and public safety**
