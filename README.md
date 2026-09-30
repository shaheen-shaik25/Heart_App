# ❤️ CardioCheck — Heart Disease Risk Prediction

> **An end-to-end machine learning web application for heart disease risk prediction, model comparison, explainable predictions, batch inference, and interactive health analytics.**

Built with **Python, Flask, Scikit-learn, Pandas, SQLite, HTML, CSS, and JavaScript**.

---

## 🚀 Overview

**CardioCheck** transforms a machine-learning heart disease prediction model into a complete web application.

The application accepts 13 clinical features and uses multiple machine-learning models to estimate the probability of heart disease. Users can compare models, inspect prediction explanations, explore dataset insights, upload batches of patient records, maintain prediction history, and generate PDF reports.

The project was developed by converting an original ML notebook into a structured, reusable Flask application with:

- Automated preprocessing pipelines
- Multiple machine-learning models
- Hyperparameter tuning
- Cross-validation
- Model comparison
- Ensemble prediction
- Explainable predictions
- Batch prediction
- Interactive dashboards
- Prediction history
- PDF reporting
- REST API endpoints

> ⚠️ **Educational purpose only:** This project is a machine-learning demonstration and is **not a medical device or a substitute for professional medical advice or diagnosis.**

---

# ✨ Features

| Feature | Description |
|---|---|
| ❤️ **Heart Disease Prediction** | Predict heart disease risk from 13 clinical features |
| 🤖 **Multiple ML Models** | Logistic Regression, SVM, Random Forest, Decision Tree, Gradient Boosting, KNN and Voting Ensemble |
| 🏆 **Model Comparison** | Compare accuracy, precision, recall, F1-score and ROC-AUC |
| 📊 **Cross-Validation** | 5-fold cross-validation used during model evaluation |
| 🔬 **Hyperparameter Tuning** | Grid-search based model optimization |
| 💡 **Explainable Predictions** | Identify which patient features influence an individual prediction |
| 🎚️ **What-If Analysis** | Change patient values and observe prediction changes without saving a record |
| 📈 **Interactive Dashboard** | Explore class balance, age groups, feature distributions and correlations |
| 🧠 **Feature Importance** | Permutation-based feature importance |
| 📁 **Batch Prediction** | Upload a CSV containing multiple patient records |
| 📥 **CSV Export** | Download batch prediction results |
| 🗂️ **Prediction History** | Store predictions using SQLite |
| 🧾 **PDF Reports** | Generate reports for predictions |
| 🌐 **REST API** | Programmatically access prediction and model information |
| 🎨 **Responsive UI** | Dark-themed responsive web interface |
| 🔒 **Input Validation** | Validate and sanitize prediction inputs |
| 📋 **Model Consensus** | Compare probabilities from all available models |

---

# 🖥️ Application Pages

## ❤️ 1. Prediction

The main prediction interface accepts the following 13 features:

```text
Age
Sex
Chest Pain Type
Resting Blood Pressure
Cholesterol
Fasting Blood Sugar
Resting ECG
Maximum Heart Rate
Exercise-Induced Angina
ST Depression
Slope
Number of Major Vessels
Thalassemia
```

After submitting the form, the application provides:

- Predicted probability
- Risk category
- Prediction label
- Selected model
- Model consensus
- Feature-level explanation
- Rule-based health guidance
- PDF report
- Prediction history entry

---

# 🎯 Risk Prediction

The selected model produces a probability:

```text
Patient Features
       ↓
Preprocessing Pipeline
       ↓
Machine Learning Model
       ↓
Probability
       ↓
Risk Band
```

The application displays the prediction using three risk bands:

```text
Low
Moderate
High
```

The probability threshold and risk-band logic are implemented in the application configuration.

---

# 🧠 Machine Learning Pipeline

The project uses Scikit-learn preprocessing pipelines so that transformations are learned only from the training data.

```text
Raw Patient Data
       │
       ▼
Input Validation
       │
       ▼
Numerical Features ──► Scaling
       │
       ▼
Categorical Features ──► One-Hot Encoding
       │
       ▼
Machine Learning Model
       │
       ▼
Probability Prediction
       │
       ▼
Risk Classification
```

This prevents preprocessing leakage between training and test data.

---

# 🤖 Machine Learning Models

The application trains and stores multiple models:

### 1. Logistic Regression

A linear classification model that estimates the probability of heart disease.

### 2. Support Vector Machine

An SVM classifier that identifies a decision boundary between the classes.

### 3. Random Forest

An ensemble of decision trees designed to capture nonlinear relationships.

### 4. Decision Tree

A tree-based classifier that makes predictions through feature-based decision rules.

### 5. Gradient Boosting

A sequential ensemble method that combines weak learners to improve predictive performance.

### 6. K-Nearest Neighbors

Predicts the class based on neighboring observations in feature space.

### 7. Soft Voting Ensemble

Combines predictions from multiple models to produce an ensemble probability.

---

# 📊 Model Evaluation

The project evaluates models using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- 5-fold cross-validation ROC-AUC
- Confusion matrix
- ROC curve

The evaluation results are stored in:

```text
models/metadata.json
```

---

# 🏆 Model Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 83.61% | 84.85% | 84.85% | 84.85% | 0.9048 | 0.9278 |
| SVM | 78.69% | 81.25% | 78.79% | 80.00% | 0.8874 | 0.9233 |
| Random Forest | 80.33% | 81.82% | 81.82% | 81.82% | 0.8907 | 0.9170 |
| Decision Tree | 85.25% | 83.33% | 90.91% | 86.96% | 0.8658 | 0.8733 |
| Gradient Boosting | 73.77% | 74.29% | 78.79% | 76.47% | 0.8615 | 0.9039 |
| KNN | 83.61% | 87.10% | 81.82% | 84.38% | 0.9080 | 0.9120 |
| Ensemble (Voting) | 80.33% | 81.82% | 81.82% | 81.82% | 0.8961 | 0.9239 |

The application currently selects **Logistic Regression** as the configured best model based on the cross-validation ROC-AUC criterion.

> These metrics are based on the included cleaned dataset and evaluation split. They should not be interpreted as clinical performance or evidence of medical effectiveness.

---

# 🔬 Data Cleaning

The original dataset contained duplicate records.

The project removes duplicates before model training:

```text
Original rows: 1025
Clean rows:     302
```

The cleaned dataset contains:

```text
Healthy: 138
Disease: 164
```

Removing duplicate records is important because duplicated observations can appear in both training and testing data and produce overly optimistic evaluation results.

---

# 💡 Explainable Predictions

CardioCheck provides patient-level explanations rather than only returning a probability.

The application uses a **feature neutralisation approach**.

Conceptually:

```text
Original Patient
       ↓
Prediction Probability
       ↓
Neutralize One Feature
       ↓
Predict Again
       ↓
Measure Probability Change
```

The process is repeated for the input features.

This produces an explanation showing which features have the greatest effect on the prediction.

Example:

```text
Feature              Impact
--------------------------------
Chest Pain Type      Higher impact
Number of Vessels    Higher impact
ST Depression        Moderate impact
Cholesterol          Lower impact
Age                  Lower impact
```

This is intended as a model explanation, not a medical interpretation.

---

# 🎚️ What-If Analysis

The application provides an interactive **what-if** interface.

Users can change patient features and request a new prediction without saving the result.

```text
Patient Input
      ↓
Change Feature
      ↓
/whatif API
      ↓
Model Prediction
      ↓
Updated Risk Probability
```

This allows users to explore how changes in the input affect the model's output.

---

# 📊 Insights Dashboard

The Insights page provides interactive visualizations including:

### Class Distribution

```text
Healthy vs Disease
```

### Age Groups

```text
<40
40–49
50–59
60–69
70+
```

### Clinical Feature Distributions

Examples include:

- Chest pain
- Sex
- Exercise-induced angina
- Thalassemia
- Number of vessels
- Slope

### Correlation Heatmap

Shows relationships between numerical features.

### Scatter Plot

Visualizes relationships such as:

```text
Age ↔ Maximum Heart Rate
```

with class information.

### Feature Importance

Permutation-based feature importance is displayed for model interpretation.

---

# 📁 Batch Prediction

CardioCheck supports predictions for multiple patients.

Workflow:

```text
CSV File
   ↓
Upload
   ↓
Validate Required Columns
   ↓
Select Model
   ↓
Generate Predictions
   ↓
Risk Probability
   ↓
Risk Level
   ↓
Download CSV
```

A maximum of **2,000 rows** is accepted per batch upload.

The required input columns are:

```text
age
sex
cp
trestbps
chol
fbs
restecg
thalach
exang
oldpeak
slope
ca
thal
```

The resulting file includes:

```text
risk_probability_%
risk_level
prediction
```

in addition to the original input columns.

---

# 🗂️ Prediction History

Predictions are stored locally using **SQLite**.

The History page supports:

- Viewing previous predictions
- Patient name
- Selected model
- Probability
- Risk level
- Timestamp
- Deleting records
- CSV export
- Generating reports

Database:

```text
data/predictions.db
```

> For production systems, sensitive patient data should use appropriate security, access control, encryption, retention policies, and privacy safeguards.

---

# 🧾 PDF Reports

The application can generate PDF reports containing prediction information.

Reports can include:

- Patient information
- Model used
- Probability
- Risk category
- Input features
- Prediction explanation
- Supporting information

PDF generation is implemented in:

```text
report.py
```

---

# 🌐 REST API

The application exposes API endpoints for programmatic access.

## Health Check

```http
GET /api/health
```

Example:

```bash
curl http://127.0.0.1:5000/api/health
```

---

## Prediction API

```http
POST /api/predict
```

Example:

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
-H "Content-Type: application/json" \
-d '{"age":63,"sex":1,"cp":0,"trestbps":150,"chol":290,"fbs":1,"restecg":0,"thalach":110,"exang":1,"oldpeak":3.2,"slope":1,"ca":2,"thal":3}'
```

The API validates the input and returns the prediction result.

---

## Models API

```http
GET /api/models
```

Returns information about the available trained models.

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │    Heart Dataset    │
                    │     heart.csv       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Cleaning       │
                    │ Duplicate Removal   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Preprocessing       │
                    │ Scaling + Encoding  │
                    └──────────┬──────────┘
                               │
                               ▼
                 ┌─────────────┴─────────────┐
                 │       Model Training       │
                 └─────────────┬─────────────┘
                               │
          ┌────────┬───────────┼──────────┬──────────┐
          ▼        ▼           ▼          ▼          ▼
        Logistic  SVM     Random Forest  KNN    Gradient Boost
          │        │           │          │          │
          └────────┴───────────┼──────────┴──────────┘
                               ▼
                       Voting Ensemble
                               │
                               ▼
                     Model Evaluation
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
        ROC-AUC           CV Results       Feature Importance
             │                 │                 │
             └─────────────────┼─────────────────┘
                               ▼
                         Flask Backend
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
          Prediction        Dashboard        REST API
              │                │                │
              ▼                ▼                ▼
          Explainability   Visualization    Integration
              │
              ▼
         SQLite History
              │
              ▼
         PDF Reports
```

---

# 📁 Project Structure

```text
heart_app/
│
├── app.py
├── config.py
├── train.py
├── report.py
├── requirements.txt
├── README.md
├── original_notebook.ipynb
│
├── data/
│   ├── heart.csv
│   └── predictions.db
│
├── models/
│   ├── Decision_Tree.joblib
│   ├── Ensemble_Voting.joblib
│   ├── Gradient_Boosting.joblib
│   ├── KNN.joblib
│   ├── Logistic_Regression.joblib
│   ├── Random_Forest.joblib
│   ├── SVM.joblib
│   └── metadata.json
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── dashboard.html
│   ├── models.html
│   ├── history.html
│   ├── batch.html
│   └── about.html
│
└── static/
    ├── css/
    │   └── style.css
    └── js/
        └── predict.js
```

---

# 🛠️ Tech Stack

### Programming

- Python 3.9+

### Machine Learning

- Scikit-learn
- NumPy
- Pandas
- Joblib

### Models

- Logistic Regression
- Support Vector Machine
- Random Forest
- Decision Tree
- Gradient Boosting
- K-Nearest Neighbors
- Soft Voting Ensemble

### Web Development

- Flask
- Jinja2
- HTML5
- CSS3
- JavaScript

### Database

- SQLite

### Reporting

- PDF generation

### Visualization

- Chart.js

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-prediction.git
cd heart-disease-prediction
```

Replace `YOUR_USERNAME` with your GitHub username.

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

The repository already contains trained models.

Simply run:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

---

# 🧠 Retrain the Models

If you want to retrain the models from the dataset:

```bash
python train.py
```

The training script performs:

```text
Load Dataset
      ↓
Remove Duplicates
      ↓
Train/Test Split
      ↓
Preprocessing Pipeline
      ↓
Hyperparameter Search
      ↓
5-Fold Cross Validation
      ↓
Model Evaluation
      ↓
ROC Curves
      ↓
Feature Importance
      ↓
Save Models
      ↓
metadata.json
```

The trained models are saved inside:

```text
models/
```

---

# 📋 Dataset

The project uses a heart disease dataset containing clinical features and a binary target.

Input features:

| Feature | Description |
|---|---|
| `age` | Age |
| `sex` | Sex |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar |
| `restecg` | Resting ECG result |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels |
| `thal` | Thalassemia |
| `target` | Heart disease target |

---

# 🔬 Data Preprocessing

The training pipeline handles preprocessing separately for numerical and categorical features.

### Numerical features

Numerical values are scaled using a Scikit-learn preprocessing pipeline.

### Categorical features

Categorical variables are transformed using one-hot encoding.

This allows the models to receive consistently processed numerical input while preventing preprocessing information from leaking from the test set into training.

---

# 📈 Why Multiple Models?

Different algorithms can capture different relationships in the data.

Instead of relying on a single classifier, the project trains several models and compares them using multiple evaluation metrics.

This allows the application to expose model behavior rather than treating one algorithm as universally optimal.

---

# 📊 Evaluation Metrics

### Accuracy

\[
Accuracy =
\frac{TP+TN}{TP+TN+FP+FN}
\]

Measures the overall proportion of correct predictions.

### Precision

\[
Precision =
\frac{TP}{TP+FP}
\]

Measures how many predicted positive cases are actually positive.

### Recall

\[
Recall =
\frac{TP}{TP+FN}
\]

Measures how many actual positive cases are detected.

### F1 Score

\[
F1 =
2\frac{Precision\times Recall}
{Precision+Recall}
\]

Balances precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between the two classes across classification thresholds.

---

# 🔍 Feature Importance

The project calculates permutation-based feature importance.

The general process is:

```text
Trained Model
     ↓
Shuffle One Feature
     ↓
Measure Performance Change
     ↓
Repeat
     ↓
Feature Importance
```

The included metadata shows features such as:

- `ca`
- `sex`
- `thal`
- `slope`
- `trestbps`
- `oldpeak`
- `cp`

among the features with larger measured importance in the current model evaluation.

These values describe model behavior on this dataset and should not be interpreted as independent clinical evidence.

---

# 🛡️ Input Validation

The application validates incoming prediction data before sending it to the model.

This helps prevent:

- Missing required fields
- Invalid numerical values
- Unsupported categorical values
- Malformed API requests

Batch uploads are also checked for required columns and invalid values.

---

# ⚠️ Limitations

This project has several important limitations.

### Dataset Size

After duplicate removal, the included dataset contains **302 unique records**, which is relatively small for a medical prediction system.

### Dataset Dependence

Model performance depends heavily on the dataset used for training and evaluation.

### Clinical Validation

The model has not been clinically validated.

### External Generalization

Performance on this dataset does not guarantee similar performance on patients from another population, hospital, or healthcare system.

### Explainability

The feature-neutralisation explanation describes how the model responds to changes in input features. It does not establish medical causation.

### Risk Thresholds

Prediction thresholds used by the application are application-level rules and should not be treated as clinical diagnostic thresholds.

---

# 🔐 Privacy & Security

This project stores prediction history in:

```text
data/predictions.db
```

If deployed with real patient information, additional protections would be required, including:

- Authentication
- Authorization
- Encryption
- Secure database storage
- Audit logging
- Data retention policies
- Removal or protection of personally identifiable information
- Appropriate healthcare/privacy compliance

**Do not upload real patient data to a public GitHub repository.**

---

# 🚀 Future Improvements

## 1. Larger Dataset

Train and validate using larger and more diverse datasets.

## 2. External Validation

Evaluate the trained model on an independent dataset.

## 3. Advanced Explainability

Add methods such as:

- SHAP
- LIME
- Partial Dependence

## 4. Model Monitoring

Add:

- Prediction drift detection
- Data drift detection
- Model performance monitoring

## 5. Authentication

Add secure user accounts and role-based access.

## 6. Cloud Deployment

Deploy the application using:

- AWS
- Azure
- Google Cloud
- Render
- Railway

## 7. REST API Integration

Connect the model API to:

- React
- Mobile applications
- Hospital dashboards
- Other ML applications

---

# 📚 Learning Outcomes

This project demonstrates practical experience with:

- Machine learning classification
- Data cleaning
- Duplicate detection
- Feature preprocessing
- One-hot encoding
- Feature scaling
- Hyperparameter tuning
- Cross-validation
- Model comparison
- Ensemble learning
- ROC-AUC analysis
- Confusion matrices
- Feature importance
- Explainable ML
- Flask development
- REST APIs
- SQLite
- Batch inference
- PDF generation
- Interactive dashboards
- Frontend integration
- Model serialization
- End-to-end ML deployment concepts

---

# 🎯 Project Highlights

```text
✓ 7 Machine Learning Models
✓ Voting Ensemble
✓ Hyperparameter Tuning
✓ 5-Fold Cross Validation
✓ ROC-AUC Evaluation
✓ Explainable Predictions
✓ Feature Importance
✓ What-If Analysis
✓ Batch Prediction
✓ SQLite Prediction History
✓ PDF Reports
✓ REST API
✓ Interactive Dashboard
✓ Flask Web Application
✓ Responsive UI
```

---

# 📜 Disclaimer

This application is developed for **educational and research purposes only**.

It is not intended to:

- Diagnose disease
- Replace a doctor
- Recommend medical treatment
- Make clinical decisions

Predictions should not be interpreted as medical advice.

---

# 👨‍💻 Author

**Shaheen Shaik**

Computer Science Undergraduate

### Interests

```text
Machine Learning
Artificial Intelligence
Data Science
Explainable AI
Healthcare AI
Full-Stack ML Applications
```

### Project

❤️ **CardioCheck — Heart Disease Risk Prediction**

Built with:

```text
Python • Scikit-learn • Pandas • Flask • SQLite • JavaScript
```

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.
