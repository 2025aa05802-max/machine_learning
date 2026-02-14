# Dry Bean Classification — ML Assignment 2

## a. Problem Statement

The goal of this project is to classify **dry bean varieties** based on their geometric and shape features using multiple machine learning classification models. Accurate classification of dry beans is important for quality control in agricultural production. We implement and compare six different ML classifiers and deploy an interactive Streamlit dashboard for real-time model evaluation and prediction.

## b. Dataset Description

| Property | Detail |
|---|---|
| **Name** | Dry Bean Dataset |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) |
| **Total Instances** | 13,611 |
| **Total Features** | 16 (all numeric) |
| **Number of Classes** | 7 |
| **Target Variable** | Class (bean variety) |
| **Train / Test Split** | 10,888 / 2,723 (80/20, stratified) |

**Classes:** BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA

**Features:** Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, Roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4

The features capture the geometrical properties of bean images obtained through a computer vision system. There are no missing values in the dataset.

## c. Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9207 | 0.9934 | 0.9215 | 0.9207 | 0.9209 | 0.9042 |
| Decision Tree | 0.8942 | 0.9340 | 0.8942 | 0.8942 | 0.8941 | 0.8722 |
| KNN | 0.9137 | 0.9837 | 0.9144 | 0.9137 | 0.9139 | 0.8956 |
| Naive Bayes | 0.8979 | 0.9902 | 0.9007 | 0.8979 | 0.8981 | 0.8773 |
| Random Forest (Ensemble) | 0.9199 | 0.9921 | 0.9199 | 0.9199 | 0.9198 | 0.9032 |
| XGBoost (Ensemble) | 0.9229 | 0.9933 | 0.9231 | 0.9229 | 0.9229 | 0.9067 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Performs very well (92.07% accuracy, AUC 0.9934) despite being a linear model. The high AUC indicates excellent class-separation capability across all bean types. The strong performance suggests that the geometric features have significant linear separability. It is also computationally efficient and provides interpretable coefficients. |
| Decision Tree | Lowest performer among all models (89.42% accuracy). The relatively lower AUC (0.9340) indicates it struggles with probability calibration. Prone to overfitting on certain feature splits despite using max_depth=15. The gap between its AUC and other models highlights its limitation in capturing smooth decision boundaries needed for this dataset. |
| KNN | Achieves solid accuracy (91.37%) and benefits from the well-scaled numeric features. The distance-based approach works well for bean classification since similar bean types cluster together in feature space. Performance is slightly lower than ensemble methods, likely because some class boundaries overlap in the 16-dimensional feature space. |
| Naive Bayes | Accuracy is moderate (89.79%) but the AUC is surprisingly high (0.9902), indicating good probabilistic ranking despite the conditional independence assumption being violated (geometric features are correlated). The gap between AUC and accuracy suggests that while the model ranks classes well, its hard decision boundaries are less precise. |
| Random Forest (Ensemble) | Strong performer (91.99% accuracy, AUC 0.9921). The ensemble of 200 trees effectively reduces overfitting compared to the single Decision Tree. Captures non-linear feature interactions well. Close to XGBoost in all metrics, demonstrating the power of bagging with random feature selection for this dataset. |
| XGBoost (Ensemble) | Best overall performer across all metrics (92.29% accuracy, 0.9933 AUC, 0.9067 MCC). The gradient boosting approach excels by iteratively correcting errors. Its sequential learning strategy gives it an edge over Random Forest's parallel approach. The highest MCC score confirms balanced performance across all seven bean classes. |

## Project Structure

```
project-folder/
│── app.py                  # Streamlit web application
│── requirements.txt        # Python dependencies
│── README.md               # Project documentation
│── download_data.py        # Script to download dataset
│── data/
│   ├── dry_bean.csv        # Full dataset
│   └── dry_bean_test.csv   # Test split for app demo
│── model/
│   ├── train_models.py     # Model training script
│   ├── results.json        # Evaluation metrics (JSON)
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
```

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Live App

[Click here to open the Streamlit App](#) *(update with deployed link)*

## References

- Koklu, M. and Ozkan, I.A., 2020. Multiclass classification of dry beans using computer vision and machine learning techniques. *Computers and Electronics in Agriculture*, 174, 105507.
- [UCI Machine Learning Repository — Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
