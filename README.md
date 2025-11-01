Machine Learning Projects

This repository showcases multiple supervised machine learning models implemented in Python.
Each project demonstrates the end-to-end ML pipeline — including data preprocessing, feature engineering, model training, evaluation, and prediction.

📁 Projects Overview
1. 🪙 Gold Price Prediction

Objective:
Develop a regression model to predict gold prices based on historical financial data.

Machine Learning Task:

Type: Supervised Learning (Regression)

Algorithm Used: Linear Regression

Pipeline Overview:

Data Preprocessing:

Handled missing values (if any)

Performed feature correlation analysis

Split the dataset into training (80%) and testing (20%) sets

Exploratory Data Analysis (EDA):

Correlation heatmaps to identify highly influential features

Pair plots and trend visualization of target variable

Model Training:

Applied Linear Regression using the scikit-learn library

Trained the model on normalized continuous financial variables

Model Evaluation:

Metrics: R² Score, Mean Absolute Error (MAE), Mean Squared Error (MSE)

The model achieved a high coefficient of determination (R²), indicating strong predictive performance

Key Insights:

Gold price shows a strong correlation with variables such as stock indices and USD fluctuations

Linear regression provided interpretable coefficients for feature importance

Technologies & Libraries:

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

2. ⚙️ Rock vs Mine Prediction

Objective:
Build a binary classification model to distinguish between metal cylinders (mines) and rocks based on sonar signal data.

Machine Learning Task:

Type: Supervised Learning (Classification)

Algorithm Used: Logistic Regression

Pipeline Overview:

Data Acquisition:

Used the Sonar dataset from the UCI Machine Learning Repository

Data Preprocessing:

Normalized input features for better convergence

Encoded target labels (R for Rock, M for Mine)

Feature Engineering:

No explicit dimensionality reduction (since dataset is already low-dimensional)

Potential for Principal Component Analysis (PCA) in future versions

Model Training:

Trained Logistic Regression classifier using scikit-learn

Evaluation Metrics:

Accuracy Score

Confusion Matrix

Precision / Recall / F1-Score (optional)

Model Validation:

Cross-validation to ensure generalization across unseen samples

Key Insights:

Logistic regression achieved strong classification performance with minimal preprocessing

The model effectively separated sonar signal patterns between metallic and non-metallic objects

Technologies & Libraries:

Python, Pandas, NumPy, Scikit-learn

⚙️ Installation & Requirements

Install all dependencies using pip:

pip install numpy pandas matplotlib seaborn scikit-learn


Ensure that Jupyter Notebook is installed for running .ipynb files:

pip install jupyter

🚀 How to Run

Clone this repository:

git clone https://github.com/srikar0313/ML-projects.git


Navigate to the project directory:

cd ML-projects


Launch Jupyter Notebook:

jupyter notebook


Open the desired notebook (Gold_price_prediction.ipynb or Rock_vs_Mine_prediction.ipynb)

Run all cells sequentially.

📊 Future Enhancements

Integrate ensemble methods like Random Forest and Gradient Boosting for comparison

Apply hyperparameter tuning using GridSearchCV or RandomizedSearchCV

Introduce regularization techniques (L1/L2) to improve model generalization

Experiment with deep learning models (ANNs) for non-linear relationships

Add feature importance visualization for interpretability

🧑‍💻 Author

Srikar
