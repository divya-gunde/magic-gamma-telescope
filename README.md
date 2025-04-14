# MAGIC Gamma Telescope – Particle Classification with Machine Learning

This project leverages the **MAGIC Gamma Telescope dataset** to classify high-energy particle events as either gamma rays or background cosmic rays using various machine learning algorithms.

## 📖 Overview

The dataset comprises data from the **Major Atmospheric Gamma Imaging Cherenkov (MAGIC)** telescope. The primary objective is to develop models that can accurately distinguish between signal (gamma) and background (hadron) events.

Key steps in this project include:
- Data preprocessing and exploration
- Feature scaling and correlation analysis
- Model training using multiple classification algorithms
- Evaluation using metrics like accuracy, precision, recall, F1-score, and ROC AUC

## 📂 Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope)
- **Instances**: 19,020
- **Features**: 10 numerical attributes
- **Target**: Binary classification – `gamma` or `hadron`

## 🧠 Machine Learning Models Implemented

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

## 📊 Results

| Model                 | Accuracy |
|-----------------------|----------|
| Logistic Regression   | 82.3%    |
| Random Forest         | 88.5%    |
| SVM                   | 85.7%    |
| KNN                   | 83.1%    |

*Note: These results are based on initial evaluations. Further tuning and validation may improve performance.*

## 🛠️ Technologies Used

- Python
- Google Colab
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## 📁 Repository Structure

magic-gamma-telescope/ ├── magic_gamma_telescope.ipynb # Main Jupyter notebook ├── requirements.txt # List of dependencies └── README.md # Project documentation

markdown
Copy
Edit

## 🚀 Future Work

- Hyperparameter tuning for improved model performance
- Implementation of ensemble methods like Gradient Boosting or XGBoost
- Feature importance analysis and visualization
- Deployment of the model using Streamlit or Flask for interactive predictions

## 👩‍💻 Author

**Divya Gunde** – Data Analyst | Business Intelligence | Machine Learning Enthusiast

## 📄 License

This project is licensed under the [MIT License](LICENSE).
