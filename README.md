# Decision Tree Titanic - Survival Prediction

This repository contains a machine learning project focused on predicting the survival of passengers aboard the Titanic using the Decision Tree algorithm. The dataset used for this project is the well-known Titanic dataset, which is commonly used for classification problems in data science.

## Project Overview

The goal of this project is to predict whether a passenger survived or not based on various features such as:

- Gender
- Age
- Passenger class (`Pclass`)
- Fare
- Number of siblings/spouses aboard (`SibSp`)
- Number of parents/children aboard (`Parch`)

The model is implemented using the Decision Tree algorithm from Scikit-learn. It works by recursively partitioning the dataset into subsets based on the most significant features, ultimately producing a tree structure that predicts survival.

## Requirements

To run this project, ensure the following Python libraries are installed:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install these dependencies using the following command:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/handikatriarlan/decision-tree-titanic.git
```

2. Navigate to the project directory:

```bash
cd decision-tree-titanic
```

3. Run the main Python script to train the decision tree model and make predictions:

```bash
python src/decision_tree_model.py
```

This script:

- Loads and preprocesses the Titanic dataset.
- Handles missing values.
- Trains a Decision Tree Classifier.
- Evaluates the model on a validation set.
- Saves the trained model and generates visualizations.

## Features

**1. Data Preprocessing**:

- Handles missing data in critical columns like `Age` and `Fare` using median imputation.
- Encodes categorical features, such as `Sex`, into numerical values (`male = 0`, `female = 1`).

**2. Decision Tree Classifier**

- Implements a Decision Tree algorithm to learn survival patterns.
- Features include limiting tree depth (`max_depth`) for generalization and avoiding overfitting.

**3. Model Evaluation**

- Evaluates performance using metrics such as:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Outputs a classification report and confusion matrix.

**4. Visualizations**

- Feature Importance Chart:
  - Highlights how features like `Sex` and `Age` significantly impact survival prediction.
- Survival Distribution by Gender:
  - Displays survival rates among male and female passengers, emphasizing historical patterns.

## Results

The Decision Tree model achieved an accuracy of approximately 80% on the validation dataset, with the following key insights:
**1. Women and Children First**:

- Gender (`Sex`) is the most influential feature, followed by passenger age.
- Female passengers and younger individuals had higher survival rates.

**2. Passenger Class and Fare**:

- Passengers in higher classes (`Pclass = 1`) had better survival chances.

**3. Feature Importance**:

- The `Sex` feature contributes most significantly to the decision tree's predictions, underscoring its critical role in the Titanic survival context.

## Visualization

**1. Feature Importance**:
![Feature Importance](https://ucarecdn.com/5189ecd8-9343-48cd-928e-d1ef3e068386/featureimportanceindecisiontree.png)
Displays which features are most influential in the decision tree.

**2. Survival Distribution by Gender**:
![Survival Distribution by Gender](https://ucarecdn.com/9e6bf6e5-a575-42d3-a318-a9b17fde4d7f/survivaldistributionbygender.png)
Visualizes survival rates for male and female passengers.

## Acknowledgements

- Titanic dataset is provided by [Kaggle](https://www.kaggle.com/datasets/heptapod/titanic).
- Scikit-learn for the machine learning tools and algorithms.
