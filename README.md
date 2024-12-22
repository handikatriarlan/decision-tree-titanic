# Decision Tree Titanic - Survival Prediction

This repository contains a machine learning project focused on predicting the survival of passengers aboard the Titanic using the Decision Tree algorithm. Originally designed as a script-based prediction system, the project has now evolved into a full-fledged web application powered by Django, enabling users to input passenger data and receive real-time survival predictions based on a pre-trained model.

---

![Home Page](https://ucarecdn.com/48b92335-62c0-459c-931a-b8daef6a350f/Screenshot20241222132838127001.png)

---

## Project Overview

The goal of this project is to predict whether a passenger survived or not based on various features such as:

- Gender
- Age
- Passenger class (`Pclass`)
- Fare
- Number of siblings/spouses aboard (`SibSp`)
- Number of parents/children aboard (`Parch`)
- Embarkation port (`Embarked`)
- Cabin information (`Cabin`)

The model is implemented using the Decision Tree algorithm from Scikit-learn. The dataset used is the well-known Titanic dataset, which is commonly used for classification problems in data science. The trained model is saved in `.pkl` format and utilized in the Django web application for predictions.

---

## Web Application Features

### **1. User-Friendly Web Form:**

Users can input passenger details through a visually appealing web form developed with Tailwind CSS. Features include:

- Dropdown selections for categorical data like `Sex` and `Embarked`.
- Number inputs for continuous variables like `Age` and `Fare`.

### **2. Real-Time Predictions:**

Upon submitting the form, the web application processes the input data and predicts whether the passenger would have survived. The result is displayed dynamically with:

- Gradient-colored text for "Survived."
- Red-colored text for "Not Survived."

### **3. Model Integration:**

The web application loads the pre-trained Decision Tree model from a `.pkl` file and applies feature engineering to ensure compatibility with the model’s requirements.

---

## Machine Learning Workflow

### **1. Data Preprocessing**

- Handles missing data in critical columns like `Age` and `Fare` using median imputation.
- Encodes categorical features, such as `Sex` (`male = 0`, `female = 1`) and `Embarked` (one-hot encoding).
- Adds derived features like `FamilySize` and `CabinKnown` for improved prediction accuracy.

### **2. Feature Selection**

- Selected features include: `Pclass`, `Sex`, `Age`, `Fare`, `FamilySize`, `CabinKnown`, `Embarked_Q`, and `Embarked_S`.

### **3. Decision Tree Classifier**

- Implements a Decision Tree algorithm to learn survival patterns.
- Limits tree depth (`max_depth`) for better generalization and to avoid overfitting.

### **4. Model Evaluation**

- Evaluates performance using metrics such as:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Outputs a classification report and confusion matrix.

---

## Web Application Setup

### **1. Clone the Repository:**

```bash
git clone https://github.com/handikatriarlan/decision-tree-titanic.git
```

### **2. Navigate to the Project Directory:**

```bash
cd decision-tree-titanic
```

### **3. Install Dependencies:**

You can install these dependencies using the following command:

```bash
pip install -r requirements.txt
```

### **4. Migrate Django Database:**

```bash
python titanic_web/manage.py migrate
```

### **5. Run the Development Server:**

```bash
python titanic_web/manage.py runserver 7000
```

### **6. Access the Web Application:**

Open your browser and navigate to: [http://127.0.0.1:7000/](http://127.0.0.1:7000/)

---

## Results

The Decision Tree model achieved an accuracy of approximately 80% on the validation dataset, with the following key insights:

### **1. Women and Children First:**

- Gender (`Sex`) is the most influential feature, followed by passenger age.
- Female passengers and younger individuals had higher survival rates.

### **2. Passenger Class and Fare:**

- Passengers in higher classes (`Pclass = 1`) had better survival chances.

### **3. Feature Importance:**

- The `Sex` feature contributes most significantly to the decision tree's predictions, underscoring its critical role in the Titanic survival context.

---

## Screenshots

### **1. Input Form:**

A clean and modern form for inputting passenger data, powered by Tailwind CSS.

### **2. Prediction Result:**

Dynamic display of survival predictions:

- Gradient text for "Survived."
- Red-colored text for "Not Survived."

---

## Acknowledgements

- Titanic dataset is provided by [Kaggle](https://www.kaggle.com/competitions/titanic/overview).
- [Scikit-learn](https://scikit-learn.org/stable) for the machine learning tools and algorithms.
- [Django](https://www.djangoproject.com) for powering the web application development.
- [Tailwind](https://tailwindcss.com) CSS for styling the web application.

---

Thank you for exploring this project! Feel free to contribute or raise issues for improvements.
