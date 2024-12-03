import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Eksplorasi Data
def explore_data(data):
    print("Preview Data:")
    print(data.head())
    print("\nInformasi Dataset:")
    print(data.info())
    print("\nStatistik Deskriptif:")
    print(data.describe())
    print("\nJumlah Nilai Kosong:")
    print(data.isnull().sum())
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Data Heatmap')
    plt.show()

# Validasi Underfitting/Overfitting
def evaluate_overfitting(model, X_train, y_train, X_val, y_val):
    train_accuracy = model.score(X_train, y_train)
    val_accuracy = model.score(X_val, y_val)
    print(f"Training Accuracy: {train_accuracy:.2f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}")
    if abs(train_accuracy - val_accuracy) > 0.1:
        print("Model menunjukkan tanda-tanda overfitting.")
    elif train_accuracy < 0.6:
        print("Model menunjukkan tanda-tanda underfitting.")
    else:
        print("Model memiliki performa yang seimbang.")
    return train_accuracy, val_accuracy

# Fungsi lainnya tetap sama
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def handle_missing_data(data):
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    data['Cabin'] = data['Cabin'].fillna('Unknown')
    return data

def feature_engineering(data):
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['CabinKnown'] = data['Cabin'].apply(lambda x: 0 if x == 'Unknown' else 1)
    data = pd.get_dummies(data, columns=['Embarked'], prefix='Embarked', drop_first=True)
    return data

def select_features(data):
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'CabinKnown', 
                'Embarked_Q', 'Embarked_S']
    X = data[features]
    y = data['Survived']
    return X, y

def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print("Classification Report:\n", classification_report(y_val, y_pred))
    return accuracy

def plot_feature_importance(model, features):
    importance = model.feature_importances_
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance, y=features, palette='viridis', legend=False)
    plt.title('Feature Importance in Decision Tree')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.show()

def plot_survival_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x='Survived', hue='Sex', palette='Set2')
    plt.title('Survival Distribution by Gender')
    plt.xlabel('Survived')
    plt.ylabel('Count')
    plt.legend(title='Gender', loc='upper right', labels=['Male', 'Female'])
    plt.show()

# Main function
def main():
    data = load_data('data/data.csv')

    # Eksplorasi data
    explore_data(data)

    # Handle missing data
    data = handle_missing_data(data)

    # Feature engineering
    data = feature_engineering(data)

    # Select features and split data
    X, y = select_features(data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train decision tree
    model = train_decision_tree(X_train, y_train)

    # Evaluate model
    print("\n=== Evaluasi Model ===")
    accuracy = evaluate_model(model, X_val, y_val)
    print(f"Validation Accuracy: {accuracy:.2f}")

    # Evaluate overfitting/underfitting
    print("\n=== Analisis Overfitting/Underfitting ===")
    evaluate_overfitting(model, X_train, y_train, X_val, y_val)

    # Plot feature importance and survival distribution
    plot_feature_importance(model, X.columns)
    plot_survival_distribution(data)

if __name__ == "__main__":
    main()
