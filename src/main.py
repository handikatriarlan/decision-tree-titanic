import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    return train_data, test_data

def handle_missing_data(train_data, test_data):
    for dataset in [train_data, test_data]:
        dataset['Age'] = dataset['Age'].fillna(dataset['Age'].median())
        dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
        dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])
        dataset['Cabin'] = dataset['Cabin'].fillna('Unknown')
    
    return train_data, test_data

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
    y = data['Survived'] if 'Survived' in data else None
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

def make_predictions(model, test_data, features):
    test_features = test_data[features]
    predictions = model.predict(test_features)
    test_data['Survived'] = predictions
    return test_data

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

def main():
    train_data, test_data = load_data('data/train.csv', 'data/test.csv')

    train_data, test_data = handle_missing_data(train_data, test_data)

    train_data = feature_engineering(train_data)
    test_data = feature_engineering(test_data)

    X, y = select_features(train_data)
    X_test, _ = select_features(test_data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_decision_tree(X_train, y_train)

    accuracy = evaluate_model(model, X_val, y_val)
    print(f"Validation Accuracy: {accuracy:.2f}")

    test_data = make_predictions(model, test_data, X.columns)

    plot_feature_importance(model, X.columns)

    plot_survival_distribution(train_data)

    survived_count = test_data['Survived'].sum()
    total_count = len(test_data)
    print(f"Out of {total_count} passengers, {survived_count} are predicted to survive.")

if __name__ == "__main__":
    main()
