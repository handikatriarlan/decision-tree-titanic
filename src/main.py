import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

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

def train_and_save_model():
    data = load_data('data/data.csv')
    data = handle_missing_data(data)
    data = feature_engineering(data)
    X, y = select_features(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'model/decision_tree_model.pkl')

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

def load_model():
    return joblib.load('model/decision_tree_model.pkl')

if __name__ == "__main__":
    train_and_save_model()
