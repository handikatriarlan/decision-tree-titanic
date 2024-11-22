import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    return train_data, test_data

def handle_missing_data(train_data, test_data):
    train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
    test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())

    train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].median())
    test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())

    train_data['Emb_1'] = train_data['Emb_1'].fillna(0)
    test_data['Emb_1'] = test_data['Emb_1'].fillna(0)

    train_data['Emb_2'] = train_data['Emb_2'].fillna(0)
    test_data['Emb_2'] = test_data['Emb_2'].fillna(0)

    train_data['Emb_3'] = train_data['Emb_3'].fillna(1)
    test_data['Emb_3'] = test_data['Emb_3'].fillna(1)

    return train_data, test_data

def select_features(data):
    features = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex', 'Age', 'Fare',
                'Family_size', 'Title_1', 'Title_2', 'Title_3', 'Title_4',
                'Emb_1', 'Emb_2', 'Emb_3']
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

def make_predictions(model, test_data, features):
    test_features = test_data[features]
    predictions = model.predict(test_features)
    test_data['Survived'] = predictions
    return test_data

def plot_survived_vs_sex(data):
    data['Sex_label'] = data['Sex'].map({1: 'female', 0: 'male'})
    survival_by_sex = pd.crosstab(data['Sex_label'], data['Survived'])
    survival_by_sex.plot(kind='bar', stacked=True, color=['#d9534f', '#5cb85c'])
    plt.title('Survived vs Sex')
    plt.xlabel('Sex')
    plt.ylabel('Number of Passengers')
    plt.legend(['Not Survived', 'Survived'])
    plt.show()

def main():
    train_data = pd.read_csv('data/train_data.csv')
    test_data = pd.read_csv('data/test_data.csv')

    train_data, test_data = handle_missing_data(train_data, test_data)
    
    X_train, y_train = select_features(train_data)
    X_test, _ = select_features(test_data)
    
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)
    
    model = train_decision_tree(X_train_split, y_train_split)
    accuracy = evaluate_model(model, X_val_split, y_val_split)
    print(f"Validation Accuracy: {accuracy:.2f}")
    
    test_data = make_predictions(model, test_data, X_train.columns)
    
    survived_count = test_data['Survived'].sum()
    total_count = len(test_data)
    
    print(f"Out of {total_count} passengers, {survived_count} are predicted to survive.")
    
    plot_survived_vs_sex(train_data)

if __name__ == "__main__":
    main()
