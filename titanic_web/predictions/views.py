from django.shortcuts import render
from joblib import load
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'decision_tree_model.pkl')

model = load(MODEL_PATH)

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
    required_features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'CabinKnown', 
                        'Embarked_Q', 'Embarked_S']
    
    for feature in required_features:
        if feature not in data.columns:
            data[feature] = 0
    
    return data[required_features]

def predict(request):
    if request.method == 'POST':
        Pclass = int(request.POST.get('Pclass', 0))
        Age = float(request.POST.get('Age', 0))
        SibSp = int(request.POST.get('SibSp', 0))
        Parch = int(request.POST.get('Parch', 0))
        Fare = float(request.POST.get('Fare', 0))
        Sex = int(request.POST.get('Sex_male', 0))
        Embarked_S = int(request.POST.get('Embarked_S', 0))
        Embarked_Q = 0

        input_dict = {
            'Pclass': [Pclass],
            'Sex': ['male' if Sex == 1 else 'female'],
            'Age': [Age],
            'SibSp': [SibSp],
            'Parch': [Parch],
            'Fare': [Fare],
            'Embarked': ['S' if Embarked_S == 1 else 'Q' if Embarked_Q == 1 else 'C'],
            'Cabin': ['Unknown']
        }

        input_data = pd.DataFrame(input_dict)

        input_data = handle_missing_data(input_data)
        input_data = feature_engineering(input_data)
        input_data = select_features(input_data)

        try:
            prediction = model.predict(input_data)
            result = "Survived" if prediction[0] == 1 else "Not Survived"

            return render(request, 'predictions/result.html', {'result': result})
        except ValueError as e:
            return render(request, 'predictions/index.html', {'error': str(e)})

    return render(request, 'predictions/index.html')
