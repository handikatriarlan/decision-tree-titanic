from django.shortcuts import render
from joblib import load
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'decision_tree_model.pkl')

model = load(MODEL_PATH)

def feature_engineering(data):
    data['CabinKnown'] = data['Cabin'].apply(lambda x: 0 if x == 'Unknown' else 1)
    data = pd.get_dummies(data, columns=['Embarked'], prefix='Embarked', drop_first=True)
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    return data


def select_features(data):
    required_features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'CabinKnown', 'Embarked_Q', 'Embarked_S']
    for feature in required_features:
        if feature not in data.columns:
            data[feature] = 0

    return data[required_features]


def predict(request):
    if request.method == 'POST':
        Pclass = int(request.POST.get('Pclass', 0))
        Cabin = request.POST.get('Cabin', 'Unknown')
        Age = float(request.POST.get('Age', 0))
        SibSp = int(request.POST.get('SibSp', 0))
        Parch = int(request.POST.get('Parch', 0))
        Fare = float(request.POST.get('Fare', 0))
        Sex = int(request.POST.get('Sex', 0))
        Embarked = request.POST.get('Embarked', 'C')

        input_dict = {
            'Pclass': [Pclass],
            'Sex': [Sex],
            'Age': [Age],
            'SibSp': [SibSp],
            'Parch': [Parch],
            'Fare': [Fare],
            'Embarked': [Embarked],
            'Cabin': [Cabin],
        }

        input_data = pd.DataFrame(input_dict)

        input_data = feature_engineering(input_data)
        input_data = select_features(input_data)

        try:
            prediction = model.predict(input_data)
            result = "Survived" if prediction[0] == 1 else "Not Survived"

            return render(request, 'result.html', {'result': result})
        except ValueError as e:
            return render(request, 'index.html', {'error': f"Error in prediction: {str(e)}"})

    return render(request, 'index.html')

