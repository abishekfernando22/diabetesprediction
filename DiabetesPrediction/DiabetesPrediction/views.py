from django.shortcuts import render
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from django.http import HttpResponse

def home(request):
    return render(request, 'home.html')

def result(request):
    if request.method == 'GET':
        data = pd.read_csv(r'C:\Users\Pushkara Samarakoon\Desktop\Abishek Fernando\Bsc_Final_Project\diabetes.csv')

        X = data.drop(columns='Outcome', axis=1)
        Y = data['Outcome']

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

        model_rf_final = RandomForestClassifier(max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=100)
        model_rf_final.fit(X_train, Y_train)

        try:
            val1 = float(request.GET.get('n1', ''))
            val2 = float(request.GET.get('n2', ''))
            val3 = float(request.GET.get('n3', ''))
            val4 = float(request.GET.get('n4', ''))
            val5 = float(request.GET.get('n5', ''))
            val6 = float(request.GET.get('n6', ''))
            val7 = float(request.GET.get('n7', ''))
            val8 = float(request.GET.get('n8', ''))
        except ValueError:
            # Handle invalid input, e.g., display an error message or set default values
            return HttpResponse("Invalid input. Please enter Values. Go back to Start Predicting...")

        pred = model_rf_final.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

        if pred == 1:
            result1 = "Positive"
        else:
            result1 = "Negative"

        return render(request, 'predict.html', {"result2": result1})
    else:
        return HttpResponse("Invalid request method. Only requests are allowed.")


def predict(request):
    if request.method == 'GET':
        name = request.POST.get('n9', '')
        gender = request.POST.get('gender', '')
        physical_activity = request.POST.get('n11', '')
        educational_level = request.POST.get('Educational Level', '')
        smoking_history = request.POST.get('n12', '')
        pregnancies = request.POST.get('n1', '')
        glucose = request.POST.get('n2', '')
        blood_pressure = request.POST.get('n3', '')
        skin_thickness = request.POST.get('n4', '')
        insulin = request.POST.get('n5', '')
        bmi = request.POST.get('n6', '')
        diabetes_pedigree_function = request.POST.get('n7', '')
        age = request.POST.get('n8', '')


        return render(request, 'predict.html', {
            'name': name,
            'gender': gender,
            'physical_activity': physical_activity,
            'educational_level': educational_level,
            'smoking_history': smoking_history,
            'pregnancies': pregnancies,
            'glucose': glucose,
            'blood_pressure': blood_pressure,
            'skin_thickness': skin_thickness,
            'insulin': insulin,
            'bmi': bmi,
            'diabetes_pedigree_function': diabetes_pedigree_function,
            'age': age
        })
    else:
        return render(request, 'predict.html')

def predresult(request):
    # This view renders the predresult.html page directly
    return render(request, 'predresult.html')

def bmi(request):
    return render(request, 'bmi.html')

def bmi(request):
    return render(request, 'bmi.html')