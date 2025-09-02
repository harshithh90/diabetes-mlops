import requests

url = "http://localhost:8000/predict"
data = {
    "features": [
        1,
        85,           # Glucose
        66,           # BloodPressure
        29,           # SkinThickness
        0,            # Insulin
        26.6,         # BMI
        0.351,        # DiabetesPedigreeFunction
        31            # Age
    ]
}

response = requests.post(url, json=data)
print(response.json())
