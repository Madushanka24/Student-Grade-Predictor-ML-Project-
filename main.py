import numpy as np
from sklearn.linear_model import LinearRegression

# Training Data
X = np.array([
    [2, 60, 50],
    [4, 70, 60],
    [6, 80, 70],
    [8, 90, 85],
    [10, 95, 90]
])

y = np.array([50, 60, 70, 85, 95])

model = LinearRegression()
model.fit(X, y)

print("=== Student Grade Predictor ===")

study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance %: "))
assignment = float(input("Enter assignment score: "))

prediction = model.predict([[study_hours, attendance, assignment]])

print("Predicted Final Marks:", round(prediction[0], 2))

# Grade classification
if prediction >= 85:
    print("Grade: A")
elif prediction >= 70:
    print("Grade: B")
elif prediction >= 55:
    print("Grade: C")
else:
    print("Grade: D")