import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load dataset
data = pd.read_csv("student_dataset.csv")
X = data[["study_hours", "attendance", "assignment"]]
y = data["final_marks"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if model exists
if os.path.exists("student_model.pkl"):
    model = joblib.load("student_model.pkl")
else:
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Save model
    joblib.dump(model, "student_model.pkl")

# Evaluate model
y_pred = model.predict(X_test)
r2 = round(r2_score(y_test, y_pred), 3)
mae = round(mean_absolute_error(y_test, y_pred), 3)

# Create static folder if not exists
if not os.path.exists("static"):
    os.makedirs("static")

# Generate chart: Actual vs Predicted marks
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Marks")
plt.tight_layout()
plt.savefig("static/chart.png")
plt.close()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    grade = None
    if request.method == "POST":
        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])
        assignment = float(request.form["assignment"])

        prediction = model.predict([[study_hours, attendance, assignment]])
        final_marks = round(prediction[0], 2)
        result = final_marks

        if final_marks >= 85:
            grade = "A"
        elif final_marks >= 70:
            grade = "B"
        elif final_marks >= 55:
            grade = "C"
        else:
            grade = "D"

    return render_template(
        "index.html",
        result=result,
        grade=grade,
        model_r2=r2,
        model_mae=mae
    )

if __name__ == "__main__":
    app.run(debug=True)