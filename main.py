from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

app = Flask(__name__)

# Load Dataset
data = pd.read_csv("student_dataset.csv")

X = data[["study_hours", "attendance", "assignment"]]
y = data["final_marks"]

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Model R2 Score:", round(r2, 3))
print("Model MAE:", round(mae, 3))

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    grade = None
    model_r2 = round(r2, 3)
    model_mae = round(mae, 3)

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
        model_r2=model_r2,
        model_mae=model_mae
    )

if __name__ == "__main__":
    app.run(debug=True)