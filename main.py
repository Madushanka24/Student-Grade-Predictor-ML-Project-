from flask import Flask, render_template, request
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

app = Flask(__name__)

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

        # Grade classification
        if final_marks >= 85:
            grade = "A"
        elif final_marks >= 70:
            grade = "B"
        elif final_marks >= 55:
            grade = "C"
        else:
            grade = "D"

        # Optional: save input to CSV
        data = {"Study Hours": [study_hours],
                "Attendance": [attendance],
                "Assignment": [assignment],
                "Predicted Marks": [final_marks],
                "Grade": [grade]}
        df = pd.DataFrame(data)
        df.to_csv("student_data.csv", mode="a", header=False, index=False)

    return render_template("index.html", result=result, grade=grade)

if __name__ == "__main__":
    app.run(debug=True)