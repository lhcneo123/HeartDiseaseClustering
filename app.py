from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Cargar el modelo
with open('clustering_embeding.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        try:
            Age = float(request.form['Age'])
            Sex = float(request.form['Sex'])
            ChestPainType = float(request.form['ChestPainType'])
            RestingBP = float(request.form['RestingBP'])
            FastingBS = float(request.form['FastingBS'])
            RestingECG = float(request.form['RestingECG'])
            MaxHR = float(request.form['MaxHR'])
            ExerciseAngina = float(request.form['ExerciseAngina'])
            Oldpeak = float(request.form['Oldpeak'])
            ST_Slope = float(request.form['ST_Slope'])
            Cholesterol= float(request.form['Cholesterol'])

            features = np.array([[
                Age, Sex, ChestPainType, RestingBP, RestingBP, FastingBS,
                RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope, Cholesterol
            ]])
            prediction = model.predict(features)[0]

        except Exception as e:
            prediction = f'Error: {str(e)}'

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
