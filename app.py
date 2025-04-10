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
            # Obtener los valores del formulario
            Age = float(request.form['Age'])
            RestingBP = float(request.form['RestingBP'])
            Cholesterol = float(request.form['Cholesterol'])
            FastingBS = int(request.form['FastingBS'])
            MaxHR = float(request.form['MaxHR'])
            Oldpeak = float(request.form['Oldpeak'])

            # Variables categóricas como string
            Sex = request.form['Sex']  # 'M' o 'F'
            ChestPainType = request.form['ChestPainType']  # 'ATA', 'NAP', 'ASY', 'TA'
            RestingECG = request.form['RestingECG']  # 'Normal', 'ST', 'LVH'
            ExerciseAngina = request.form['ExerciseAngina']  # 'Y' o 'N'
            ST_Slope = request.form['ST_Slope']  # 'Up', 'Flat', 'Down'

            # Crear diccionario de entrada
            input_dict = {
                'Age': Age,
                'RestingBP': RestingBP,
                'Cholesterol': Cholesterol,
                'FastingBS': FastingBS,
                'MaxHR': MaxHR,
                'Oldpeak': Oldpeak,
                'Sex': Sex,
                'ChestPainType': ChestPainType,
                'RestingECG': RestingECG,
                'ExerciseAngina': ExerciseAngina,
                'ST_Slope': ST_Slope
            }

            # Columnas que el modelo espera
            columnas_modelo = [
                'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
                'Sex_F', 'Sex_M',
                'ChestPainType_ASY', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
                'RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_ST',
                'ExerciseAngina_N', 'ExerciseAngina_Y',
                'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up'
            ]

            # Convertir a DataFrame y hacer one-hot encoding
            df_input = pd.DataFrame([input_dict])
            df_encoded = pd.get_dummies(df_input)

            # Asegurar que todas las columnas estén presentes
            for col in columnas_modelo:
                if col not in df_encoded:
                    df_encoded[col] = 0

            # Ordenar columnas
            df_encoded = df_encoded[columnas_modelo]

            # Escalar como en el entrenamiento
            input_scaled = scaler.transform(df_encoded)

            # Hacer predicción
            prediction = model.predict(input_scaled)[0]

        except Exception as e:
            prediction = f'Error: {str(e)}'

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
