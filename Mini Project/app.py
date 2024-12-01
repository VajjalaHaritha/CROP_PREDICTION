from flask import Flask, render_template, request, url_for
import numpy as np
import joblib

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('crop_prediction_model.joblib')
scaler = joblib.load('scaler.joblib')


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get input values from the form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Make prediction
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)

        return render_template('result.html', prediction=prediction[0])
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
