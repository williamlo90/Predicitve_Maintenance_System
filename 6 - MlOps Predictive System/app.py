from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import input_data, Pred_Pipeline

# Inisialisasi aplikasi Flask
application = Flask(__name__)
app = application

# Halaman landing / beranda
@app.route('/')
def index():
    return render_template('index.html')

# Halaman prediksi data mesin
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Ambil data input dari form HTML
        data = input_data(
            Type=request.form.get('Type'),
            Air_temperature=float(request.form.get('Air_temperature_K')),
            Process_temperature=float(request.form.get('Process_temperature_K')),
            Rotational_speed=request.form.get('Rotational_speed_rpm'),
            Torque=float(request.form.get('Torque_Nm')),
            Tool_wear=request.form.get('Tool_wear_min')
        )

        # Transformasikan data menjadi DataFrame
        pred_data = data.transfrom_data_as_dataframe()
        print("📦 Data untuk prediksi:\n", pred_data)

        # Jalankan prediksi
        predict_pipeline = Pred_Pipeline()
        print("🔍 Melakukan prediksi...")
        results = predict_pipeline.predict(pred_data)
        print("✅ Prediksi selesai.")

        # Interpretasi hasil prediksi
        if results[0] == 1:
            message = "⚠️ There are high chances of machine failure soon. Immediate attention required."
        else:
            message = "✅ No chances of machine failure. The machine is performing well."

        return render_template('home.html', results=message)

# Jalankan server Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
