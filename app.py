from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import models
from sklearn.preprocessing import MinMaxScaler
import json

app = Flask(__name__)

# Cargar el modelo entrenado
model = tf.keras.models.load_model('lstm_model_meteorologico.keras')
print( model.summary() )

# Variables globales necesarias
input_window = 30
feature_cols = [
    'temp_min', 'temp_max', 'temp_avg',
    'visibility_avg',
    'wind_min', 'wind_max', 'wind_avg',
    'pressure_min', 'pressure_max', 'pressure_avg',
    'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'year'
]
coordinate_cols = ['latitude', 'longitude']
disaster_cols = [ 'disaster_deslizamiento', 'disaster_granizada', 'disaster_granizada e inundacion',
                  'disaster_derrumbe', 'disaster_crisis del agua (sequia)',
                  'disaster_riada', 'disaster_desborde de rio', 'disaster_inundacion' ]
output_disaster_dim = len(disaster_cols)
output_coordinates_dim = len(coordinate_cols)

def slice_last_window(x):
    return x[:, -output_window:, :]

output_window = 5  # Debe coincidir con el valor usado durante el entrenamiento

# Cargar los escaladores utilizados durante el entrenamiento
# Asegúrate de guardar los escaladores durante el preprocesamiento y cargarlos aquí
# Si no los tienes guardados, tendrás que recrearlos con los parámetros usados durante el entrenamiento

# Por simplicidad, supongamos que tenemos los escaladores guardados en archivos pickle
import pickle

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('coordinate_scaler.pkl', 'rb') as f:
    coordinate_scaler = pickle.load(f)

@app.route('/api/get-predictions/', methods=['POST'])
def get_predictions():
    try:
        # Obtener los datos de entrada en formato JSON
        data = request.get_json()

        # Convertir los datos de entrada a un DataFrame
        input_df = pd.DataFrame(data)

        # Convertir 'date' a datetime
        input_df['date'] = pd.to_datetime(input_df['date'])

        # Calcular 'day_of_week_sin' y 'day_of_week_cos' si no están en los datos
        if 'day_of_week_sin' not in input_df.columns or 'day_of_week_cos' not in input_df.columns:
            input_df['day_of_week'] = input_df['date'].dt.dayofweek
            input_df['day_of_week_sin'] = np.sin(2 * np.pi * input_df['day_of_week'] / 7)
            input_df['day_of_week_cos'] = np.cos(2 * np.pi * input_df['day_of_week'] / 7)
            input_df = input_df.drop(columns=['day_of_week'])

        # Calcular 'month_sin' y 'month_cos' si no están en los datos
        if 'month_sin' not in input_df.columns or 'month_cos' not in input_df.columns:
            input_df['month'] = input_df['date'].dt.month
            input_df['month_sin'] = np.sin(2 * np.pi * input_df['month'] / 12)
            input_df['month_cos'] = np.cos(2 * np.pi * input_df['month'] / 12)
            input_df = input_df.drop(columns=['month'])

        # Verificar que el DataFrame tiene las columnas necesarias
        missing_cols = set(feature_cols) - set(input_df.columns)
        if missing_cols:
            return jsonify({'error': f'Faltan las siguientes columnas: {missing_cols}'}), 400

        # Asegurarse de que hay suficientes datos para la ventana de entrada
        if len(input_df) < input_window:
            return jsonify({'error': f'Se requieren al menos {input_window} registros de datos'}), 400

        # Tomar solo los últimos 'input_window' registros
        input_df = input_df.tail(input_window)

        # Escalar las características
        input_df[feature_cols] = scaler.transform(input_df[feature_cols])

        # Crear la secuencia de entrada
        X_input = input_df[feature_cols].values.reshape(1, input_window, len(feature_cols))

        # Hacer la predicción
        predictions = model.predict(X_input)

        # Obtener las probabilidades de desastres y coordenadas predichas
        disaster_pred_probs = predictions[0][0]  # Forma: (output_window, num_disaster_types)
        coordinates_pred = predictions[1][0]     # Forma: (output_window, 2)

        # Transformar las coordenadas predichas a los valores originales
        coordinates_pred_original = coordinate_scaler.inverse_transform(coordinates_pred)

        # Preparar la respuesta
        response = []
        last_date = input_df['date'].iloc[-1]
        for t in range(output_window):
            pred_date = last_date + pd.Timedelta(days=t+1)
            pred = {
                'date': pred_date.strftime('%Y-%m-%d'),
                'disaster_probabilities': {},
                'predicted_coordinates': {
                    'latitude': float(coordinates_pred_original[t, 0]),
                    'longitude': float(coordinates_pred_original[t, 1])
                }
            }
            for j, disaster_type in enumerate(disaster_cols):
                prob = disaster_pred_probs[t, j]
                pred['disaster_probabilities'][disaster_type.replace('disaster_', '')] = float(prob)
            response.append(pred)

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
