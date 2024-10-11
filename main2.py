import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import joblib
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from functools import reduce
import seaborn as sns
import matplotlib.dates as mdates

# 1. Cargar los datasets
visibility_df = pd.read_csv('./dataset/interpolated-visibility-data.csv')
wind_df = pd.read_csv('./dataset/interpolated-wind-data.csv')
pressure_df = pd.read_csv('./dataset/interpolated-atmospheric-pressure-data.csv')
temperature_df = pd.read_csv('./dataset/interpolated-temperature-data.csv')
disasters_df = pd.read_csv('./dataset/disasters_normalized_names.csv')

# 2. Renombrar columnas para evitar conflictos
visibility_df.rename(columns={'avg': 'visibility_avg'}, inplace=True)
wind_df.rename(columns={'min': 'wind_min', 'max': 'wind_max', 'avg': 'wind_avg'}, inplace=True)
pressure_df.rename(columns={'min': 'pressure_min', 'max': 'pressure_max', 'avg': 'pressure_avg'}, inplace=True)
temperature_df.rename(columns={'min': 'temp_min', 'max': 'temp_max', 'avg': 'temp_avg'}, inplace=True)

# 3. Convertir 'date' a datetime
for df in [visibility_df, wind_df, pressure_df, temperature_df, disasters_df]:
    df['date'] = pd.to_datetime(df['date'])

# 4. Fusionar los datasets meteorológicos
data_frames = [visibility_df, wind_df, pressure_df, temperature_df]
weather_df = reduce(lambda left, right: pd.merge(left, right, on='date'), data_frames)

print("\nDatos meteorológicos combinados:")
print(weather_df.head())

# 5. Filtrar los días con desastres
disasters_non_null = disasters_df[disasters_df['natural-disaster'] != 'no hubo desastre']

# 6. Definir todas las categorías de desastres naturales
all_disaster_types = [
    'inundacion', 'riada', 'granizada e inundacion',
    'deslizamiento', 'derrumbe', 'granizada',
    'desborde de rio', 'crisis del agua (sequia)'
]

# 7. One-hot encoding en 'natural-disaster'
one_hot_disasters = pd.get_dummies(disasters_non_null['natural-disaster'], prefix='disaster')

# 8. Definir las columnas esperadas
expected_columns = [f'disaster_{disaster.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")}' for disaster in all_disaster_types]

# 9. Reindexar para incluir todas las columnas esperadas, rellenando con 0 donde falten
one_hot_disasters = one_hot_disasters.reindex(columns=expected_columns, fill_value=0)

# 10. Agregar las columnas one-hot al DataFrame de desastres
disasters_encoded = pd.concat([disasters_non_null[['date', 'address']], one_hot_disasters], axis=1)

print("\nDatos de desastres codificados (con todas las categorías):")
print(disasters_encoded.head())
df_normalized = pd.read_csv('./dataset/interpolated-normalized.csv')

# 11. Codificar las direcciones
# Reemplazar valores nulos en 'address' con una categoría específica
disasters_encoded['address'] = disasters_encoded['address'].fillna('no_address')

# Inicializar el codificador
address_encoder = LabelEncoder()
disasters_encoded['address_encoded'] = address_encoder.fit_transform(disasters_encoded['address'])

# Guardar el codificador para uso futuro
joblib.dump(address_encoder, 'address_encoder.joblib')

print("\nDirecciones codificadas:")
print(disasters_encoded[['address', 'address_encoded']].head())

# 12. Agregar las columnas one-hot a nivel diario
disasters_daily = disasters_encoded.groupby('date').sum().reset_index()

# 13. Fusionar con el DataFrame meteorológico
combined_df = pd.merge(weather_df, disasters_daily, on='date', how='left')

# 14. Reemplazar NaN con 0 para los días sin desastres
combined_df.fillna(0, inplace=True)

print("\nDatos combinados (meteorología + desastres):")
print(combined_df.head())

# 15. Definir las columnas de características
feature_columns = [
    'visibility_avg', 'wind_min', 'wind_max', 'wind_avg',
    'pressure_min', 'pressure_max', 'pressure_avg',
    'temp_min', 'temp_max', 'temp_avg'
]

# 16. Definir las columnas de etiquetas
label_columns = [f'disaster_{disaster.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")}' for disaster in all_disaster_types]

print("\nCaracterísticas seleccionadas:")
print(feature_columns)

print("\nEtiquetas seleccionadas:")
print(label_columns)

# 17. Verificar que todas las etiquetas están presentes
missing_labels = [label for label in label_columns if label not in combined_df.columns]
print("\nEtiquetas faltantes (deberían ser añadidas):", missing_labels)

# 18. Si hay etiquetas faltantes, añadirlas con valor 0
for label in missing_labels:
    combined_df[label] = 0

print("\nColumnas después de asegurar todas las etiquetas:")
print(combined_df.columns)

# 19. Ordenar el DataFrame combinado por fecha
combined_df = combined_df.sort_values('date')

# 20. Definir la fecha de corte para el conjunto de prueba (enero de 2024)
test_start_date = '2024-01-01'
train_df = combined_df[combined_df['date'] < test_start_date]
test_df = combined_df[combined_df['date'] >= test_start_date]
# imprimir cuales son las entradas necesarias para el modelo avoid truncated columns

print("\nTamaño del conjunto de entrenamiento:", train_df.shape)
print("Tamaño del conjunto de prueba:", test_df.shape)

# 21. Normalizar las características
scaler = MinMaxScaler()

# Ajustar el scaler en el conjunto de entrenamiento y transformar ambos conjuntos
train_features = scaler.fit_transform(train_df[feature_columns])
test_features = scaler.transform(test_df[feature_columns])

# Convertir de nuevo a DataFrame para facilidad
train_features = pd.DataFrame(train_features, columns=feature_columns, index=train_df.index)
test_features = pd.DataFrame(test_features, columns=feature_columns, index=test_df.index)

print("\nCaracterísticas escaladas (conjunto de entrenamiento):")
print(train_features.head())

print("\nCaracterísticas escaladas (conjunto de prueba):")
print(test_features.head())

# Guardar el scaler para uso futuro
joblib.dump(scaler, 'scaler.joblib')

# 22. Extraer etiquetas
train_labels = train_df[label_columns].values
test_labels = test_df[label_columns].values

print("\nEtiquetas del conjunto de entrenamiento:")
print(train_labels[:5])

print("\nEtiquetas del conjunto de prueba:")
print(test_labels[:5])

# 23. Construir el modelo
model = Sequential([
    Dense(64, input_dim=len(feature_columns), activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(len(label_columns), activation='sigmoid')  # Sigmoid para multi-etiqueta
])

# Resumen del modelo
print("\nResumen del modelo:")
model.summary()

# 24. Compilar el modelo
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 25. Definir EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 26. Entrenar el modelo
history = model.fit(
    train_features,
    train_labels,
    epochs=100,
    batch_size=32,
    validation_split=0.2,  # 20% para validación
    callbacks=[early_stop],
    verbose=1
)

# 27. Evaluar el modelo
loss, accuracy = model.evaluate(test_features, test_labels, verbose=0)

print(f"\nPérdida en el conjunto de prueba: {loss}")
print(f"Precisión en el conjunto de prueba: {accuracy}")

# 28. Graficar las métricas de entrenamiento
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida del Modelo')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del Modelo')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.tight_layout()
plt.show()

# 29. Preparar datos para predicción
last_5_days = combined_df.tail(5)
last_5_features = last_5_days[feature_columns]
last_5_features_scaled = scaler.transform(last_5_features)
last_5_features_scaled = pd.DataFrame(last_5_features_scaled, columns=feature_columns)

print("\nCaracterísticas de los últimos 5 días (escaladas):")
print(last_5_features_scaled)

# 30. Realizar predicciones
predictions = model.predict(last_5_features_scaled)

predictions_df = pd.DataFrame(predictions, columns=label_columns)
predictions_df['date'] = last_5_days['date'].values

print("\nPredicciones de probabilidad para los últimos 5 días:")
print(predictions_df)

# 31. Preparar datos para la gráfica
predictions_melted = predictions_df.melt(id_vars=['date'], value_vars=label_columns,
                                         var_name='Desastre', value_name='Probabilidad')
predictions_melted['Desastre'] = predictions_melted['Desastre'].str.replace('disaster_', '').str.replace('_', ' ').str.title()

print("\nDatos preparados para la gráfica:")
print(predictions_melted.head())

def normalize_date(df_normalized, last_date, norm_length=5):
    df_normalized['date'] = pd.to_datetime(df_normalized['date'])
    last_date = pd.to_datetime(last_date)

    filtered_rows = df_normalized[df_normalized['date'] > last_date]

    filtered_rows = filtered_rows.head(norm_length)

    normalized_list = []
    for _, row in filtered_rows.iterrows():
        row_dict = row.to_dict()
        row_dict['date'] = row_dict['date'].strftime('%Y-%m-%d')
        normalized_list.append(row_dict)

    return normalized_list

# 32. Graficar las probabilidades
plt.figure(figsize=(14, 8))
sns.lineplot(data=predictions_melted, x='date', y='Probabilidad', hue='Desastre', marker='o')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.xticks(rotation=45)
plt.title('Probabilidades de Desastres Naturales en los Próximos 5 Días')
plt.xlabel('Fecha')
plt.ylabel('Probabilidad')
plt.legend(title='Tipo de Desastre')
plt.tight_layout()
plt.show()

# 33. Guardar el modelo
model.save('modelo_prediccion_desastres.h5')
print("\nModelo guardado exitosamente como 'modelo_prediccion_desastres.h5'")
print(f"Forma de entrada esperada por el modelo: {model.input_shape}")
print( "feature_columns", feature_columns )

'''
Servidor Backend
'''

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from tensorflow.keras.models import load_model

# Inicializar la aplicación Flask
app = Flask(__name__)
cors = CORS(app)

# Cargar el scaler y el modelo guardados
try:
    scaler = joblib.load('scaler.joblib')
    model = load_model('modelo_prediccion_desastres.h5')
    print("Escalador y modelo cargados exitosamente.")
except Exception as e:
    print(f"Error al cargar el scaler o el modelo: {e}")


label_columns = [f'disaster_{disaster.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")}' for disaster in all_disaster_types]
print( 'hasta aqui' )

@app.route('/api/get-predictions', methods=['POST'])
def get_predictions():
    data = request.get_json()

    if not isinstance(data, list):
        return jsonify({'error': 'Input should be a list of objects'}), 400

    for item in data:
        if 'date' not in item:
            return jsonify({'error': 'Each object must contain a date'}), 400

    input_df = pd.DataFrame(data)

    response = []
    get_last_date =  pd.to_datetime(input_df['date'].iloc[-1]).strftime('%Y-%m-%d')

    normalized_list = normalize_date(df_normalized, get_last_date)

    for i, pred in enumerate([1,2,3,4,5]):
        # pred_dict = {label: prob for label, prob in zip(label_columns, pred)}
        response.append(normalized_list[i])

    return jsonify({'predictions': response}), 200
