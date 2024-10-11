import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras

# Cargar los datasets
visibility_df = pd.read_csv('./dataset/interpolated-visibility-data.csv')
wind_df = pd.read_csv('./dataset/interpolated-wind-data.csv')
pressure_df = pd.read_csv('./dataset/interpolated-atmospheric-pressure-data.csv')
temperature_df = pd.read_csv('./dataset/interpolated-temperature-data.csv')
disasters_df = pd.read_csv('./dataset/disasters_normalized_names.csv')

print("Datos de visibilidad:")
print(visibility_df.head())
print("\nDatos de velocidad del viento:")
print(wind_df.head())

# Convertir 'date' a datetime
for df in [visibility_df, wind_df, pressure_df, temperature_df, disasters_df]:
    df['date'] = pd.to_datetime(df['date'])

# Renombrar columnas
visibility_df.rename(columns={'avg': 'visibility_avg'}, inplace=True)
wind_df.rename(columns={'min': 'wind_min', 'max': 'wind_max', 'avg': 'wind_avg'}, inplace=True)
pressure_df.rename(columns={'min': 'pressure_min', 'max': 'pressure_max', 'avg': 'pressure_avg'}, inplace=True)
temperature_df.rename(columns={'min': 'temp_min', 'max': 'temp_max', 'avg': 'temp_avg'}, inplace=True)


print("Columnas después de renombrar:")
print(visibility_df.columns)

# Fusionar datasets meteorológicos
weather_df = visibility_df.merge(wind_df, on='date', how='inner')
weather_df = weather_df.merge(pressure_df, on='date', how='inner')
weather_df = weather_df.merge(temperature_df, on='date', how='inner')

print("Datos meteorológicos fusionados:")
print(weather_df.head())

# Crear una codificación one-hot de los desastres por fecha
disasters_onehot = disasters_df.pivot_table(index='date', columns='natural-disaster', aggfunc='size', fill_value=0)

# Restablecer el índice para fusionar más adelante
disasters_onehot.reset_index(inplace=True)

print("Datos de desastres naturales (one-hot):")
print(disasters_onehot.head())

# Fusionar datasets
data_df = weather_df.merge(disasters_onehot, on='date', how='left')

# Rellenar NaN con 0
data_df.fillna(0, inplace=True)

print("DataFrame final después de fusionar:")
print(data_df.head())

# Características numéricas
feature_cols = ['visibility_avg', 'wind_min', 'wind_max', 'wind_avg',
                'pressure_min', 'pressure_max', 'pressure_avg',
                'temp_min', 'temp_max', 'temp_avg']

# Etiquetas (nombres de desastres)
label_cols = disasters_onehot.columns.drop('date').tolist()

# Inicializar el escalador
scaler = MinMaxScaler()

# Ajustar y transformar las características
data_df[feature_cols] = scaler.fit_transform(data_df[feature_cols])

print("Características normalizadas:")
print(data_df[feature_cols].head())

# Fecha de corte para entrenamiento (antes de enero de 2024)
cutoff_date = pd.to_datetime('2024-01-01')

# Datos de entrenamiento
train_data = data_df[data_df['date'] < cutoff_date]

# Datos de prueba
test_data = data_df[data_df['date'] >= cutoff_date]

# Características y etiquetas de entrenamiento
X_train = train_data[feature_cols]
y_train = train_data[label_cols]

# Características y etiquetas de prueba
X_test = test_data[feature_cols]
y_test = test_data[label_cols]

print("Forma de X_train:", X_train.shape)
print("Forma de y_train:", y_train.shape)


# Definir el modelo
model = keras.models.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(len(label_cols), activation='sigmoid')  # Salida para múltiples etiquetas
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# resumen
print( "Resumen del modelo:" )
print( model.summary() )

# Entrenar el modelo
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Pérdida en prueba: {loss}')
print(f'Precisión en prueba: {accuracy}')

# Supongamos que 'future_weather_df' contiene los datos meteorológicos futuros
# Por ahora, duplicaremos el último registro para simular datos futuros

last_weather = weather_df.tail(1).copy()
future_dates = [weather_df['date'].max() + timedelta(days=i) for i in range(1, 6)]
future_weather_df = pd.concat([last_weather]*5, ignore_index=True)
future_weather_df['date'] = future_dates

# Normalizar las características
future_weather_df[feature_cols] = scaler.transform(future_weather_df[feature_cols])

print("Datos meteorológicos futuros:")
print(future_weather_df)

# Predicciones
future_X = future_weather_df[feature_cols]
predictions = model.predict(future_X)

# Convertir las predicciones a un DataFrame
predictions_df = pd.DataFrame( predictions, columns=label_cols )
predictions_df['date'] = future_weather_df['date'].values

print("Predicciones para los próximos 5 días:")
print(predictions_df)

# Guardar el modelo
model.save('disaster_prediction_model.keras')

# Cargar el modelo
loaded_model = keras.models.load_model('disaster_prediction_model.keras')

# Evaluar el modelo con el mismo conjunto de datos
loss, accuracy = loaded_model.evaluate(X_test, y_test)

