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
from itertools import product

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

# 5. Filtrar desastres que ocurrieron
disasters_non_null = disasters_df[disasters_df['natural-disaster'] != 'no hubo desastre']

# 6. Definir todas las categorías de desastres naturales
all_disaster_types = [
    'inundacion', 'riada', 'granizada e inundacion',
    'deslizamiento', 'derrumbe', 'granizada',
    'desborde de rio', 'crisis del agua (sequia)'
]

# 7. Realizar one-hot encoding en 'natural-disaster'
one_hot_disasters = pd.get_dummies(disasters_non_null['natural-disaster'], prefix='disaster')

# 8. Definir las columnas esperadas
expected_columns = [f'disaster_{disaster.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")}' for disaster in all_disaster_types]

# 9. Reindexar para incluir todas las columnas esperadas, rellenando con 0 donde falten
one_hot_disasters = one_hot_disasters.reindex(columns=expected_columns, fill_value=0)

# 10. Crear todas las combinaciones de fecha y dirección
all_addresses = disasters_df['address'].dropna().unique().tolist()
all_dates = weather_df['date'].unique()
date_address_combinations = pd.DataFrame(list(product(all_dates, all_addresses)), columns=['date', 'address'])

print(f"Direcciones únicas encontradas: {len(all_addresses)}")
print(all_addresses[:10])

# 11. Fusionar combinaciones con desastres para asignar desastres a cada fecha-dirección
merged_disasters = pd.merge(date_address_combinations, disasters_non_null, on=['date', 'address'], how='left')

# 12. Rellenar los desastres ausentes con 'no hubo desastre'
merged_disasters['natural-disaster'] = merged_disasters['natural-disaster'].fillna('no hubo desastre')

print("\nDatos de desastres por fecha y dirección:")
print(merged_disasters.head())

# 13. Agregar las columnas one-hot al DataFrame de desastres
merged_disasters = pd.concat([merged_disasters[['date', 'address']], one_hot_disasters], axis=1)

print("\nDatos de desastres codificados (con todas las categorías):")
print(merged_disasters.head())

# 14. Codificar las direcciones
address_encoder = LabelEncoder()
merged_disasters['address_encoded'] = address_encoder.fit_transform(merged_disasters['address'])

# Guardar el codificador para uso futuro
joblib.dump(address_encoder, 'address_encoder.joblib')

print("\nDirecciones codificadas:")
print(merged_disasters[['address', 'address_encoded']].head())

# 15. Fusionar con los datos meteorológicos
combined_df = pd.merge(date_address_combinations, weather_df, on='date', how='left')

# 16. Fusionar con los datos de desastres codificados
combined_df = pd.merge(combined_df, merged_disasters, on=['date', 'address'], how='left')

# 17. Reemplazar NaN con 0 para las columnas de desastres
for disaster in expected_columns:
    combined_df[disaster] = combined_df[disaster].fillna(0)

# 18. Rellenar 'address_encoded' con 0 si es necesario
combined_df['address_encoded'] = combined_df['address_encoded'].fillna(0).astype(int)

print("\nDatos combinados (meteorología + desastres por dirección):")
print(combined_df.head())

# 19. Definir las columnas de características
feature_columns = [
    'visibility_avg', 'wind_min', 'wind_max', 'wind_avg',
    'pressure_min', 'pressure_max', 'pressure_avg',
    'temp_min', 'temp_max', 'temp_avg',
    'address_encoded'  # Incluir la dirección codificada como característica
]

# 20. Definir las columnas de etiquetas
label_columns = expected_columns  # ['disaster_inundacion', 'disaster_riada', ...]

print("\nCaracterísticas seleccionadas:")
print(feature_columns)

print("\nEtiquetas seleccionadas:")
print(label_columns)

# 21. Ordenar el DataFrame combinado por fecha
combined_df = combined_df.sort_values('date')

# 22. Definir la fecha de corte para el conjunto de prueba (enero de 2024)
test_start_date = '2024-01-01'

# 23. Dividir los datos
train_df = combined_df[combined_df['date'] < test_start_date]
test_df = combined_df[combined_df['date'] >= test_start_date]

print("\nTamaño del conjunto de entrenamiento:", train_df.shape)
print("Tamaño del conjunto de prueba:", test_df.shape)

# 24. Normalizar las características
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

# 25. Extraer etiquetas
train_labels = train_df[label_columns].values
test_labels = test_df[label_columns].values

print("\nEtiquetas del conjunto de entrenamiento:")
print(train_labels[:5])

print("\nEtiquetas del conjunto de prueba:")
print(test_labels[:5])

# 26. Convertir etiquetas a float
train_labels = train_labels.astype(float)
test_labels = test_labels.astype(float)

# 27. Construir el modelo
model = Sequential([
    Dense(128, input_dim=len(feature_columns), activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(label_columns), activation='sigmoid')  # Sigmoid para multi-etiqueta
])

# Resumen del modelo
print("\nResumen del modelo:")
model.summary()

# 28. Compilar el modelo
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 29. Definir EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 30. Entrenar el modelo
history = model.fit(
    train_features,
    train_labels,
    epochs=100,
    batch_size=64,
    validation_split=0.2,  # 20% para validación
    callbacks=[early_stop],
    verbose=1
)

# 31. Evaluar el modelo
loss, accuracy = model.evaluate(test_features, test_labels, verbose=0)
print(f"\nPérdida en el conjunto de prueba: {loss}")
print(f"Precisión en el conjunto de prueba: {accuracy}")

# 32. Graficar las métricas de entrenamiento
plt.figure(figsize=(14,6))

# Pérdida
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida del Modelo')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

# Precisión
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del Modelo')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.tight_layout()
plt.show()

# 33. Preparar datos para predicción
# Seleccionar las últimas 5 fechas de entrenamiento para generar predicciones
last_date_train = train_df['date'].max()
prediction_dates = pd.date_range(start=last_date_train + pd.Timedelta(days=1), periods=5)

# Seleccionar todas las direcciones para las fechas de predicción
prediction_combinations = pd.DataFrame(list(product(prediction_dates, all_addresses)), columns=['date', 'address'])

# Realizar las mismas transformaciones que antes
# One-hot encoding de desastres (en este caso, inicialmente sin desastres)
for disaster in expected_columns:
    prediction_combinations[disaster] = 0

# Codificar las direcciones
prediction_combinations['address_encoded'] = address_encoder.transform(prediction_combinations['address'])

# Fusionar con los datos meteorológicos
# Asumimos que tienes datos meteorológicos para las fechas de predicción
# Si no los tienes, deberás obtenerlos o realizar una predicción meteorológica previa
# Aquí, usaremos los últimos registros disponibles para simular
# (Reemplaza esta parte con los datos meteorológicos reales de las fechas de predicción)
last_meteorological_data = weather_df[weather_df['date'] == last_date_train].iloc[0]
meteorological_predictions = pd.DataFrame([last_meteorological_data] * len(prediction_combinations))
meteorological_predictions = meteorological_predictions.reset_index(drop=True)

# Combinar con las combinaciones de fecha y dirección
prediction_input = pd.concat([prediction_combinations, meteorological_predictions], axis=1)

# Seleccionar las características
prediction_features = prediction_input[feature_columns]

# Escalar las características
prediction_features_scaled = scaler.transform(prediction_features)

# Convertir a DataFrame
prediction_features_scaled = pd.DataFrame(prediction_features_scaled, columns=feature_columns)

print("\nCaracterísticas para predicciones (escaladas):")
print(prediction_features_scaled.head())

# 34. Realizar predicciones
predictions = model.predict(prediction_features_scaled)
predictions_df = pd.DataFrame(predictions, columns=label_columns)
predictions_df['date'] = prediction_combinations['date'].values
predictions_df['address'] = prediction_combinations['address'].values

print("\nPredicciones de probabilidad para las combinaciones fecha-dirección:")
print(predictions_df.head())

# 35. Preparar datos para la gráfica
predictions_melted = predictions_df.melt(
    id_vars=['date', 'address'],
    value_vars=label_columns,
    var_name='Desastre',
    value_name='Probabilidad'
)

# Reemplazar los nombres de desastres para que sean más legibles
predictions_melted['Desastre'] = predictions_melted['Desastre'].str.replace('disaster_', '').str.replace('_', ' ').str.title()

# Opcional: Filtrar solo las predicciones con cierta probabilidad
predictions_melted = predictions_melted[predictions_melted['Probabilidad'] > 0.01]

print("\nDatos preparados para la gráfica con direcciones:")
print(predictions_melted.head())

# 36. Graficar las probabilidades
plt.figure(figsize=(14, 8))
sns.lineplot(
    data=predictions_melted,
    x='date',
    y='Probabilidad',
    hue='Desastre',
    style='address',
    markers=True,
    dashes=False
)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.xticks(rotation=45)
plt.title('Probabilidades de Desastres Naturales por Dirección en los Próximos 5 Días')
plt.xlabel('Fecha')
plt.ylabel('Probabilidad')
plt.legend(title='Tipo de Desastre')
plt.tight_layout()
plt.show()

# 37. Guardar el modelo
model.save('modelo_prediccion_desastres_por_direccion.h5')
print("\nModelo guardado exitosamente como 'modelo_prediccion_desastres_por_direccion.h5'")
