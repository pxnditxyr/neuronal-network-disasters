import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
import joblib
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K

# Cargar los datasets
visibility_df = pd.read_csv('./dataset/interpolated-visibility-data.csv')
wind_df = pd.read_csv('./dataset/interpolated-wind-data.csv')
pressure_df = pd.read_csv('./dataset/interpolated-atmospheric-pressure-data.csv')
temperature_df = pd.read_csv('./dataset/interpolated-temperature-data.csv')
disasters_df = pd.read_csv('./dataset/disasters_normalized_names.csv')

# Renombrar columnas
visibility_df.rename(columns={'avg': 'visibility_avg'}, inplace=True)
wind_df.rename(columns={'min': 'wind_min', 'max': 'wind_max', 'avg': 'wind_avg'}, inplace=True)
pressure_df.rename(columns={'min': 'pressure_min', 'max': 'pressure_max', 'avg': 'pressure_avg'}, inplace=True)
temperature_df.rename(columns={'min': 'temp_min', 'max': 'temp_max', 'avg': 'temp_avg'}, inplace=True)

# Convertir 'date' a datetime
for df in [visibility_df, wind_df, pressure_df, temperature_df, disasters_df]:
    df['date'] = pd.to_datetime(df['date'])

# Fusionar los datasets meteorológicos
meteorological_df = visibility_df.merge(wind_df, on='date') \
                                .merge(pressure_df, on='date') \
                                .merge(temperature_df, on='date')

# Agrupar desastres por fecha
disasters_grouped = disasters_df.groupby('date').agg({
    'natural-disaster': lambda x: list(x[x != 'no hubo desastre']),
    'address': lambda x: list(x[x != 'no hubo desastre'])
}).reset_index()

# Reemplazar listas vacías por "no hubo desastre" y "Sin dirección"
disasters_grouped['natural-disaster'] = disasters_grouped['natural-disaster'].apply(
    lambda x: ['no hubo desastre'] if len(x) == 0 else x
)
disasters_grouped['address'] = disasters_grouped['address'].apply(
    lambda x: ['Sin dirección'] if len(x) == 0 else x
)

# Fusionar meteorological_df con disasters_grouped
full_df = meteorological_df.merge(disasters_grouped, on='date')

# One-Hot Encoding para 'natural-disaster'
mlb = MultiLabelBinarizer()
disaster_encoded = mlb.fit_transform(full_df['natural-disaster'])
disaster_labels = pd.DataFrame(disaster_encoded, columns=mlb.classes_, index=full_df.index)

# Guardar el binarizador
joblib.dump(mlb, 'mlb_disasters.pkl')

# Label Encoding para 'address'
all_addresses = [addr for sublist in full_df['address'] for addr in sublist]
le = LabelEncoder()
le.fit(all_addresses)

def encode_addresses(address_list):
    return [le.transform([addr])[0] for addr in address_list]

full_df['address_encoded'] = full_df['address'].apply(encode_addresses)

# Guardar el LabelEncoder
joblib.dump(le, 'le_address.pkl')

# Escalar las variables numéricas
numerical_features = ['visibility_avg', 'wind_min', 'wind_max', 'wind_avg',
                      'pressure_min', 'pressure_max', 'pressure_avg',
                      'temp_min', 'temp_max', 'temp_avg']

scaler = MinMaxScaler()
full_df[numerical_features] = scaler.fit_transform(full_df[numerical_features])

# Guardar el scaler
joblib.dump(scaler, 'scaler_numerical.pkl')

# Crear etiquetas desplazadas para los próximos 5 días
forecast_horizon = 5
for i in range(1, forecast_horizon + 1):
    shifted_labels = disaster_labels.shift(-i).rename(
        columns=lambda x: f"{x}_+{i}"
    )
    full_df = pd.concat([full_df, shifted_labels], axis=1)

# Eliminar las últimas 5 filas con etiquetas NaN
full_df = full_df[:-forecast_horizon]

# Definir la fecha de corte para entrenamiento y prueba
# Aumentamos el tamaño del conjunto de prueba
test_start_date = pd.to_datetime('2024-02-20')
test_end_date = pd.to_datetime('2024-03-20')  # 30 días de prueba

# Crear el conjunto de prueba
test_df = full_df[(full_df['date'] >= test_start_date) & (full_df['date'] <= test_end_date)]

# Crear el conjunto de entrenamiento
train_df = full_df[full_df['date'] < test_start_date]

print(f"Dimensiones del conjunto de entrenamiento: {train_df.shape}")
print(f"Dimensiones del conjunto de prueba: {test_df.shape}")

# Definir las características (features)
feature_columns = numerical_features  # Puedes agregar más features si lo deseas

X_train = train_df[feature_columns].values
X_test = test_df[feature_columns].values

# Definir las etiquetas para los próximos 5 días
label_columns = [col for col in full_df.columns if '+1' in col or '+2' in col or '+3' in col or '+4' in col or '+5' in col]
y_train = train_df[label_columns].values
y_test = test_df[label_columns].values

print(f"Forma de X_train: {X_train.shape}")
print(f"Forma de y_train: {y_train.shape}")
print(f"Forma de X_test: {X_test.shape}")
print(f"Forma de y_test: {y_test.shape}")

# Definir Focal Loss
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        weight = alpha * y_true * K.pow((1 - y_pred), gamma)
        loss = weight * cross_entropy
        return K.mean(K.sum(loss, axis=1))
    return focal_loss_fixed

# Construir el modelo
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_train.shape[1], activation='sigmoid')
])

# Compilar el modelo con Focal Loss
model.compile(optimizer='adam',
              loss=focal_loss(gamma=2., alpha=.25),
              metrics=['accuracy'])

model.summary()

# Definir el callback de EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluar en el conjunto de prueba
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Precisión en el conjunto de prueba: {accuracy:.4f}")
print(f"Pérdida en el conjunto de prueba: {loss:.4f}")

# Realizar predicciones
predictions = model.predict(X_test)

# Aplicar un umbral para determinar la presencia de desastres
threshold = 0.5
predicted_labels = (predictions >= threshold).astype(int)

# Convertir las predicciones a DataFrame
predicted_df = pd.DataFrame(predicted_labels, columns=label_columns)
predicted_df['date'] = test_df['date'].values

print("Predicciones para la semana de prueba:")
print(predicted_df)

# Generar reportes de clasificación por día de pronóstico
forecast_horizon = 5  # Asegurarse de que coincida con el número de días de pronóstico

for i in range(1, forecast_horizon + 1):
    # Seleccionar las columnas correspondientes al día i
    day_label_cols = [f"{cls}_+{i}" for cls in mlb.classes_]
    
    # Extraer las etiquetas verdaderas y predichas para el día i
    y_true_day = y_test[:, (i-1)*len(mlb.classes_):i*len(mlb.classes_)]
    y_pred_day = predicted_labels[:, (i-1)*len(mlb.classes_):i*len(mlb.classes_)]
    
    print(f"\nReporte de Clasificación para el día +{i}:")
    report = classification_report(y_true_day, y_pred_day, target_names=mlb.classes_, zero_division=0)
    print(report)
