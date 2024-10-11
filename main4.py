import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Cargar los datasets
visibility_df = pd.read_csv('./dataset/interpolated-visibility-data.csv')
wind_df = pd.read_csv('./dataset/interpolated-wind-data.csv')
pressure_df = pd.read_csv('./dataset/interpolated-atmospheric-pressure-data.csv')
temperature_df = pd.read_csv('./dataset/interpolated-temperature-data.csv')
disasters_df = pd.read_csv('./dataset/disasters_normalized_names.csv')

visibility_df.rename(columns={'avg': 'visibility_avg'}, inplace=True)
wind_df.rename(columns={'min': 'wind_min', 'max': 'wind_max', 'avg': 'wind_avg'}, inplace=True)
pressure_df.rename(columns={'min': 'pressure_min', 'max': 'pressure_max', 'avg': 'pressure_avg'}, inplace=True)
temperature_df.rename(columns={'min': 'temp_min', 'max': 'temp_max', 'avg': 'temp_avg'}, inplace=True)

print("Datos de visibilidad:")
print(visibility_df.head())
print("\nDatos de velocidad del viento:")
print(wind_df.head())

# Convertir 'date' a datetime
for df in [visibility_df, wind_df, pressure_df, temperature_df, disasters_df]:
    df['date'] = pd.to_datetime(df['date'])

# Fusionar los datasets meteorológicos
meteorological_df = visibility_df.merge(wind_df, on='date') \
                                .merge(pressure_df, on='date') \
                                .merge(temperature_df, on='date')

print("Datos meteorológicos combinados:")
print(meteorological_df.head())

# Verificar dimensiones
print(f"Dimensiones del DataFrame meteorológico: {meteorological_df.shape}")

# Agrupar desastres por fecha
disasters_grouped = disasters_df.groupby('date').agg({
    'natural-disaster': lambda x: list(x[x != 'no hubo desastre']),
    'address': lambda x: list(x[x != 'no hubo desastre'])
}).reset_index()

print("Datos de desastres agrupados por fecha:")
print(disasters_grouped.head())

# Verificar si hay fechas sin desastres
no_disasters = disasters_grouped['natural-disaster'].apply(len) == 0
print(f"Cantidad de fechas sin desastres: {no_disasters.sum()}")

# Reemplazar listas vacías por "no hubo desastre"
disasters_grouped['natural-disaster'] = disasters_grouped['natural-disaster'].apply(
    lambda x: ['no hubo desastre'] if len(x) == 0 else x
)
disasters_grouped['address'] = disasters_grouped['address'].apply(
    lambda x: [np.nan] if len(x) == 0 else x
)

print("Datos de desastres después del procesamiento:")
print(disasters_grouped.head())

# Fusionar meteorological_df con disasters_grouped
full_df = meteorological_df.merge(disasters_grouped, on='date')

print("Datos completos después de la fusión:")
print(full_df.head())

# Verificar dimensiones
print(f"Dimensiones del DataFrame completo: {full_df.shape}")

# Reemplazar NaN en 'address' por 'Sin dirección'
full_df['address'] = full_df['address'].apply(lambda x: x if isinstance(x, list) else ['Sin dirección'])

print("Datos después de reemplazar valores nulos en 'address':")
print(full_df[['date', 'natural-disaster', 'address']].head())

from sklearn.preprocessing import MultiLabelBinarizer

# Inicializar el binarizador
mlb = MultiLabelBinarizer()

# Aplicar one-hot encoding
disaster_encoded = mlb.fit_transform(full_df['natural-disaster'])

# Crear un DataFrame con las etiquetas
disaster_labels = pd.DataFrame(disaster_encoded, columns=mlb.classes_, index=full_df.index)

print("Etiquetas de desastres después de one-hot encoding:")
print(disaster_labels.head())

# Guardar el binarizador para futuras transformaciones
joblib.dump(mlb, 'mlb_disasters.pkl')

from sklearn.preprocessing import LabelEncoder

# Combinar todas las direcciones en una lista
all_addresses = [addr for sublist in full_df['address'] for addr in sublist]

# Inicializar el LabelEncoder
le = LabelEncoder()
le.fit(all_addresses)

# Función para codificar las direcciones
def encode_addresses(address_list):
    return [le.transform([addr])[0] for addr in address_list]

# Aplicar la codificación
full_df['address_encoded'] = full_df['address'].apply(encode_addresses)

print("Direcciones codificadas:")
print(full_df[['address', 'address_encoded']].head())

# Guardar el LabelEncoder
joblib.dump(le, 'le_address.pkl')

# Seleccionar las columnas numéricas
numerical_features = ['visibility_avg', 'wind_min', 'wind_max', 'wind_avg',
                      'pressure_min', 'pressure_max', 'pressure_avg',
                      'temp_min', 'temp_max', 'temp_avg']

# Inicializar el scaler
scaler = MinMaxScaler()

# Ajustar y transformar los datos
full_df[numerical_features] = scaler.fit_transform(full_df[numerical_features])

print("Datos meteorológicos escalados:")
print(full_df[numerical_features].head())

# Guardar el scaler
joblib.dump(scaler, 'scaler_numerical.pkl')

# Número de días a predecir
forecast_horizon = 5

# Crear etiquetas para los próximos 5 días
for i in range(1, forecast_horizon + 1):
    shifted_labels = disaster_labels.shift(-i).rename(
        columns=lambda x: f"{x}_+{i}"
    )
    full_df = pd.concat([full_df, shifted_labels], axis=1)

print("Datos con etiquetas desplazadas:")
print(full_df[[col for col in full_df.columns if '+1' in col or '+2' in col or '+3' in col or '+4' in col or '+5' in col]].head())

# Eliminar las últimas 5 filas
full_df = full_df[:-forecast_horizon]

print(f"Dimensiones del DataFrame después de eliminar las últimas {forecast_horizon} filas: {full_df.shape}")

# Definir la fecha de corte para entrenamiento y prueba
test_start_date = pd.to_datetime('2024-02-20')
test_end_date = pd.to_datetime('2024-02-25')

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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Definir el número de características y etiquetas
input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

# Construir el modelo
model = Sequential([
    Dense(128, input_dim=input_dim, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(output_dim, activation='sigmoid')
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
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

from sklearn.metrics import classification_report

# Generar reporte de clasificación
report = classification_report(y_test, predicted_labels, target_names=mlb.classes_, zero_division=0)
print("Reporte de Clasificación:")
print(report)
