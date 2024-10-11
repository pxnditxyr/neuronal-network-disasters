# Importar las librerías necesarias
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics
from tensorflow.keras.callbacks import EarlyStopping

# Definir la ruta al directorio 'dataset' en tu Google Drive
data_dir = './dataset/'

# Cargar los datasets meteorológicos
temperature_df = pd.read_csv(data_dir + 'interpolated-temperature-data.csv')
visibility_df = pd.read_csv(data_dir + 'interpolated-visibility-data.csv')
wind_df = pd.read_csv(data_dir + 'interpolated-wind-data.csv')
pressure_df = pd.read_csv(data_dir + 'interpolated-atmospheric-pressure-data.csv')

# Cargar el dataset de desastres naturales
disasters_df = pd.read_csv(data_dir + 'disasters_normalized.csv')

# Convertir la columna 'date' a tipo datetime en todos los DataFrames
temperature_df['date'] = pd.to_datetime(temperature_df['date'])
visibility_df['date'] = pd.to_datetime(visibility_df['date'])
wind_df['date'] = pd.to_datetime(wind_df['date'])
pressure_df['date'] = pd.to_datetime(pressure_df['date'])
disasters_df['date'] = pd.to_datetime(disasters_df['date'])

# Renombrar columnas
wind_df = wind_df.rename(columns={'min': 'wind_min', 'max': 'wind_max', 'avg': 'wind_avg'})
visibility_df = visibility_df.rename(columns={'avg': 'visibility_avg'})
temperature_df = temperature_df.rename(columns={'min': 'temp_min', 'max': 'temp_max', 'avg': 'temp_avg'})
pressure_df = pressure_df.rename(columns={'min': 'pressure_min', 'max': 'pressure_max', 'avg': 'pressure_avg'})

# Unir los datasets meteorológicos
weather_df = temperature_df.merge(visibility_df, on='date', how='inner')
weather_df = weather_df.merge(wind_df, on='date', how='inner')
weather_df = weather_df.merge(pressure_df, on='date', how='inner')

# Reemplazar valores vacíos en 'latitude' y 'longitude' por NaN
disasters_df['latitude'] = disasters_df['latitude'].replace('', np.nan)
disasters_df['longitude'] = disasters_df['longitude'].replace('', np.nan)

# Convertir 'latitude' y 'longitude' a tipo numérico
disasters_df['latitude'] = pd.to_numeric(disasters_df['latitude'], errors='coerce')
disasters_df['longitude'] = pd.to_numeric(disasters_df['longitude'], errors='coerce')

# Llenar valores nulos en 'natural-disaster' si los hubiera
disasters_df['natural-disaster'] = disasters_df['natural-disaster'].fillna('no hubo desastre')

# Agrupar los desastres por fecha y consolidar los tipos de desastres y las coordenadas
grouped_disasters = disasters_df.groupby('date').agg({
    'natural-disaster': lambda x: list(x),
    'latitude': lambda x: list(x),
    'longitude': lambda x: list(x)
}).reset_index()

# Crear un conjunto de todos los tipos de desastres, excluyendo 'no hubo desastre'
all_disaster_types = set(disasters_df['natural-disaster'].unique()) - {'no hubo desastre'}

# Función para crear columnas One-Hot para múltiples desastres
def create_disaster_ohe(row):
    disaster_dict = {f'disaster_{disaster}': 0 for disaster in all_disaster_types}
    for disaster in row['natural-disaster']:
        if disaster != 'no hubo desastre':
            disaster_dict[f'disaster_{disaster}'] = 1
    return pd.Series(disaster_dict)

# Aplicar la función al DataFrame agrupado
disaster_ohe = grouped_disasters.apply(create_disaster_ohe, axis=1)

# Combinar las columnas One-Hot con el DataFrame agrupado
grouped_disasters = pd.concat([grouped_disasters, disaster_ohe], axis=1)

# Convertir 'latitude' y 'longitude' de listas a valores escalares (promedio)
def mean_or_nan(x):
    x = [v for v in x if not np.isnan(v)]
    return np.mean(x) if x else np.nan

grouped_disasters['latitude'] = grouped_disasters['latitude'].apply(mean_or_nan)
grouped_disasters['longitude'] = grouped_disasters['longitude'].apply(mean_or_nan)

# Unir weather_df y grouped_disasters en base a 'date'
final_df = weather_df.merge(grouped_disasters, on='date', how='inner')

# Extraer variables temporales
final_df['day_of_week'] = final_df['date'].dt.dayofweek  # Lunes=0, Domingo=6
final_df['month'] = final_df['date'].dt.month
final_df['year'] = final_df['date'].dt.year

# Codificar 'day_of_week' y 'month' de forma cíclica
final_df['day_of_week_sin'] = np.sin(2 * np.pi * final_df['day_of_week'] / 7)
final_df['day_of_week_cos'] = np.cos(2 * np.pi * final_df['day_of_week'] / 7)
final_df['month_sin'] = np.sin(2 * np.pi * final_df['month'] / 12)
final_df['month_cos'] = np.cos(2 * np.pi * final_df['month'] / 12)

# Eliminar las columnas originales 'day_of_week' y 'month'
final_df = final_df.drop(columns=['day_of_week', 'month'])

# Importar MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# Seleccionar las columnas numéricas que queremos normalizar
numeric_cols = [
    'temp_min', 'temp_max', 'temp_avg',
    'visibility_avg',
    'wind_min', 'wind_max', 'wind_avg',
    'pressure_min', 'pressure_max', 'pressure_avg',
    'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'year'
]

# Crear una instancia del escalador
scaler = MinMaxScaler()

# Aplicar el escalado a las columnas seleccionadas
final_df[numeric_cols] = scaler.fit_transform(final_df[numeric_cols])

# Normalizar las coordenadas
coordinate_cols = ['latitude', 'longitude']
coordinate_scaler = MinMaxScaler()
final_df[coordinate_cols] = coordinate_scaler.fit_transform(final_df[coordinate_cols])

# Definir ventanas
input_window = 30  # Ventana de entrada de 30 días
output_window = 5  # Ventana de salida de 5 días

def create_sequences(data, feature_cols, disaster_cols, coordinate_cols, input_window, output_window):
    num_samples = len(data) - input_window - output_window + 1
    num_features = len(feature_cols)
    num_disasters = len(disaster_cols)
    num_coordinates = len(coordinate_cols)

    # Preasignar arrays de NumPy
    X = np.empty((num_samples, input_window, num_features), dtype=np.float32)
    y_disaster = np.empty((num_samples, output_window, num_disasters), dtype=np.float32)
    y_coordinates = np.empty((num_samples, output_window, num_coordinates), dtype=np.float32)

    for i in range(num_samples):
        # Secuencia de entrada
        X[i] = data[feature_cols].iloc[i : i + input_window].values

        # Secuencia de salida para desastres
        y_disaster[i] = data[disaster_cols].iloc[i + input_window : i + input_window + output_window].values

        # Secuencia de salida para coordenadas
        y_coordinates[i] = data[coordinate_cols].iloc[i + input_window : i + input_window + output_window].values

    return X, y_disaster, y_coordinates

# Asegurarnos de que los datos estén ordenados por fecha
final_df = final_df.sort_values('date').reset_index(drop=True)

# Definir las columnas de entrada y salida
feature_cols = [
    'temp_min', 'temp_max', 'temp_avg',
    'visibility_avg',
    'wind_min', 'wind_max', 'wind_avg',
    'pressure_min', 'pressure_max', 'pressure_avg',
    'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'year'
]

# Columnas One-Hot de desastres
disaster_cols = [col for col in final_df.columns if col.startswith('disaster_')]

# Coordenadas
coordinate_cols = ['latitude', 'longitude']

# Crear las secuencias
X, y_disaster, y_coordinates = create_sequences(
    final_df, feature_cols, disaster_cols, coordinate_cols, input_window, output_window
)

# Reemplazar NaN en coordenadas por ceros
y_coordinates = np.nan_to_num(y_coordinates)

# Determinar el índice de división
train_size = int(len(X) * 0.8)

# Dividir X
X_train = X[:train_size]
X_test = X[train_size:]

# Dividir y_disaster
y_disaster_train = y_disaster[:train_size]
y_disaster_test = y_disaster[train_size:]

# Dividir y_coordinates
y_coordinates_train = y_coordinates[:train_size]
y_coordinates_test = y_coordinates[train_size:]


# Definir parámetros del modelo LSTM
input_shape = (input_window, len(feature_cols))
output_disaster_dim = y_disaster.shape[-1]
output_coordinates_dim = y_coordinates.shape[-1]

# Definir entradas
inputs = layers.Input(shape=input_shape)

# Modelo LSTM
x = layers.LSTM(128, return_sequences=True)(inputs)
x = layers.Dropout(0.2)(x)
x = layers.LSTM(64, return_sequences=True)(x)
x = layers.Dropout(0.2)(x)

from keras.saving import register_keras_serializable
from keras import backend as K

@register_keras_serializable()
def slice_last_window(x):
    return x[:, -output_window:, :]

#x = layers.Lambda(lambda x: x[:, -output_window:, :])(x)
x = layers.Lambda(slice_last_window)(x)

# Salida para y_disaster
disaster_output = layers.TimeDistributed(layers.Dense(64, activation='relu'))(x)
disaster_output = layers.TimeDistributed(layers.Dense(output_disaster_dim, activation='sigmoid'), name='disaster_output')(disaster_output)

# Salida para y_coordinates
coordinates_output = layers.TimeDistributed(layers.Dense(64, activation='relu'))(x)
coordinates_output = layers.TimeDistributed(layers.Dense(output_coordinates_dim, activation='linear'), name='coordinates_output')(coordinates_output)

# Definir el modelo con dos salidas
model = models.Model(inputs=inputs, outputs=[disaster_output, coordinates_output])

# Compilar el modelo
# losses = {
#     'disaster_output': 'binary_crossentropy',
#     'coordinates_output': 'mean_squared_error'
# }
losses = {
    'disaster_output': 'binary_crossentropy',       # Para la clasificación de desastres
    'coordinates_output': 'mean_squared_error'     # Para la regresión de coordenadas
}

loss_weights = {
    'disaster_output': 1.0,
    'coordinates_output': 1.0
}

metrics = {
    'disaster_output': [metrics.Precision(), metrics.Recall()],
    'coordinates_output': [metrics.MeanAbsoluteError()]
}

optimizer = optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss=['binary_crossentropy', 'mean_squared_error'],  # Orden: ['disaster_output', 'coordinates_output']
    loss_weights=[1.0, 1.0],
    metrics={
        'disaster_output': [metrics.Precision(), metrics.Recall()],
        'coordinates_output': [metrics.MeanAbsoluteError()]
    }
)

# Definir EarlyStopping para evitar sobreajuste
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("model summary")
print(model.summary())
print("model output names")
print(model.output_names)
print(f'Output Shapes:')
print(f'disaster_output: {y_disaster_train.shape}')      # Debe ser (num_samples, 5, 8)
print(f'coordinates_output: {y_coordinates_train.shape}')  # Debe ser (num_samples, 5, 2)')


# Entrenar el modelo
history = model.fit(
    X_train,
    [y_disaster_train, y_coordinates_train],  # Orden: [y_disaster_train, y_coordinates_train]
    validation_data=(X_test, [y_disaster_test, y_coordinates_test]),
    epochs=50,
    batch_size=64,
    callbacks=[early_stopping]
)
