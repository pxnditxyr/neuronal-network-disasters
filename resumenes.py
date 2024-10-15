import pandas as pd

def summarize_disasters_by_month(csv_file_path):
    # Leer el archivo CSV
    df = pd.read_csv(csv_file_path)

    # Convertir la columna 'date' a tipo datetime
    df['date'] = pd.to_datetime(df['date'])

    # Añadir una columna para el mes
    df['month'] = df['date'].dt.strftime('%B')

    # Agrupar por mes y contar los desastres naturales que no sean "no hubo desastre"
    monthly_summary = df[df['natural-disaster'] != "no hubo desastre"].groupby('month').size().reindex(
        ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
        fill_value=0
    )

    # Generar el resumen en el formato solicitado
    summary = []
    for month, count in monthly_summary.items():
        if count == 0:
            summary.append(f"{month} - no se presentó ningún evento")
        else:
            summary.append(f"{month} - se presentaron {count} desastres en todos los años recopilados")

    # Unir el resumen en una sola cadena para visualizar
    return "\n".join(summary)

# Ejemplo de uso (reemplazar 'ruta_del_archivo.csv' por la ruta real del archivo CSV)
resumen = summarize_disasters_by_month('./dataset/disasters_normalized_names.csv')
print(resumen)
