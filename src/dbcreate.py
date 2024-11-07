import pandas as pd
import os
import json
import re

# ----------------------------------------------------------------
# Añadir los archivos de test a la DB
# ----------------------------------------------------------------

# Paso 1: Define la ruta de tu archivo txt
file_path = r'C:\Users\alvar\OneDrive\Escritorio\TFG\ADReSS-IS2020\ADReSS-IS2020-data\test\meta_data_test.txt'

# Paso 2: Leer el archivo .txt en un DataFrame de pandas
df = pd.read_csv(file_path, sep=';')

# Paso 3: Crear la columna 'root' basada en la columna 'ID'
df['root'] = r'test/Full_wave_enhanced_audio/' + df['ID'] + '.wav'

# Paso 4: Crear la columna 'AD_status' basada en el valor de 'Label'
df['AD_status'] = df['Label'].apply(lambda x: 'non-AD' if x == 0 else 'AD')

# Paso 5: Crear la columna 'tt' con valor constante 'test'
df['tt'] = 'test'

# Paso 6: Reorganizar las columnas para que 'AD_status' esté después de 'Label'
df = df[['ID', 'age', 'gender', 'Label', 'AD_status', 'tt', 'mmse', 'root']]

# Paso 7: Definir la ruta de salida para el archivo CSV
output_path = r'C:\Users\alvar\PycharmProjects\DementiaBank\ADReSS_db.csv'

# Paso 8: Exportar el DataFrame a un archivo CSV con las nuevas columnas
df.to_csv(output_path, index=False)



# ----------------------------------------------------------------
# Añadir los cc (AD) a la DB
# ----------------------------------------------------------------

# Paso 1: Leer el archivo CSV existente
csv_file_path = r'C:\Users\alvar\PycharmProjects\DementiaBank\ADReSS_db.csv'
df_existing = pd.read_csv(csv_file_path)

# Paso 2: Leer el archivo .txt con las nuevas filas
txt_file_path = r'C:\Users\alvar\OneDrive\Escritorio\TFG\ADReSS-IS2020\ADReSS-IS2020-data\train\cc_meta_data.txt'

# Leer el archivo txt y manejar espacios en blanco
df_new = pd.read_csv(txt_file_path, sep=';', engine='python')

# Limpiar espacios adicionales en los nombres de las columnas y valores
df_new.columns = df_new.columns.str.strip()
df_new = df_new.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Comprobar que las columnas se leyeron correctamente
print(df_new.head())  # Muestra las primeras filas para ver si está bien cargado
print(df_new.columns)  # Verifica los nombres de las columnas

# Paso 3: Añadir las columnas que faltan a las nuevas filas
df_new['Label'] = 0  # Columna 'Label' con valor 0
df_new['AD_status'] = 'non-AD'  # Columna 'AD_status' con valor 'non-AD'
df_new['tt'] = 'train'  # Columna 'tt' con valor 'test'
df_new['root'] = r'train/Full_wave_enhanced_audio/cc/' + df_new['ID'] + '.wav'

# Reorganizar las columnas para que coincidan con el DataFrame existente
df_new = df_new[['ID', 'age', 'gender', 'Label', 'AD_status', 'tt', 'mmse', 'root']]

# Paso 4: Combinar ambos DataFrames (filas nuevas + filas existentes)
df_combined = pd.concat([df_existing, df_new], ignore_index=True)

# Paso 5: Guardar el DataFrame combinado en el archivo CSV
output_file_path = r'C:\Users\alvar\PycharmProjects\DementiaBank\ADReSS_db.csv'
df_combined.to_csv(output_file_path, index=False)


# ----------------------------------------------------------------
# Añadir los cd (non-AD) a la DB
# ----------------------------------------------------------------

# Paso 1: Leer el archivo CSV existente
csv_file_path = r'C:\Users\alvar\PycharmProjects\DementiaBank\ADReSS_db.csv'
df_existing = pd.read_csv(csv_file_path)

# Paso 2: Leer el archivo .txt con las nuevas filas
txt_file_path = r'C:\Users\alvar\OneDrive\Escritorio\TFG\ADReSS-IS2020\ADReSS-IS2020-data\train\cd_meta_data.txt'

# Leer el archivo txt y manejar espacios en blanco
df_new = pd.read_csv(txt_file_path, sep=';', engine='python')

# Limpiar espacios adicionales en los nombres de las columnas y valores
df_new.columns = df_new.columns.str.strip()
df_new = df_new.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Comprobar que las columnas se leyeron correctamente
print(df_new.head())  # Muestra las primeras filas para ver si está bien cargado
print(df_new.columns)  # Verifica los nombres de las columnas

# Paso 3: Añadir las columnas que faltan a las nuevas filas
df_new['Label'] = 1  # Columna 'Label' con valor 0
df_new['AD_status'] = 'AD'  # Columna 'AD_status' con valor 'non-AD'
df_new['tt'] = 'train'  # Columna 'tt' con valor 'test'
df_new['root'] = r'train/Full_wave_enhanced_audio/cd/' + df_new['ID'] + '.wav'

# Reorganizar las columnas para que coincidan con el DataFrame existente
df_new = df_new[['ID', 'age', 'gender', 'Label', 'AD_status', 'tt', 'mmse', 'root']]

# Paso 4: Combinar ambos DataFrames (filas nuevas + filas existentes)
df_combined = pd.concat([df_existing, df_new], ignore_index=True)



# Añade la nueva columna 'audio_type' con el valor 'Cookie_Theft_description'
df_combined['audio_type'] = 'Cookie_Theft_description'

# Paso 5: Guardar el DataFrame combinado en el archivo CSV
output_file_path = r'C:\Users\alvar\PycharmProjects\DementiaBank\ADReSS_db.csv'
df_combined.to_csv(output_file_path, index=False)







# # ----------------------------------------------------------------
# # Añadir la columna 'full_transcriptions' basada en archivos de transcripción
# # ----------------------------------------------------------------

# Paso 1: Definir la función para obtener la lista de transcripciones de cada paciente
def obtener_transcripciones(patient_id, tt, ad_status):
    # Definir la ruta base para las transcripciones
    base_dir = r'C:\Users\alvar\OneDrive\Escritorio\TFG\ADReSS-IS2020\ADReSS-IS2020-data'

    # Seleccionar la ruta según las condiciones de 'tt' y 'AD_status'
    if tt == 'test':
        transcription_file = os.path.join(base_dir, 'test', 'transcription', f'{patient_id}.c2elan.csv')
    elif tt == 'train' and ad_status == 'non-AD':
        transcription_file = os.path.join(base_dir, 'train', 'transcription', 'cc', f'{patient_id}.c2elan.csv')
    elif tt == 'train' and ad_status == 'AD':
        transcription_file = os.path.join(base_dir, 'train', 'transcription', 'cd', f'{patient_id}.c2elan.csv')
    else:
        return []  # En caso de que no se cumpla ninguna condición, devolver lista vacía

    # Comprobar si el archivo existe
    if os.path.exists(transcription_file):
        # Leer el archivo de transcripción SIN header para incluir la primera fila
        df_transcription = pd.read_csv(transcription_file, header=None)

        # Asegurarse de que tiene al menos 4 columnas
        if df_transcription.shape[1] >= 4:
            # Obtener todas las filas de la columna 4 (índice 3) como una lista
            transcriptions = df_transcription.iloc[:, 3].astype(str).tolist()
            transcriptions2 = []
            for i in transcriptions:
                transcriptions2.append(i)
            return transcriptions2
        else:
            return []  # Devolver lista vacía si no hay suficientes columnas
    else:
        return []  # Devolver lista vacía si el archivo no existe


# Paso 2: Aplicar la función a cada fila del DataFrame 'df_combined'
df_combined['transcriptions'] = df_combined.apply(
    lambda row: obtener_transcripciones(row['ID'], row['tt'], row['AD_status']), axis=1
)

# Paso 3: Guardar el DataFrame con la nueva columna en el archivo CSV
output_file_path = r'C:\Users\alvar\PycharmProjects\DementiaBank\ADReSS_db.csv'
df_combined['transcriptions'] = df_combined['transcriptions'].apply(json.dumps)
df_combined.to_csv(output_file_path, index=False)


# ----------------------------------------------------------------
# Añadir la columna 'audio_fragments' basada en archivos de audios fragmentados
# ----------------------------------------------------------------

# Paso 1: Definir la función para obtener la lista de audios fragmentados de cada paciente
def obtener_audio_fragments(patient_id, tt, ad_status):
    # Definir la ruta base para los audios fragmentados
    base_dir = r'C:\Users\alvar\OneDrive\Escritorio\TFG\ADReSS-IS2020\ADReSS-IS2020-data'

    # Seleccionar la carpeta correcta según 'tt' y 'AD_status'
    if tt == 'test':
        audio_dir_r = f'test/Fragmented_audios/'
        audio_dir = os.path.join(base_dir, 'test', 'Fragmented_audios')
    elif tt == 'train' and ad_status == 'non-AD':
        audio_dir_r = f'train/Fragmented_audios/cc/'
        audio_dir = os.path.join(base_dir, 'train', 'Fragmented_audios', 'cc')
    elif tt == 'train' and ad_status == 'AD':
        audio_dir_r = f'train/Fragmented_audios/cd/'
        audio_dir = os.path.join(base_dir, 'train', 'Fragmented_audios', 'cd')
    else:
        return []  # Si no se cumple ninguna condición, devolver una lista vacía

    # Obtener la lista de archivos de audio que coinciden con el ID del paciente
    patient_audio_files = []
    if os.path.exists(audio_dir):
        # Buscar archivos que empiecen con el ID del paciente
        for file in os.listdir(audio_dir):
            if file.startswith(f'{patient_id}-') and file.endswith('.wav'):
                # audio_i = f'{audio_dir}/{file}'
                # patient_audio_files.append(audio_i)
                patient_audio_files.append(os.path.join(audio_dir_r, file))

    def get_audio_number(file_path):
        # Extraer la parte numérica después del guion (S161-1.wav -> 1)
        match = re.search(rf'{patient_id}-(\d+)\.wav', file_path)
        if match:
            return int(match.group(1))
        return 0  # Si no encuentra un número, lo dejamos al final

    # Ordenar la lista usando el número extraído
    patient_audio_files.sort(key=get_audio_number)

    return patient_audio_files  # Devolver la lista de archivos de audio


# Paso 2: Aplicar la función a cada fila del DataFrame 'df_combined'
df_combined['audio_roots'] = df_combined.apply(
    lambda row: obtener_audio_fragments(row['ID'], row['tt'], row['AD_status']), axis=1
)

# Paso 3: Guardar el DataFrame con la nueva columna en el archivo CSV
output_file_path = r'C:\Users\alvar\PycharmProjects\DementiaBank\ADReSS_db.csv'
df_combined['audio_roots'] = df_combined['audio_roots'].apply(
    json.dumps)  # Convertir las listas en formato JSON
df_combined.to_csv(output_file_path, index=False)
