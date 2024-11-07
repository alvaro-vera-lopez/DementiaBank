import os
import pandas as pd
from pydub import AudioSegment

# Definir rutas principales
input_audio_dir = r'C:\Users\alvar\OneDrive\Escritorio\TFG\ADReSS-IS2020\ADReSS-IS2020-data\test\Full_wave_enhanced_audio'
input_csv_dir = r'C:\Users\alvar\OneDrive\Escritorio\TFG\ADReSS-IS2020\ADReSS-IS2020-data\test\transcription'
output_dir = r'C:\Users\alvar\OneDrive\Escritorio\TFG\ADReSS-IS2020\ADReSS-IS2020-data\test\Fragmented_audios'

# Crear la carpeta de destino si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Función para procesar cada audio y su respectivo CSV
def procesar_audio(id_paciente):
    # Ruta del archivo de audio del paciente
    audio_path = os.path.join(input_audio_dir, f'{id_paciente}.wav')
    # Ruta del CSV correspondiente
    csv_path = os.path.join(input_csv_dir, f'{id_paciente}.c2elan.csv')

    # Verificar que el archivo de audio y el CSV existen
    if not os.path.exists(audio_path):
        print(f"El archivo de audio {audio_path} no existe.")
        return
    if not os.path.exists(csv_path):
        print(f"El archivo CSV {csv_path} no existe.")
        return

    # Cargar el audio completo
    audio = AudioSegment.from_wav(audio_path)

    # Leer el CSV (sin índice automático)
    df = pd.read_csv(csv_path, header=None)

    # Iterar sobre cada fila del CSV para obtener los fragmentos
    for numfrag, row in enumerate(df.itertuples(index=False), start=1):
        inicio_ms = int(row[0] * 1000)  # Convertir a milisegundos
        fin_ms = int(row[1] * 1000)  # Convertir a milisegundos

        # Extraer el fragmento de audio
        fragmento = audio[inicio_ms:fin_ms]

        # Definir el nombre del archivo de salida para el fragmento
        fragment_output_path = os.path.join(output_dir, f'{id_paciente}-{numfrag}.wav')

        # Exportar el fragmento
        fragmento.export(fragment_output_path, format="wav")
        print(f"Fragmento {numfrag} del paciente {id_paciente} guardado en: {fragment_output_path}")


# Procesar todos los archivos de audio en el directorio de entrada
for file_name in os.listdir(input_audio_dir):
    if file_name.endswith('.wav'):
        id_paciente = file_name.replace('.wav', '')
        procesar_audio(id_paciente)

