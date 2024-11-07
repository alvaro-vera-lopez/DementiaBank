import pandas as pd
import matplotlib.pyplot as plt

# Función para parsear el archivo de texto y extraer los datos de GSV
def parse_nmea_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Parsear líneas de tipo GPGSV o GLGSV
            if line.startswith(('$GPGSV', '$GLGSV')):
                parts = line.split(',')
                timestamp = parts[1]  # Ajustar para el campo de tiempo real en el archivo
                # Extraer datos de satélites
                for i in range(4, len(parts) - 4, 4):
                    if len(parts[i:i + 4]) == 4:
                        sv_prn, elevation, azimuth, snr = parts[i:i + 4]
                        print(f'Elevacion: {elevation}')
                        # Verificar si elevation, azimuth y snr no están vacíos
                        if elevation and azimuth and snr:
                            data.append({
                                'timestamp': timestamp,
                                'satellite': sv_prn,
                                'elevation': float(elevation),
                                'azimuth': float(azimuth),
                                'snr': float(snr),
                            })
    return pd.DataFrame(data)

# Cargar los datos de los archivos
file_path1 = 'C:\\Users\\alvar\\OneDrive\\Escritorio\\1_MASTER\\RADIO\\PRACTICAS\\PR5\\datos_recibidos_2.txt'  # Ajusta el path
file_path2 = 'C:\\Users\\alvar\\OneDrive\\Escritorio\\1_MASTER\\RADIO\\PRACTICAS\\PR5\\datos_recibidos_3.txt'  # Ajusta el path

# Parsear los archivos
df1 = parse_nmea_file(file_path1)
# df2 = parse_nmea_file(file_path2)
#
# # Combinar los DataFrames
# df = pd.concat([df1, df2])

df = df1

# Graficar Azimuth vs SNR
plt.figure(figsize=(12, 7))  # Aumentar el tamaño de la figura
for sat in df['satellite'].unique():
    sat_data = df[df['satellite'] == sat]
    plt.scatter(sat_data['azimuth'], sat_data['snr'], label=f'Sat {sat}', alpha=0.6)

plt.xlabel('Azimuth (grados)')
plt.ylabel('SNR (dB)')
plt.title('Comparación de Azimuth y SNR por Satélite')

# Colocar la leyenda fuera del gráfico
plt.legend(title='Satélites', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.tight_layout()  # Ajustar el layout para no cortar elementos
plt.show()

# Graficar Elevación vs SNR
plt.figure(figsize=(12, 7))  # Aumentar el tamaño de la figura
for sat in df['satellite'].unique():
    sat_data = df[df['satellite'] == sat]
    plt.scatter(sat_data['elevation'], sat_data['snr'], label=f'Sat {sat}', alpha=0.6)

plt.xlabel('Elevación (grados)')
plt.ylabel('SNR (dB)')
plt.title('Comparación de Elevación y SNR por Satélite')

# Colocar la leyenda fuera del gráfico
plt.legend(title='Satélites', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()
plt.tight_layout()  # Ajustar el layout para no cortar elementos
plt.show()
