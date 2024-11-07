import math

# Función para calcular la distancia entre dos puntos en la Tierra usando la fórmula de Haversine
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radio de la Tierra en kilómetros
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c  # Devuelve la distancia en kilómetros


# Función para extraer los datos de los satélites de las líneas del archivo NMEA
def extraer_satelites(nmea_lines):
    satelites = {}

    for line in nmea_lines:
        # Filtramos las líneas con información de satélites (GPGSV, GLGSV, GNGSV)
        if line.startswith('$GPGSV') or line.startswith('$GLGSV') or line.startswith('$GNGSV'):
            fields = line.split(',')
            sat_id = int(fields[3])  # ID del satélite

            snr_str = fields[7] if len(fields) > 7 else ''  # SNR (si existe)
            try:
                snr = float(snr_str) if snr_str else 0.0
            except ValueError:
                snr = 0.0  # Asignamos un valor predeterminado si no se puede convertir

            # Validamos que latitud y longitud tengan valores numéricos antes de convertir
            try:
                lat = float(fields[2]) if fields[2] else 0.0  # Si no tiene valor, asignamos 0.0
                lon = float(fields[4]) if fields[4] else 0.0  # Lo mismo para longitud
            except ValueError:
                lat = lon = 0.0  # Si hay error en la conversión, asignamos 0.0 como valor predeterminado

            satelites[sat_id] = {
                'lat': lat,
                'lon': lon,
                'snr': snr
            }

    return satelites


# Función para procesar y comparar los satélites
def comparar_satélites(satelites_1, satelites_2):
    # Diccionario para almacenar los datos de satélites comunes
    satelites_comunes = {}

    # Buscar satélites comunes entre los dos conjuntos de datos
    for sat_id in satelites_1:
        if sat_id in satelites_2:
            satelites_comunes[sat_id] = {
                'lat1': satelites_1[sat_id]['lat'],
                'lon1': satelites_1[sat_id]['lon'],
                'snr1': satelites_1[sat_id]['snr'],
                'lat2': satelites_2[sat_id]['lat'],
                'lon2': satelites_2[sat_id]['lon'],
                'snr2': satelites_2[sat_id]['snr']
            }

    # Comparar la distancia y el cambio de SNR
    for sat_id, sat_data in satelites_comunes.items():
        lat1, lon1, snr1 = sat_data['lat1'], sat_data['lon1'], sat_data['snr1']
        lat2, lon2, snr2 = sat_data['lat2'], sat_data['lon2'], sat_data['snr2']

        # Calcular la distancia entre las dos posiciones
        distancia = haversine(lat1, lon1, lat2, lon2)

        # Calcular el cambio en el SNR
        cambio_snr = snr2 - snr1

        # Imprimir los resultados
        print(f"Satélite {sat_id}:")
        print(f"  Distancia entre posiciones: {distancia:.2f} km")
        print(f"  Cambio en SNR: {cambio_snr:.2f} dB\n")

    # Mostrar los satélites comunes y sus datos
    print("Satélites comunes entre los dos archivos:")
    for sat_id, sat_data in satelites_comunes.items():
        print(f"Satélite {sat_id} -> Pos1: ({sat_data['lat1']}, {sat_data['lon1']}) SNR1: {sat_data['snr1']}, Pos2: ({sat_data['lat2']}, {sat_data['lon2']}) SNR2: {sat_data['snr2']}")

# Leer el contenido de los archivos de texto y procesarlos
def leer_archivo_y_extraer_datos(archivo):
    with open(archivo, 'r') as f:
        lines = f.readlines()

    # Extraer los datos de los satélites
    return extraer_satelites(lines)


# Cargar y procesar los dos archivos
archivo_1 = "C:/Users/alvar/OneDrive/Escritorio/1_MASTER/RADIO/PRACTICAS/PR5/datos2.txt"
archivo_2 = "C:/Users/alvar/OneDrive/Escritorio/1_MASTER/RADIO/PRACTICAS/PR5/datos3.txt"

# Obtener todos los satélites de ambos archivos
satelites_1 = leer_archivo_y_extraer_datos(archivo_1)
satelites_2 = leer_archivo_y_extraer_datos(archivo_2)

# Comparar los satélites entre los dos archivos
comparar_satélites(satelites_1, satelites_2)
