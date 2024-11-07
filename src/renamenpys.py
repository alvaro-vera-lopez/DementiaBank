import os

# Ruta de la carpeta que contiene los archivos
folder_path = r'C:\Users\alvar\OneDrive\Escritorio\TFG\DEMENTIABANK\numpys1515'

# Iteramos sobre todos los archivos en la carpeta
for filename in os.listdir(folder_path):
    # Si el archivo tiene "_1515" en su nombre, lo renombramos
    if filename.endswith('_7103.npy'):
        # Creamos el nuevo nombre reemplazando "_1515" por "_2002"
        new_filename = filename.replace('_7103', '_7104')

        # Obtenemos las rutas completas para renombrar el archivo
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_filename)

        # Renombramos el archivo
        os.rename(old_file, new_file)
        print(f'Renombrado: {filename} -> {new_filename}')

print('Proceso completado.')
