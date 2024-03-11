import os

folder_path = "test"
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".jpg"):
            # Obtener el nombre de la carpeta raíz
            root_folder = os.path.basename(root)
            # Obtener la ruta completa del archivo
            old_file_path = os.path.join(root, file)
            # Nuevo nombre del archivo
            new_file_name = f"{root_folder}_{file}"
            # Nueva ruta completa del archivo
            new_file_path = os.path.join(root, new_file_name)
            # Renombrar el archivo
            os.rename(old_file_path, new_file_path)
            print(f"Archivo renombrado: {old_file_path} -> {new_file_path}")