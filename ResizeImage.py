from PIL import Image
import os
import shutil

output_dir = "Imagenes/resultados"

# 1. Borrar la carpeta anterior si existe para tener solo imágenes frescas
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

# 2. Crear la carpeta limpia
os.makedirs(output_dir)

images = ['1.jpg', '2.jpg', '3.jpg']
sizes = [(2000, 2000), (4000, 4000), (6000, 6000)]

for img_name, size in zip(images, sizes):
    input_path = os.path.join("Imagenes", img_name)
    
    if not os.path.exists(input_path):
        print(f"Aviso: No se encontró la imagen {input_path}")
        continue

    with Image.open(input_path) as img:
        # Redimensionar al tamaño correspondiente (2000, 4000 o 6000)
        resized_img = img.resize(size)  
        
        # Nombrar el archivo usando únicamente la dimensión (ej: "2000.jpg")
        dimension_str = str(size[0])  # Toma el ancho (2000, 4000 o 6000)
        ext = os.path.splitext(img_name)[1]  # Mantiene la extensión original (.jpg)
        output_filename = f"{dimension_str}{ext}"
        
        output_path = os.path.join(output_dir, output_filename)
        
        # Guardar la imagen
        resized_img.save(output_path)
        print(f"Saved: {output_path}")