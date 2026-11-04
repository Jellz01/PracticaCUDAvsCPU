"""Genera las entradas de la practica a partir de una sola imagen.

Pasos: recorte cuadrado centrado de 6000x6000, escala de grises (Pillow modo L,
ITU-R 601) y reduccion a 4000 y 2000 con LANCZOS. Se guarda en PNG, sin perdida,
para que todas las implementaciones lean exactamente los mismos pixeles.

Uso (desde la raiz del repositorio): python ResizeImage.py
"""
import hashlib
import os

from PIL import Image

ORIGEN = "8k image.jpg"
SALIDA = "Imagenes/entrada"
LADO_RECORTE = 6000
TAMANOS = (6000, 4000, 2000)

os.makedirs(SALIDA, exist_ok=True)

with Image.open(ORIGEN) as original:
    ancho, alto = original.size
    izq = (ancho - LADO_RECORTE) // 2
    arriba = (alto - LADO_RECORTE) // 2
    caja = (izq, arriba, izq + LADO_RECORTE, arriba + LADO_RECORTE)
    recorte = original.convert("RGB").crop(caja).convert("L")
    print(f"Origen {ancho}x{alto}, recorte {caja}")

lineas = []
for lado in TAMANOS:
    imagen = recorte if lado == LADO_RECORTE else recorte.resize((lado, lado), Image.LANCZOS)
    ruta = f"{SALIDA}/{lado}.png"
    imagen.save(ruta)
    with open(ruta, "rb") as archivo:
        resumen = hashlib.sha256(archivo.read()).hexdigest()
    lineas.append(f"{resumen}  {lado}.png")
    print(f"{ruta}: {imagen.size}, {imagen.mode}, sha256 {resumen}")

with open(f"{SALIDA}/SHA256SUMS.txt", "w", encoding="utf-8") as archivo:
    archivo.write("\n".join(lineas) + "\n")
