#!/bin/bash

# Define arrays con las opciones a probar
resoluciones=("1" "2" "3")       # 1=2000x2000, 2=4000x4000, 3=6000x6000
mascaras=(3 9 21)                # Tamaños de máscara para filtro media
filtros=("1" "2" "3")            # 1=Media, 2=Nitidez, 3=Laplaciano

# Carpeta donde está el ejecutable y base para ruta de imágenes
BASE_DIR="C:/Users/dauqu_ksp124/Desktop/JW/Practica1/codigo"
EJEC="$BASE_DIR/main.exe"

for res in "${resoluciones[@]}"; do
  for filtro in "${filtros[@]}"; do
    if [ "$filtro" == "1" ]; then
      # Para filtro media, probar con todas las máscaras
      for mascara in "${mascaras[@]}"; do
        echo "Ejecutando: resolución $res, filtro $filtro (media), mascara $mascara"
        "$EJEC" <<EOF
1
$filtro
$res
$mascara
EOF
      done
    else
      # Para filtros nitidez y laplaciano no se usa mascara
      echo "Ejecutando: resolución $res, filtro $filtro"
      "$EJEC" <<EOF
1
$filtro
$res
EOF
    fi
  done
done

echo "Pruebas completadas."
