#!/bin/bash
# Reproduce toda la practica. Ejecutar desde la raiz del repositorio, con el
# entorno de Visual Studio cargado (cl.exe y cmake en el PATH).
set -e

python ResizeImage.py

cd codigo
cmake -S . -B build-local -G "NMake Makefiles"
cmake --build build-local

./pruebas.exe ../Imagenes/entrada ../experimentos | tee ../experimentos/pruebas.log
./benchmark.exe ../Imagenes/entrada ../experimentos/crudo.csv 10 2

cd ..
python experimentos/graficar.py
