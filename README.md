# Convolución de imágenes de alta resolución con CUDA C

Práctica 01 de Computación Paralela. Filtros de media, nitidez y laplaciano en versión secuencial (C++) y paralela (CUDA C), con pruebas de corrección y experimentos de rendimiento.

## Dependencias

Versiones usadas en los resultados publicados (detalle en `experimentos/equipo.txt`):

- CUDA Toolkit 13.4 (`nvcc` 13.4.92) y una GPU NVIDIA
- Visual Studio 2026 Build Tools (MSVC 19.51) en x64
- CMake 3.18 o superior
- Python 3 con `pillow`, `pandas` y `matplotlib`

## Compilar y ejecutar

Desde una terminal con el entorno de Visual Studio cargado (`vcvars64.bat`):

```powershell
python ResizeImage.py                 # desde la raíz: genera Imagenes/entrada
cd codigo
cmake -S . -B build-local -G "NMake Makefiles"
cmake --build build-local             # Release por defecto

.\main.exe                            # programa interactivo: CPU o CUDA
.\pruebas.exe                         # pruebas de corrección, escribe validacion.csv
.\benchmark.exe                       # experimentos, escribe crudo.csv (varios minutos)
cd ..
python experimentos/graficar.py       # genera resumen.csv y las gráficas
```

Los programas se ejecutan desde `codigo/` porque usan rutas relativas. `josephFabian.sh` hace toda la secuencia en Git Bash.

`filtros.cu` no tiene `main()`, así que no se compila solo como ejecutable: CMake lo une a cada programa.

## Estructura

| Ruta | Contenido |
|---|---|
| `codigo/filtros.h` | Máscaras y cálculo por píxel, compartidos por CPU y GPU |
| `codigo/filtros.cpp` | Versión secuencial y conversión a 0-255 |
| `codigo/filtros.cu` | Kernels, distribución en bloques, transferencias y tiempos |
| `codigo/main.cpp` | Programa interactivo |
| `codigo/pruebas.cpp` | Pruebas de corrección |
| `codigo/benchmark.cpp` | Experimentos de rendimiento |
| `ResizeImage.py` | Preparación de las imágenes |
| `Imagenes/entrada/` | Entradas en PNG y sus SHA-256 |
| `experimentos/crudo.csv` | Registros originales de tiempos |
| `experimentos/resumen.csv` | Mediana, IQR, desviación, aceleración y transferencias |
| `experimentos/validacion.csv` | Errores CPU-GPU por configuración |
| `experimentos/pruebas.log` | Salida completa de `pruebas.exe` |
| `experimentos/equipo.txt` | Hardware y versiones de las herramientas |
| `experimentos/graficas/`, `experimentos/imagenes/` | Gráficas e imágenes para el informe |

## Filtros

Todos usan extensión con ceros: un píxel fuera de la imagen vale 0. La entrada es de 8 bits en escala de grises y la salida es `float`, sin convertir.

| Filtro | Máscara | Efecto | Conversión a 0-255 |
|---|---|---|---|
| Media k×k | Todos los coeficientes valen 1/(k×k) | Suaviza | Redondeo |
| Nitidez | `[0,-1,0; -1,5,-1; 0,-1,0]` | Realza detalles | Saturación a [0, 255] |
| Laplaciano | `[0,1,0; 1,-4,1; 0,1,0]` | Resalta cambios de intensidad | Suma 128 y saturación a [0, 255] |

La conversión se aplica solo para guardar la imagen. Las pruebas comparan los valores anteriores a ella.

## Distribución del trabajo

Se usan bloques e hilos solo en X, y cada hilo calcula un píxel leyendo la imagen desde memoria global:

```text
bloques = ceil(ancho × alto / hilosPorBloque)
indice  = blockIdx.x × blockDim.x + threadIdx.x
fila    = indice / ancho
columna = indice % ancho
```

Si `indice >= ancho × alto`, el hilo termina sin hacer nada. Ese caso ocurre en el último bloque cuando el total de píxeles no es múltiplo de los hilos por bloque. La entrada y la salida son arreglos separados.

## Validación

`pruebas.exe` devuelve 0 solo si todo pasa.

**Prueba manual.** Matriz de entrada, con los resultados calculados a mano:

```text
10 20 30
40 50 60
70 80 90
```

| Filtro | Resultado esperado |
|---|---|
| Media 3×3 (entre 9) | `120 210 160; 270 450 330; 240 390 280` |
| Nitidez | `-10 10 70; 70 50 130; 230 190 310` |
| Laplaciano | `20 10 -40; -30 0 -70; -160 -110 -220` |

Ejemplo de la esquina superior izquierda en nitidez: `5×10 - 20 - 40 = -10`. Los vecinos de arriba y de la izquierda están fuera de la imagen y valen 0.

**Dimensiones no múltiplos.** Se prueban 1×1, 7×5, 513×1, 1×257 y 1001×777 con 128, 256 y 512 hilos por bloque, y los 3 filtros (media con k = 3, 9 y 21).

**Comparación CPU-GPU.** Se reportan el error absoluto máximo y el medio sobre la salida `float`, antes de convertirla, en `experimentos/validacion.csv`.

**Tolerancia: 1e-4.** Cerca de 255, dos `float` consecutivos se separan 1.5e-5, así que 1e-4 admite solo unos pocos pasos de redondeo. Queda muy por debajo de 0.5, la diferencia que cambiaría un nivel de gris.

## Metodología de medición

- **Configuraciones:** las de la guía. Las mediciones repetidas se reutilizan entre experimentos, como indica la columna `experimentos` de `crudo.csv`. Fuera del experimento de hilos se usan 256 hilos por bloque.
- **Repeticiones:** 2 de calentamiento y 10 medidas por configuración. El calentamiento queda en el CSV con `fase = calentamiento` y no entra en el resumen.
- **Variabilidad:** mediana, rango intercuartílico (IQR), desviación estándar y coeficiente de variación.
- **Tiempos CUDA:** las transferencias y el tiempo conjunto se miden con `steady_clock` y `cudaDeviceSynchronize`. El kernel se mide con eventos CUDA.
- **Excluido de la medición:** lectura y escritura de archivos, conversión a 0-255, y reserva y liberación de memoria.
- **Aceleración de cómputo:** tiempo secuencial / tiempo del kernel.
- **Aceleración con transferencias:** tiempo secuencial / tiempo conjunto.

Las transferencias no son simétricas: CPU→GPU mueve 1 byte por píxel y GPU→CPU mueve 4, porque la salida es `float`.

## Datos de entrada

`ResizeImage.py` parte de `8k image.jpg` (8000×6000), hace un recorte centrado común de 6000×6000 en `(1000, 0, 7000, 6000)`, convierte a escala de grises (ITU-R 601) y reduce a 4000 y 2000 con LANCZOS. Las salidas son PNG sin pérdida y sus resúmenes están en `Imagenes/entrada/SHA256SUMS.txt`.

**Fuente de la imagen original:** [Boat at Balos Beach](https://unsplash.com/photos/white-boat-at-balos-beach-sKOBxEb6E-E), de Joshua Kettle, en Unsplash ([licencia de Unsplash](https://unsplash.com/license)).
