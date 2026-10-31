#include "filtros.h"
#include <algorithm>
#include <iostream>

// Filtro Laplaciano (sin cambio)
void aplicarFiltroLaplaciano(const unsigned char* entrada, unsigned char* salida, int ancho, int alto) {
    for (int y = 0; y < alto; y++) {
        for (int x = 0; x < ancho; x++) {
            int suma = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int nx = std::min(std::max(x + kx, 0), ancho - 1);
                    int ny = std::min(std::max(y + ky, 0), alto - 1);
                    int pixel = entrada[ny * ancho + nx];
                    if (ky == 0 && kx == 0)
                        suma += pixel * (-4);
                    else if (ky == 0 || kx == 0)
                        suma += pixel * 1;
                }
            }
            int val = suma + 128;
            salida[y * ancho + x] = limitar(val);
        }
    }
}

// Filtro Media con parámetro ksize para tamaño de máscara
void aplicarFiltroMedia(const unsigned char* entrada, unsigned char* salida, int ancho, int alto, int ksize) {
    int radius = ksize / 2;
    for (int y = 0; y < alto; y++) {
        for (int x = 0; x < ancho; x++) {
            int suma = 0;
            int count = 0;
            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {
                    int nx = std::min(std::max(x + kx, 0), ancho - 1);
                    int ny = std::min(std::max(y + ky, 0), alto - 1);
                    suma += entrada[ny * ancho + nx];
                    count++;
                }
            }
            salida[y * ancho + x] = static_cast<unsigned char>(suma / count);
        }
    }
}

// Filtro Nitidez sin cambio en parámetros (firma igual por consistencia)
void aplicarFiltroNitidez(const unsigned char* entrada, unsigned char* salida, int ancho, int alto) {
    for (int y = 0; y < alto; y++) {
        for (int x = 0; x < ancho; x++) {
            int suma = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int nx = std::min(std::max(x + kx, 0), ancho - 1);
                    int ny = std::min(std::max(y + ky, 0), alto - 1);
                    int pixel = entrada[ny * ancho + nx];
                    if (ky == 0 && kx == 0)
                        suma += pixel * 5;
                    else if (ky == 0 || kx == 0)
                        suma += pixel * (-1);
                }
            }
            salida[y * ancho + x] = limitar(suma);
        }
    }
}

void aplicarFiltroSegunOpcion(const unsigned char* entrada, unsigned char* salida, int ancho, int alto, int filtro) {
    switch (filtro) {
        case 1:
            aplicarFiltroLaplaciano(entrada, salida, ancho, alto);
            break;
        case 2:
            // Usamos máscara 3 por defecto para esta función, podría ajustarse
            aplicarFiltroMedia(entrada, salida, ancho, alto, 3);
            break;
        case 3:
            aplicarFiltroNitidez(entrada, salida, ancho, alto);
            break;
        default:
            std::cerr << "Filtro inválido seleccionado.\n";
            break;
    }
}
