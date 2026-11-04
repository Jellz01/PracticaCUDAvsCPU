#include "filtros.h"

#include <cmath>

void aplicarFiltroMedia(const unsigned char* entrada, float* salida, int ancho, int alto, int ksize) {
    for (int y = 0; y < alto; y++)
        for (int x = 0; x < ancho; x++)
            salida[y * ancho + x] = calcularMedia(entrada, ancho, alto, x, y, ksize);
}

void aplicarFiltroNitidez(const unsigned char* entrada, float* salida, int ancho, int alto) {
    for (int y = 0; y < alto; y++)
        for (int x = 0; x < ancho; x++)
            salida[y * ancho + x] = calcularNitidez(entrada, ancho, alto, x, y);
}

void aplicarFiltroLaplaciano(const unsigned char* entrada, float* salida, int ancho, int alto) {
    for (int y = 0; y < alto; y++)
        for (int x = 0; x < ancho; x++)
            salida[y * ancho + x] = calcularLaplaciano(entrada, ancho, alto, x, y);
}

unsigned char convertirAVisualizacion(float valor, TipoFiltro filtro) {
    float v = (filtro == FILTRO_LAPLACIANO) ? valor + 128.0f : valor;
    v = std::round(v);
    if (v < 0.0f) return 0;
    if (v > 255.0f) return 255;
    return static_cast<unsigned char>(v);
}
