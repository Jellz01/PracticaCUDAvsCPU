#ifndef FILTROS_H
#define FILTROS_H

#ifdef __CUDACC__
// Para CUDA, limitar es función device inline
__device__ inline unsigned char limitar(int val) {
    if (val < 0) return 0;
    if (val > 255) return 255;
    return static_cast<unsigned char>(val);
}
#else
// Para CPU es función inline normal
inline unsigned char limitar(int val) {
    if (val < 0) return 0;
    if (val > 255) return 255;
    return static_cast<unsigned char>(val);
}
#endif

// Firmas funciones CPU
void aplicarFiltroLaplaciano(const unsigned char* entrada, unsigned char* salida, int ancho, int alto);
void aplicarFiltroMedia(const unsigned char* entrada, unsigned char* salida, int ancho, int alto, int ksize);
void aplicarFiltroNitidez(const unsigned char* entrada, unsigned char* salida, int ancho, int alto);

#ifdef __cplusplus
extern "C" {
#endif

// Wrappers CUDA con definición extern "C"
void filtroMediaCUDA(const unsigned char* h_entrada, unsigned char* h_salida, int ancho, int alto, int ksize, int hilosPorBloque);
void filtroNitidezCUDA(const unsigned char* h_entrada, unsigned char* h_salida, int ancho, int alto, int hilosPorBloque);
void filtroLaplacianoCUDA(const unsigned char* h_entrada, unsigned char* h_salida, int ancho, int alto, int hilosPorBloque);

#ifdef __cplusplus
}
#endif

#endif // FILTROS_H
