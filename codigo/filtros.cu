#include "filtros.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

// Si una llamada CUDA falla, el programa se detiene en vez de medir resultados invalidos.
static void verificar(cudaError_t error, const char* operacion) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "Error CUDA en %s: %s\n", operacion, cudaGetErrorString(error));
        std::exit(1);
    }
}
#define CUDA_OK(llamada) verificar((llamada), #llamada)

// Distribucion: bloques e hilos solo en X.
//   indice global = blockIdx.x * blockDim.x + threadIdx.x
//   fila          = indice / ancho
//   columna       = indice % ancho
// El ultimo bloque puede tener hilos de sobra: esos hilos no procesan ningun pixel.
__global__ void kernelFiltroMedia(const unsigned char* entrada, float* salida, int ancho, int alto, int ksize) {
    int indice = blockIdx.x * blockDim.x + threadIdx.x;
    if (indice >= ancho * alto) return;
    int y = indice / ancho;
    int x = indice % ancho;
    salida[indice] = calcularMedia(entrada, ancho, alto, x, y, ksize);
}

__global__ void kernelFiltroNitidez(const unsigned char* entrada, float* salida, int ancho, int alto) {
    int indice = blockIdx.x * blockDim.x + threadIdx.x;
    if (indice >= ancho * alto) return;
    int y = indice / ancho;
    int x = indice % ancho;
    salida[indice] = calcularNitidez(entrada, ancho, alto, x, y);
}

__global__ void kernelFiltroLaplaciano(const unsigned char* entrada, float* salida, int ancho, int alto) {
    int indice = blockIdx.x * blockDim.x + threadIdx.x;
    if (indice >= ancho * alto) return;
    int y = indice / ancho;
    int x = indice % ancho;
    salida[indice] = calcularLaplaciano(entrada, ancho, alto, x, y);
}

static double msEntre(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Pasos comunes a los tres filtros. Las reservas y los eventos se crean antes de
// medir, asi que el tiempo conjunto solo incluye transferencias y ejecucion.
template <typename LanzarKernel>
static void ejecutarCUDA(const unsigned char* h_entrada, float* h_salida, int ancho, int alto,
                         int hilosPorBloque, TiemposCUDA* tiempos, LanzarKernel lanzarKernel) {
    if (hilosPorBloque < 1 || hilosPorBloque > maxHilosPorBloqueCUDA()) {
        std::fprintf(stderr, "La GPU no permite %d hilos por bloque.\n", hilosPorBloque);
        std::exit(1);
    }

    size_t pixeles = static_cast<size_t>(ancho) * alto;
    unsigned char* d_entrada = nullptr;
    float* d_salida = nullptr;
    CUDA_OK(cudaMalloc(&d_entrada, pixeles * sizeof(unsigned char)));
    CUDA_OK(cudaMalloc(&d_salida, pixeles * sizeof(float)));

    cudaEvent_t inicioKernel, finKernel;
    CUDA_OK(cudaEventCreate(&inicioKernel));
    CUDA_OK(cudaEventCreate(&finKernel));

    int bloques = static_cast<int>((pixeles + hilosPorBloque - 1) / hilosPorBloque);
    dim3 dimBloque(hilosPorBloque);
    dim3 dimMalla(bloques);

    auto t0 = std::chrono::steady_clock::now();
    CUDA_OK(cudaMemcpy(d_entrada, h_entrada, pixeles * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_OK(cudaDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();

    CUDA_OK(cudaEventRecord(inicioKernel));
    lanzarKernel(d_entrada, d_salida, dimMalla, dimBloque);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaEventRecord(finKernel));
    CUDA_OK(cudaEventSynchronize(finKernel));
    auto t2 = std::chrono::steady_clock::now();

    CUDA_OK(cudaMemcpy(h_salida, d_salida, pixeles * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaDeviceSynchronize());
    auto t3 = std::chrono::steady_clock::now();

    float kernelMs = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&kernelMs, inicioKernel, finKernel));

    if (tiempos != nullptr) {
        tiempos->h2d_ms = msEntre(t0, t1);
        tiempos->kernel_ms = kernelMs;
        tiempos->d2h_ms = msEntre(t2, t3);
        tiempos->total_ms = msEntre(t0, t3);
        tiempos->bloques = bloques;
    }

    CUDA_OK(cudaEventDestroy(inicioKernel));
    CUDA_OK(cudaEventDestroy(finKernel));
    CUDA_OK(cudaFree(d_entrada));
    CUDA_OK(cudaFree(d_salida));
}

void filtroMediaCUDA(const unsigned char* h_entrada, float* h_salida, int ancho, int alto, int ksize,
                     int hilosPorBloque, TiemposCUDA* tiempos) {
    ejecutarCUDA(h_entrada, h_salida, ancho, alto, hilosPorBloque, tiempos,
                 [=](const unsigned char* e, float* s, dim3 malla, dim3 bloque) {
                     kernelFiltroMedia<<<malla, bloque>>>(e, s, ancho, alto, ksize);
                 });
}

void filtroNitidezCUDA(const unsigned char* h_entrada, float* h_salida, int ancho, int alto,
                       int hilosPorBloque, TiemposCUDA* tiempos) {
    ejecutarCUDA(h_entrada, h_salida, ancho, alto, hilosPorBloque, tiempos,
                 [=](const unsigned char* e, float* s, dim3 malla, dim3 bloque) {
                     kernelFiltroNitidez<<<malla, bloque>>>(e, s, ancho, alto);
                 });
}

void filtroLaplacianoCUDA(const unsigned char* h_entrada, float* h_salida, int ancho, int alto,
                          int hilosPorBloque, TiemposCUDA* tiempos) {
    ejecutarCUDA(h_entrada, h_salida, ancho, alto, hilosPorBloque, tiempos,
                 [=](const unsigned char* e, float* s, dim3 malla, dim3 bloque) {
                     kernelFiltroLaplaciano<<<malla, bloque>>>(e, s, ancho, alto);
                 });
}

int maxHilosPorBloqueCUDA() {
    cudaDeviceProp propiedades;
    CUDA_OK(cudaGetDeviceProperties(&propiedades, 0));
    return propiedades.maxThreadsPerBlock;
}
