#include "filtros.h"
#include <cuda_runtime.h>
#include <iostream>

// Kernels CUDA
__global__ void kernelAplicarFiltroMedia(const unsigned char* entrada, unsigned char* salida, int ancho, int alto) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= ancho || y >= alto) return;

    int suma = 0;
    for (int ky = -1; ky <= 1; ky++)
        for (int kx = -1; kx <= 1; kx++) {
            int nx = min(max(x + kx, 0), ancho - 1);
            int ny = min(max(y + ky, 0), alto - 1);
            suma += entrada[ny * ancho + nx];
        }
    salida[y * ancho + x] = (unsigned char)(suma / 9);
}

__global__ void kernelAplicarFiltroNitidez(const unsigned char* entrada, unsigned char* salida, int ancho, int alto) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= ancho || y >= alto) return;

    int suma = 0;
    for (int ky = -1; ky <= 1; ky++)
        for (int kx = -1; kx <= 1; kx++) {
            int nx = min(max(x + kx, 0), ancho - 1);
            int ny = min(max(y + ky, 0), alto - 1);
            int pixel = entrada[ny * ancho + nx];
            if (ky == 0 && kx == 0) suma += pixel * 5;
            else if (ky == 0 || kx == 0) suma += pixel * (-1);
        }
    salida[y * ancho + x] = limitar(suma);
}

__global__ void kernelAplicarFiltroLaplaciano(const unsigned char* entrada, unsigned char* salida, int ancho, int alto) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= ancho || y >= alto) return;

    int suma = 0;
    for (int ky = -1; ky <= 1; ky++)
        for (int kx = -1; kx <= 1; kx++) {
            int nx = min(max(x + kx, 0), ancho - 1);
            int ny = min(max(y + ky, 0), alto - 1);
            int pixel = entrada[ny * ancho + nx];
            if (ky == 0 && kx == 0) suma += pixel * (-4);
            else if (ky == 0 || kx == 0) suma += pixel * 1;
        }
    int val = suma + 128;
    salida[y * ancho + x] = limitar(val);
}

// Wrappers CUDA con extern "C"
extern "C" void filtroMediaCUDA(const unsigned char* h_entrada, unsigned char* h_salida, int ancho,
                               int alto, int ksize, int hilosPorBloque) {
    size_t imgSize = ancho * alto * sizeof(unsigned char);
    unsigned char* d_entrada = nullptr;
    unsigned char* d_salida = nullptr;

    cudaMalloc(&d_entrada, imgSize);
    cudaMalloc(&d_salida, imgSize);
    cudaMemcpy(d_entrada, h_entrada, imgSize, cudaMemcpyHostToDevice);

    dim3 blockDim(hilosPorBloque, 1);
    dim3 gridDim((ancho + blockDim.x - 1) / blockDim.x, alto);

    kernelAplicarFiltroMedia<<<gridDim, blockDim>>>(d_entrada, d_salida, ancho, alto);

    cudaDeviceSynchronize();

    cudaMemcpy(h_salida, d_salida, imgSize, cudaMemcpyDeviceToHost);
    cudaFree(d_entrada);
    cudaFree(d_salida);
}

extern "C" void filtroNitidezCUDA(const unsigned char* h_entrada, unsigned char* h_salida, int ancho,
                                 int alto, int hilosPorBloque) {
    size_t imgSize = ancho * alto * sizeof(unsigned char);
    unsigned char* d_entrada = nullptr;
    unsigned char* d_salida = nullptr;

    cudaMalloc(&d_entrada, imgSize);
    cudaMalloc(&d_salida, imgSize);
    cudaMemcpy(d_entrada, h_entrada, imgSize, cudaMemcpyHostToDevice);

    dim3 blockDim(hilosPorBloque, 1);
    dim3 gridDim((ancho + blockDim.x - 1) / blockDim.x, alto);

    kernelAplicarFiltroNitidez<<<gridDim, blockDim>>>(d_entrada, d_salida, ancho, alto);

    cudaDeviceSynchronize();

    cudaMemcpy(h_salida, d_salida, imgSize, cudaMemcpyDeviceToHost);
    cudaFree(d_entrada);
    cudaFree(d_salida);
}

extern "C" void filtroLaplacianoCUDA(const unsigned char* h_entrada, unsigned char* h_salida, int ancho,
                                    int alto, int hilosPorBloque) {
    size_t imgSize = ancho * alto * sizeof(unsigned char);
    unsigned char* d_entrada = nullptr;
    unsigned char* d_salida = nullptr;

    cudaMalloc(&d_entrada, imgSize);
    cudaMalloc(&d_salida, imgSize);
    cudaMemcpy(d_entrada, h_entrada, imgSize, cudaMemcpyHostToDevice);

    dim3 blockDim(hilosPorBloque, 1);
    dim3 gridDim((ancho + blockDim.x - 1) / blockDim.x, alto);

    kernelAplicarFiltroLaplaciano<<<gridDim, blockDim>>>(d_entrada, d_salida, ancho, alto);

    cudaDeviceSynchronize();

    cudaMemcpy(h_salida, d_salida, imgSize, cudaMemcpyDeviceToHost);
    cudaFree(d_entrada);
    cudaFree(d_salida);
}
