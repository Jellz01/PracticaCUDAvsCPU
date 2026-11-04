#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include "filtros.h"

#include <cstring>
#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>  // C++17

void crearDirectorioSiNoExiste(const std::string& ruta) {
    std::filesystem::create_directories(ruta);
}

int main() {
    int modo, filtro, resolucion, mascara = 3, hilosPorBloque = 256;
    int ancho = 0, alto = 0;

    std::cout << "Seleccione modo de ejecucion:\n1) CPU\n2) CUDA GPU\n";
    std::cin >> modo;
    if (modo != 1 && modo != 2) {
        std::cout << "Modo invalido\n";
        return 1;
    }

    std::cout << "Seleccione filtro:\n1) Media\n2) Nitidez\n3) Laplaciano\n";
    std::cin >> filtro;
    if (filtro < 1 || filtro > 3) {
        std::cout << "Filtro invalido\n";
        return 1;
    }

    std::cout << "Seleccione resolucion:\n1) 2000x2000\n2) 4000x4000\n3) 6000x6000\n";
    std::cin >> resolucion;
    std::string rutaImagen;
    switch (resolucion) {
        case 1: ancho = alto = 2000; rutaImagen = "../Imagenes/entrada/2000.png"; break;
        case 2: ancho = alto = 4000; rutaImagen = "../Imagenes/entrada/4000.png"; break;
        case 3: ancho = alto = 6000; rutaImagen = "../Imagenes/entrada/6000.png"; break;
        default:
            std::cout << "Resolución inválida\n";
            return 1;
    }

    if (filtro == 1) {
        std::cout << "Ingrese tamaño de máscara impar (ejemplo: 3, 9, 21): ";
        std::cin >> mascara;
        if (mascara < 3 || mascara % 2 == 0) {
            std::cout << "Tamaño de máscara inválido\n";
            return 1;
        }
    }

    if (modo == 2) {
        std::cout << "Ingrese hilos por bloque CUDA (128, 256, 512): ";
        std::cin >> hilosPorBloque;
        if (hilosPorBloque != 128 && hilosPorBloque != 256 && hilosPorBloque != 512) {
            std::cout << "Cantidad de hilos inválida\n";
            return 1;
        }
    }

    int canales;
    unsigned char* imgCargada = stbi_load(rutaImagen.c_str(), &ancho, &alto, &canales, 1);
    if (!imgCargada) {
        std::cerr << "Error cargando imagen de entrada.\n";
        return 1;
    }

    unsigned char* h_entrada = new unsigned char[ancho * alto];
    float* h_salida = new float[ancho * alto];
    unsigned char* h_visual = new unsigned char[ancho * alto];
    memcpy(h_entrada, imgCargada, ancho * alto * sizeof(unsigned char));
    stbi_image_free(imgCargada);

    std::cout << "Ejecutando filtro...\n";

    if (modo == 1) {
        auto start = std::chrono::steady_clock::now();
        if (filtro == 1)
            aplicarFiltroMedia(h_entrada, h_salida, ancho, alto, mascara);
        else if (filtro == 2)
            aplicarFiltroNitidez(h_entrada, h_salida, ancho, alto);
        else
            aplicarFiltroLaplaciano(h_entrada, h_salida, ancho, alto);
        auto end = std::chrono::steady_clock::now();
        std::cout << "Tiempo secuencial: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms\n";
    } else {
        TiemposCUDA t{};
        if (filtro == 1)
            filtroMediaCUDA(h_entrada, h_salida, ancho, alto, mascara, hilosPorBloque, &t);
        else if (filtro == 2)
            filtroNitidezCUDA(h_entrada, h_salida, ancho, alto, hilosPorBloque, &t);
        else
            filtroLaplacianoCUDA(h_entrada, h_salida, ancho, alto, hilosPorBloque, &t);
        std::cout << "Bloques:  " << t.bloques << " de " << hilosPorBloque << " hilos\n"
                  << "CPU->GPU: " << t.h2d_ms << " ms\n"
                  << "Kernel:   " << t.kernel_ms << " ms\n"
                  << "GPU->CPU: " << t.d2h_ms << " ms\n"
                  << "Conjunto: " << t.total_ms << " ms\n";
    }

    for (int i = 0; i < ancho * alto; i++)
        h_visual[i] = convertirAVisualizacion(h_salida[i], static_cast<TipoFiltro>(filtro));

    std::string carpetaSalida = "Imagenes/final";
    crearDirectorioSiNoExiste(carpetaSalida);
    std::string rutaSalida = carpetaSalida + "/imagen_final.png";

    if (stbi_write_png(rutaSalida.c_str(), ancho, alto, 1, h_visual, ancho)) {
        std::cout << "Imagen guardada en: " << rutaSalida << "\n";
    } else {
        std::cerr << "Error al guardar la imagen\n";
    }

    delete[] h_entrada;
    delete[] h_salida;
    delete[] h_visual;

    std::cout << "Proceso completado.\n";
    return 0;
}
