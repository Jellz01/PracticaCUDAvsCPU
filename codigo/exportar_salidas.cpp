// Agregado para la practica 02. Guarda las salidas float de C/C++ secuencial y CUDA C
// en archivos .f32 (float32 crudo, fila mayor) para que pruebas.py de la practica 02
// compare las cuatro versiones contra C/C++. Usa filtros.cpp y filtros.cu sin cambios.
//
// Uso (desde codigo/): exportar_salidas.exe <carpeta_entrada> <carpeta_salida>
//   exportar_salidas.exe ..\Imagenes\entrada C:\Users\fabia\PycharmProjects\PracticaPyCuda\referencias
// Nombre de cada archivo: <etiqueta>_<filtro>_k<k>_<cpu|cudaHILOS>.f32
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "filtros.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static const char* nombreFiltro(TipoFiltro f) {
    return f == FILTRO_MEDIA ? "media" : f == FILTRO_NITIDEZ ? "nitidez" : "laplaciano";
}

static std::string carpeta;

static void guardar(const std::string& nombre, const std::vector<float>& datos) {
    std::ofstream f(carpeta + "/" + nombre, std::ios::binary);
    f.write(reinterpret_cast<const char*>(datos.data()), datos.size() * sizeof(float));
    if (!f) {
        std::cerr << "No se pudo escribir " << nombre << "\n";
        std::exit(1);
    }
}

static void exportar(const std::string& etiqueta, const unsigned char* img, int an, int al,
                     TipoFiltro f, int k, const std::vector<int>& hilos) {
    std::vector<float> s(static_cast<size_t>(an) * al);
    std::string base = etiqueta + "_" + nombreFiltro(f) + "_k" + std::to_string(k);
    if (f == FILTRO_MEDIA) aplicarFiltroMedia(img, s.data(), an, al, k);
    else if (f == FILTRO_NITIDEZ) aplicarFiltroNitidez(img, s.data(), an, al);
    else aplicarFiltroLaplaciano(img, s.data(), an, al);
    guardar(base + "_cpu.f32", s);
    for (int h : hilos) {
        std::fill(s.begin(), s.end(), -9999.0f);
        TiemposCUDA t{};
        if (f == FILTRO_MEDIA) filtroMediaCUDA(img, s.data(), an, al, k, h, &t);
        else if (f == FILTRO_NITIDEZ) filtroNitidezCUDA(img, s.data(), an, al, h, &t);
        else filtroLaplacianoCUDA(img, s.data(), an, al, h, &t);
        guardar(base + "_cuda" + std::to_string(h) + ".f32", s);
    }
    std::cout << "[OK] " << base << " " << an << "x" << al << "\n";
}

struct Caso { TipoFiltro f; int k; };
static const Caso TODOS[] = {{FILTRO_MEDIA, 3}, {FILTRO_MEDIA, 9}, {FILTRO_MEDIA, 21},
                             {FILTRO_NITIDEZ, 3}, {FILTRO_LAPLACIANO, 3}};

int main(int argc, char** argv) {
    std::string entrada = argc > 1 ? argv[1] : "../Imagenes/entrada";
    carpeta = argc > 2 ? argv[2] : "../referencias";
    std::filesystem::create_directories(carpeta);
    const std::vector<int> h256 = {256};
    const std::vector<int> hTodos = {128, 256, 512};

    // Matriz 3x3 calculada a mano (README de la practica 01).
    const unsigned char m[9] = {10, 20, 30, 40, 50, 60, 70, 80, 90};
    exportar("manual", m, 3, 3, FILTRO_MEDIA, 3, hTodos);
    exportar("manual", m, 3, 3, FILTRO_NITIDEZ, 3, hTodos);
    exportar("manual", m, 3, 3, FILTRO_LAPLACIANO, 3, hTodos);

    // Mismas dimensiones no multiplos y mismo generador que pruebas.cpp.
    const int tamanos[][2] = {{1, 1}, {7, 5}, {513, 1}, {1, 257}, {1001, 777}};
    for (const auto& d : tamanos) {
        int an = d[0], al = d[1];
        std::vector<unsigned char> img(static_cast<size_t>(an) * al);
        unsigned int estado = 12345u;
        for (auto& p : img) {
            estado = estado * 1103515245u + 12345u;
            p = static_cast<unsigned char>(estado >> 16);
        }
        std::string etiqueta = "lcg" + std::to_string(an) + "x" + std::to_string(al);
        for (const Caso& c : TODOS) exportar(etiqueta, img.data(), an, al, c.f, c.k, hTodos);
    }

    // Imagenes reales de la practica.
    for (int res : {2000, 4000, 6000}) {
        int an = 0, al = 0, canales = 0;
        std::string ruta = entrada + "/" + std::to_string(res) + ".png";
        unsigned char* img = stbi_load(ruta.c_str(), &an, &al, &canales, 1);
        if (!img || an != res || al != res) {
            std::cerr << "Imagen invalida: " << ruta << "\n";
            return 1;
        }
        std::string etiqueta = "img" + std::to_string(res);
        if (res == 6000) {
            exportar(etiqueta, img, an, al, FILTRO_MEDIA, 9, h256);
        } else {
            for (const Caso& c : TODOS) {
                bool hilos = (res == 4000 && c.f == FILTRO_MEDIA && c.k == 9);
                exportar(etiqueta, img, an, al, c.f, c.k, hilos ? hTodos : h256);
            }
        }
        stbi_image_free(img);
    }
    std::cout << "Referencias en " << carpeta << "\n";
    return 0;
}
