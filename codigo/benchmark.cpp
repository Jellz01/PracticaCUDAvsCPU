// Experimentos de rendimiento de la practica. Escribe los registros originales en CSV.
// Uso (desde codigo/): benchmark.exe [carpeta_entrada] [csv_salida] [repeticiones] [calentamientos]
// Solo mide calculo y transferencias: leer archivos y reservar memoria queda fuera.
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "filtros.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

struct Config {
    const char* experimentos;  // una medicion repetida se reutiliza entre experimentos
    TipoFiltro filtro;
    int resolucion;
    int mascara;
    int hilos;  // 0 en CPU
};

static const char* nombreFiltro(TipoFiltro f) {
    return f == FILTRO_MEDIA ? "media" : f == FILTRO_NITIDEZ ? "nitidez" : "laplaciano";
}

static double medirCPU(const Config& c, const unsigned char* e, float* s, int an, int al) {
    auto t0 = std::chrono::steady_clock::now();
    if (c.filtro == FILTRO_MEDIA) aplicarFiltroMedia(e, s, an, al, c.mascara);
    else if (c.filtro == FILTRO_NITIDEZ) aplicarFiltroNitidez(e, s, an, al);
    else aplicarFiltroLaplaciano(e, s, an, al);
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static TiemposCUDA medirGPU(const Config& c, const unsigned char* e, float* s, int an, int al) {
    TiemposCUDA t{};
    if (c.filtro == FILTRO_MEDIA) filtroMediaCUDA(e, s, an, al, c.mascara, c.hilos, &t);
    else if (c.filtro == FILTRO_NITIDEZ) filtroNitidezCUDA(e, s, an, al, c.hilos, &t);
    else filtroLaplacianoCUDA(e, s, an, al, c.hilos, &t);
    return t;
}

int main(int argc, char** argv) {
    std::string entrada = argc > 1 ? argv[1] : "../Imagenes/entrada";
    std::string rutaCsv = argc > 2 ? argv[2] : "../experimentos/crudo.csv";
    int repeticiones = argc > 3 ? std::stoi(argv[3]) : 10;
    int calentamientos = argc > 4 ? std::stoi(argv[4]) : 2;

    const std::vector<Config> cpu = {
        {"filtro;mascara", FILTRO_MEDIA, 4000, 3, 0},
        {"filtro", FILTRO_NITIDEZ, 4000, 3, 0},
        {"filtro", FILTRO_LAPLACIANO, 4000, 3, 0},
        {"resolucion", FILTRO_MEDIA, 2000, 9, 0},
        {"resolucion;mascara;hilos", FILTRO_MEDIA, 4000, 9, 0},
        {"resolucion", FILTRO_MEDIA, 6000, 9, 0},
        {"mascara", FILTRO_MEDIA, 4000, 21, 0},
    };
    const std::vector<Config> gpu = {
        {"filtro;mascara", FILTRO_MEDIA, 4000, 3, 256},
        {"filtro", FILTRO_NITIDEZ, 4000, 3, 256},
        {"filtro", FILTRO_LAPLACIANO, 4000, 3, 256},
        {"resolucion", FILTRO_MEDIA, 2000, 9, 256},
        {"resolucion;mascara;hilos", FILTRO_MEDIA, 4000, 9, 256},
        {"resolucion", FILTRO_MEDIA, 6000, 9, 256},
        {"mascara", FILTRO_MEDIA, 4000, 21, 256},
        {"hilos", FILTRO_MEDIA, 4000, 9, 128},
        {"hilos", FILTRO_MEDIA, 4000, 9, 512},
    };

    // Las imagenes se cargan una sola vez, antes de medir.
    struct Imagen { unsigned char* datos; int ancho; int alto; };
    std::map<int, Imagen> imagenes;
    for (int res : {2000, 4000, 6000}) {
        Imagen img{};
        int canales = 0;
        std::string ruta = entrada + "/" + std::to_string(res) + ".png";
        img.datos = stbi_load(ruta.c_str(), &img.ancho, &img.alto, &canales, 1);
        if (!img.datos || img.ancho != res || img.alto != res) {
            std::cerr << "Imagen invalida: " << ruta << "\n";
            return 1;
        }
        imagenes[res] = img;
    }

    std::filesystem::path rutaArchivo(rutaCsv);
    if (rutaArchivo.has_parent_path()) std::filesystem::create_directories(rutaArchivo.parent_path());
    std::ofstream csv(rutaCsv);
    csv << "implementacion,experimentos,filtro,resolucion,mascara,hilos_por_bloque,bloques,fase,repeticion,"
           "t_cpu_a_gpu,t_kernel,t_gpu_a_cpu,t_conjunto,unidad\n";
    csv.precision(6);
    csv << std::fixed;

    auto fila = [&](const char* impl, const Config& c, int bloques, const char* fase, int rep) -> std::ofstream& {
        csv << impl << ',' << c.experimentos << ',' << nombreFiltro(c.filtro) << ',' << c.resolucion << ','
            << c.mascara << ',' << c.hilos << ',' << bloques << ',' << fase << ',' << rep << ',';
        return csv;
    };

    for (const Config& c : cpu) {
        const Imagen& img = imagenes[c.resolucion];
        std::vector<float> salida(static_cast<size_t>(img.ancho) * img.alto);
        std::cout << "CPU " << nombreFiltro(c.filtro) << " " << c.resolucion << " k=" << c.mascara << std::endl;
        for (int r = 1; r <= calentamientos + repeticiones; r++) {
            double ms = medirCPU(c, img.datos, salida.data(), img.ancho, img.alto);
            bool calentando = r <= calentamientos;
            fila("cpu", c, 0, calentando ? "calentamiento" : "medicion", calentando ? r : r - calentamientos)
                << ",,," << ms << ",ms\n";
            csv.flush();
        }
    }

    for (const Config& c : gpu) {
        const Imagen& img = imagenes[c.resolucion];
        std::vector<float> salida(static_cast<size_t>(img.ancho) * img.alto);
        std::cout << "GPU " << nombreFiltro(c.filtro) << " " << c.resolucion << " k=" << c.mascara
                  << " hilos=" << c.hilos << std::endl;
        for (int r = 1; r <= calentamientos + repeticiones; r++) {
            TiemposCUDA t = medirGPU(c, img.datos, salida.data(), img.ancho, img.alto);
            bool calentando = r <= calentamientos;
            fila("cuda", c, t.bloques, calentando ? "calentamiento" : "medicion", calentando ? r : r - calentamientos)
                << t.h2d_ms << ',' << t.kernel_ms << ',' << t.d2h_ms << ',' << t.total_ms << ",ms\n";
            csv.flush();
        }
    }

    for (auto& par : imagenes) stbi_image_free(par.second.datos);
    std::cout << "Registros en " << rutaCsv << "\n";
    return 0;
}
