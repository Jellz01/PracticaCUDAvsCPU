// Pruebas de correccion. Devuelve 0 solo si todas pasan.
// Uso (desde codigo/): pruebas.exe [carpeta_entrada] [carpeta_experimentos]
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include "filtros.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Tolerancia CPU-GPU sobre la salida numerica en float. Cerca de 255, dos float
// consecutivos se separan 1.5e-5: 1e-4 admite solo unos pocos pasos de redondeo
// y queda muy por debajo de 0.5, la diferencia que cambiaria un nivel de gris.
static const double TOLERANCIA = 1e-4;

static int fallos = 0;

static void comprobar(bool ok, const std::string& nombre) {
    std::cout << (ok ? "[OK]    " : "[FALLA] ") << nombre << "\n";
    if (!ok) fallos++;
}

struct Errores {
    double maximo = 0.0;
    double medio = 0.0;
};

static Errores calcularErrores(const std::vector<float>& a, const std::vector<float>& b) {
    Errores e;
    double suma = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        suma += d;
        if (d > e.maximo) e.maximo = d;
    }
    e.medio = suma / a.size();
    return e;
}

static void filtrarCPU(TipoFiltro f, const unsigned char* e, float* s, int an, int al, int k) {
    if (f == FILTRO_MEDIA) aplicarFiltroMedia(e, s, an, al, k);
    else if (f == FILTRO_NITIDEZ) aplicarFiltroNitidez(e, s, an, al);
    else aplicarFiltroLaplaciano(e, s, an, al);
}

static void filtrarGPU(TipoFiltro f, const unsigned char* e, float* s, int an, int al, int k, int hilos) {
    TiemposCUDA t{};
    if (f == FILTRO_MEDIA) filtroMediaCUDA(e, s, an, al, k, hilos, &t);
    else if (f == FILTRO_NITIDEZ) filtroNitidezCUDA(e, s, an, al, hilos, &t);
    else filtroLaplacianoCUDA(e, s, an, al, hilos, &t);
}

static const char* nombreFiltro(TipoFiltro f) {
    return f == FILTRO_MEDIA ? "media" : f == FILTRO_NITIDEZ ? "nitidez" : "laplaciano";
}

// Matriz 3x3 calculada a mano, con extension con ceros. Ver README, seccion Validacion.
static void pruebaManual() {
    std::cout << "\n== Matriz pequena calculada a mano ==\n";
    const std::vector<unsigned char> m = {10, 20, 30, 40, 50, 60, 70, 80, 90};
    const std::vector<float> media = {120 / 9.0f, 210 / 9.0f, 160 / 9.0f, 270 / 9.0f, 450 / 9.0f,
                                      330 / 9.0f, 240 / 9.0f, 390 / 9.0f, 280 / 9.0f};
    const std::vector<float> nitidez = {-10, 10, 70, 70, 50, 130, 230, 190, 310};
    const std::vector<float> laplaciano = {20, 10, -40, -30, 0, -70, -160, -110, -220};

    struct Caso { TipoFiltro f; const std::vector<float>* esperado; };
    const Caso casos[] = {{FILTRO_MEDIA, &media}, {FILTRO_NITIDEZ, &nitidez}, {FILTRO_LAPLACIANO, &laplaciano}};
    for (const Caso& c : casos) {
        std::vector<float> s(9);
        filtrarCPU(c.f, m.data(), s.data(), 3, 3, 3);
        comprobar(calcularErrores(s, *c.esperado).maximo <= TOLERANCIA, std::string("CPU ") + nombreFiltro(c.f));
        for (int hilos : {128, 256, 512}) {
            std::fill(s.begin(), s.end(), -9999.0f);
            filtrarGPU(c.f, m.data(), s.data(), 3, 3, 3, hilos);
            comprobar(calcularErrores(s, *c.esperado).maximo <= TOLERANCIA,
                      std::string("GPU ") + nombreFiltro(c.f) + " " + std::to_string(hilos) + " hilos");
        }
    }

    std::cout << "\n== Conversion a 0-255 ==\n";
    comprobar(convertirAVisualizacion(-10, FILTRO_NITIDEZ) == 0, "nitidez -10 -> 0");
    comprobar(convertirAVisualizacion(310, FILTRO_NITIDEZ) == 255, "nitidez 310 -> 255");
    comprobar(convertirAVisualizacion(-220, FILTRO_LAPLACIANO) == 0, "laplaciano -220 -> 0");
    comprobar(convertirAVisualizacion(-30, FILTRO_LAPLACIANO) == 98, "laplaciano -30 -> 98");
    comprobar(convertirAVisualizacion(200, FILTRO_LAPLACIANO) == 255, "laplaciano 200 -> 255");
    comprobar(convertirAVisualizacion(120 / 9.0f, FILTRO_MEDIA) == 13, "media 13.33 -> 13");
    comprobar(convertirAVisualizacion(160 / 9.0f, FILTRO_MEDIA) == 18, "media 17.78 -> 18");
}

static void compararYRegistrar(std::ofstream& csv, const std::string& prueba, TipoFiltro f,
                               const unsigned char* e, int an, int al, int k, int hilos,
                               const std::vector<float>& cpu) {
    std::vector<float> gpu(cpu.size(), -9999.0f);
    filtrarGPU(f, e, gpu.data(), an, al, k, hilos);
    Errores err = calcularErrores(cpu, gpu);
    size_t pixeles = static_cast<size_t>(an) * al;
    bool ok = err.maximo <= TOLERANCIA;
    csv << prueba << ',' << nombreFiltro(f) << ',' << an << ',' << al << ',' << k << ',' << hilos << ','
        << (pixeles + hilos - 1) / hilos << ',' << pixeles % hilos << ',' << err.maximo << ',' << err.medio << ','
        << TOLERANCIA << ',' << (ok ? "si" : "no") << "\n";
    comprobar(ok, prueba + " " + nombreFiltro(f) + " " + std::to_string(an) + "x" + std::to_string(al) + " k=" +
                      std::to_string(k) + " hilos=" + std::to_string(hilos) + " errMax=" + std::to_string(err.maximo));
}

// Dimensiones cuyo total de pixeles no es multiplo de ningun tamano de bloque.
static void pruebaNoMultiplos(std::ofstream& csv) {
    std::cout << "\n== Dimensiones no multiplos de los hilos por bloque ==\n";
    const int tamanos[][2] = {{1, 1}, {7, 5}, {513, 1}, {1, 257}, {1001, 777}};
    const TipoFiltro filtros[] = {FILTRO_MEDIA, FILTRO_NITIDEZ, FILTRO_LAPLACIANO};
    for (const auto& dim : tamanos) {
        int an = dim[0], al = dim[1];
        std::vector<unsigned char> img(static_cast<size_t>(an) * al);
        unsigned int estado = 12345u;  // generador fijo: siempre los mismos datos
        for (auto& p : img) {
            estado = estado * 1103515245u + 12345u;
            p = static_cast<unsigned char>(estado >> 16);
        }
        std::vector<float> cpu(img.size());
        for (TipoFiltro f : filtros) {
            std::vector<int> mascaras = (f == FILTRO_MEDIA) ? std::vector<int>{3, 9, 21} : std::vector<int>{3};
            for (int k : mascaras) {
                filtrarCPU(f, img.data(), cpu.data(), an, al, k);
                for (int hilos : {128, 256, 512})
                    compararYRegistrar(csv, "no_multiplo", f, img.data(), an, al, k, hilos, cpu);
            }
        }
    }
}

static void pruebaImagenes(std::ofstream& csv, const std::string& entrada, const std::string& salidaImg) {
    std::cout << "\n== Imagenes reales: CPU contra GPU ==\n";
    struct Config { TipoFiltro f; int res; int k; int hilos; };
    const Config configs[] = {
        {FILTRO_MEDIA, 4000, 3, 256},  {FILTRO_NITIDEZ, 4000, 3, 256}, {FILTRO_LAPLACIANO, 4000, 3, 256},
        {FILTRO_MEDIA, 2000, 9, 256},  {FILTRO_MEDIA, 4000, 9, 256},   {FILTRO_MEDIA, 6000, 9, 256},
        {FILTRO_MEDIA, 4000, 21, 256}, {FILTRO_MEDIA, 4000, 9, 128},   {FILTRO_MEDIA, 4000, 9, 512},
    };
    for (const Config& c : configs) {
        int an = 0, al = 0, canales = 0;
        std::string ruta = entrada + "/" + std::to_string(c.res) + ".png";
        unsigned char* img = stbi_load(ruta.c_str(), &an, &al, &canales, 1);
        if (!img) {
            comprobar(false, "no se pudo leer " + ruta);
            continue;
        }
        comprobar(an == c.res && al == c.res, ruta + " tiene el tamano esperado");
        std::vector<float> cpu(static_cast<size_t>(an) * al);
        filtrarCPU(c.f, img, cpu.data(), an, al, c.k);
        compararYRegistrar(csv, "imagen", c.f, img, an, al, c.k, c.hilos, cpu);
        stbi_image_free(img);
    }

    // Imagenes representativas para el informe, convertidas despues de validar.
    std::filesystem::create_directories(salidaImg);
    int an = 0, al = 0, canales = 0;
    unsigned char* img = stbi_load((entrada + "/2000.png").c_str(), &an, &al, &canales, 1);
    if (!img) {
        comprobar(false, "no se pudo leer la imagen de 2000");
        return;
    }
    struct Muestra { TipoFiltro f; int k; const char* nombre; };
    const Muestra muestras[] = {{FILTRO_MEDIA, 3, "media_3"}, {FILTRO_MEDIA, 21, "media_21"},
                                {FILTRO_NITIDEZ, 3, "nitidez"}, {FILTRO_LAPLACIANO, 3, "laplaciano"}};
    std::vector<float> gpu(static_cast<size_t>(an) * al);
    std::vector<unsigned char> vis(gpu.size());
    for (const Muestra& m : muestras) {
        filtrarGPU(m.f, img, gpu.data(), an, al, m.k, 256);
        for (size_t i = 0; i < gpu.size(); i++) vis[i] = convertirAVisualizacion(gpu[i], m.f);
        std::string ruta = salidaImg + "/" + m.nombre + "_2000.png";
        comprobar(stbi_write_png(ruta.c_str(), an, al, 1, vis.data(), an) != 0, "guardar " + ruta);
    }
    stbi_image_free(img);
}

int main(int argc, char** argv) {
    std::string entrada = argc > 1 ? argv[1] : "../Imagenes/entrada";
    std::string experimentos = argc > 2 ? argv[2] : "../experimentos";
    std::filesystem::create_directories(experimentos);

    std::ofstream csv(experimentos + "/validacion.csv");
    csv << "prueba,filtro,ancho,alto,mascara,hilos_por_bloque,bloques,resto_pixeles_entre_hilos,"
           "error_abs_max,error_abs_medio,tolerancia,cumple\n";

    pruebaManual();
    pruebaNoMultiplos(csv);
    pruebaImagenes(csv, entrada, experimentos + "/imagenes");

    std::cout << "\n" << (fallos == 0 ? std::string("TODAS LAS PRUEBAS PASARON") : "HAY FALLOS: " + std::to_string(fallos)) << "\n";
    return fallos == 0 ? 0 : 1;
}
