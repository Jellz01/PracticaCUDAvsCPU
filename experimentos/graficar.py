"""Resume crudo.csv y genera las graficas del informe.

Uso (desde la raiz del repositorio): python experimentos/graficar.py
Salidas: experimentos/resumen.csv y experimentos/graficas/*.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CARPETA = Path(__file__).resolve().parent
GRAFICAS = CARPETA / "graficas"
GRAFICAS.mkdir(exist_ok=True)

COLOR_CPU = "#2a78d6"
COLOR_CUDA = "#eb6834"
COLOR_KERNEL = "#1baf7a"
COLOR_H2D = "#6250d6"
COLOR_D2H = "#eda100"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"
FONDO = "#fcfcfb"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 11,
    "figure.facecolor": FONDO,
    "axes.facecolor": FONDO,
    "axes.edgecolor": EJE,
    "axes.labelcolor": TINTA_2,
    "axes.titlecolor": TINTA,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": REJILLA,
    "grid.linewidth": 0.8,
    "xtick.color": TINTA_2,
    "ytick.color": TINTA_2,
    "legend.frameon": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

CLAVES = ["implementacion", "filtro", "resolucion", "mascara", "hilos_por_bloque", "bloques"]
TIEMPOS = ["t_cpu_a_gpu", "t_kernel", "t_gpu_a_cpu", "t_conjunto"]


def resumir(crudo):
    medidas = crudo[crudo.fase == "medicion"]
    filas = []
    for clave, grupo in medidas.groupby(CLAVES, sort=False):
        fila = dict(zip(CLAVES, clave))
        fila["repeticiones"] = len(grupo)
        for t in TIEMPOS:
            v = grupo[t].dropna()
            if v.empty:
                continue
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            fila[f"{t}_mediana_ms"] = med
            fila[f"{t}_q1_ms"] = q1
            fila[f"{t}_q3_ms"] = q3
            fila[f"{t}_iqr_ms"] = q3 - q1
            fila[f"{t}_desv_ms"] = v.std(ddof=1)
            fila[f"{t}_cv_pct"] = 100 * v.std(ddof=1) / v.mean()
        filas.append(fila)
    resumen = pd.DataFrame(filas)

    cpu = resumen[resumen.implementacion == "cpu"].set_index(["filtro", "resolucion", "mascara"])
    def secuencial(f):
        return cpu.loc[(f.filtro, f.resolucion, f.mascara), "t_conjunto_mediana_ms"]

    es_cuda = resumen.implementacion == "cuda"
    base = resumen[es_cuda].apply(secuencial, axis=1)
    resumen.loc[es_cuda, "aceleracion_computo"] = base / resumen.loc[es_cuda, "t_kernel_mediana_ms"]
    resumen.loc[es_cuda, "aceleracion_con_transferencias"] = base / resumen.loc[es_cuda, "t_conjunto_mediana_ms"]
    # Las medianas no se suman exacto al tiempo conjunto, asi que la proporcion
    # se calcula sobre la suma de las tres medianas, igual que en la grafica 4.
    transferencias = resumen.loc[es_cuda, "t_cpu_a_gpu_mediana_ms"] + resumen.loc[es_cuda, "t_gpu_a_cpu_mediana_ms"]
    resumen.loc[es_cuda, "pct_transferencias"] = (
        100 * transferencias / (transferencias + resumen.loc[es_cuda, "t_kernel_mediana_ms"])
    )
    return resumen


def fila(resumen, impl, filtro, res, k, hilos=None):
    m = (resumen.implementacion == impl) & (resumen.filtro == filtro) & (resumen.resolucion == res) & (resumen.mascara == k)
    if impl == "cuda":
        m &= resumen.hilos_por_bloque == (256 if hilos is None else hilos)
    return resumen[m].iloc[0]


def segundos_o_ms(ms):
    return f"{ms / 1000:.2f} s" if ms >= 1000 else f"{ms:.1f} ms"


def grafica_filtros(r):
    filtros = ["media", "nitidez", "laplaciano"]
    nombres = ["Media 3×3", "Nitidez", "Laplaciano"]
    cpu = [fila(r, "cpu", f, 4000, 3).t_conjunto_mediana_ms for f in filtros]
    gpu = [fila(r, "cuda", f, 4000, 3).t_conjunto_mediana_ms for f in filtros]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(3)
    a = 0.34
    b1 = ax.bar(x - a / 2 - 0.01, cpu, a, color=COLOR_CPU, label="CPU secuencial")
    b2 = ax.bar(x + a / 2 + 0.01, gpu, a, color=COLOR_CUDA, label="CUDA (transferencias + kernel)")
    for barras, valores in ((b1, cpu), (b2, gpu)):
        for barra, v in zip(barras, valores):
            ax.annotate(segundos_o_ms(v), (barra.get_x() + barra.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", color=TINTA, fontsize=10)
    ax.set_xticks(x, nombres)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Mediana (ms)")
    ax.set_title("Tiempo por filtro, imagen de 4000×4000")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.savefig(GRAFICAS / "1_filtros.png")
    plt.close(fig)


def linea_con_iqr(ax, x, cpu, gpu, kernel):
    for filas, col, color, etiqueta, marca in (
        (cpu, "t_conjunto", COLOR_CPU, "CPU secuencial", "o"),
        (gpu, "t_conjunto", COLOR_CUDA, "CUDA conjunto", "s"),
        (kernel, "t_kernel", COLOR_KERNEL, "CUDA solo kernel", "^"),
    ):
        y = [f[f"{col}_mediana_ms"] for f in filas]
        e = [[f[f"{col}_mediana_ms"] - f[f"{col}_q1_ms"] for f in filas],
             [f[f"{col}_q3_ms"] - f[f"{col}_mediana_ms"] for f in filas]]
        ax.errorbar(x, y, yerr=e, color=color, marker=marca, markersize=7, linewidth=2, capsize=3, label=etiqueta)
        ax.annotate(segundos_o_ms(y[-1]), (x[-1], y[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=TINTA, fontsize=10)
    ax.set_yscale("log")
    ax.set_ylabel("Mediana y cuartiles (ms, escala log.)")
    ax.set_xticks(x)
    ax.margins(x=0.14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3)


def grafica_resolucion(r):
    res = [2000, 4000, 6000]
    cpu = [fila(r, "cpu", "media", n, 9) for n in res]
    gpu = [fila(r, "cuda", "media", n, 9) for n in res]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    linea_con_iqr(ax, [n * n / 1e6 for n in res], cpu, gpu, gpu)
    ax.set_xticklabels(["2000×2000\n(4 MP)", "4000×4000\n(16 MP)", "6000×6000\n(36 MP)"])
    ax.set_title("Crecimiento con la resolución, media 9×9")
    fig.savefig(GRAFICAS / "2_resolucion.png")
    plt.close(fig)


def grafica_mascara(r):
    ks = [3, 9, 21]
    cpu = [fila(r, "cpu", "media", 4000, k) for k in ks]
    gpu = [fila(r, "cuda", "media", 4000, k) for k in ks]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    linea_con_iqr(ax, [k * k for k in ks], cpu, gpu, gpu)
    ax.set_xticklabels(["3×3\n(9)", "9×9\n(81)", "21×21\n(441)"])
    ax.set_xlabel("Máscara (lecturas por píxel)")
    ax.set_title("Trabajo por píxel, media en 4000×4000")
    fig.savefig(GRAFICAS / "3_mascara.png")
    plt.close(fig)


def configuraciones_cuda(r):
    salida = [
        ("Media 3×3, 4000", fila(r, "cuda", "media", 4000, 3)),
        ("Nitidez, 4000", fila(r, "cuda", "nitidez", 4000, 3)),
        ("Laplaciano, 4000", fila(r, "cuda", "laplaciano", 4000, 3)),
        ("Media 9×9, 2000", fila(r, "cuda", "media", 2000, 9)),
        ("Media 9×9, 4000", fila(r, "cuda", "media", 4000, 9)),
        ("Media 9×9, 6000", fila(r, "cuda", "media", 6000, 9)),
        ("Media 21×21, 4000", fila(r, "cuda", "media", 4000, 21)),
    ]
    return list(reversed(salida))


def grafica_transferencias(r):
    conf = configuraciones_cuda(r)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    izquierda = np.zeros(len(conf))
    y = np.arange(len(conf))
    for col, color, etiqueta, tinta in (
        ("t_cpu_a_gpu", COLOR_H2D, "CPU→GPU", "#ffffff"),
        ("t_kernel", COLOR_KERNEL, "Kernel", TINTA),
        ("t_gpu_a_cpu", COLOR_D2H, "GPU→CPU", TINTA),
    ):
        partes = []
        for _, f in conf:
            suma = f.t_cpu_a_gpu_mediana_ms + f.t_kernel_mediana_ms + f.t_gpu_a_cpu_mediana_ms
            partes.append(100 * f[f"{col}_mediana_ms"] / suma)
        partes = np.array(partes)
        ax.barh(y, partes, left=izquierda, color=color, label=etiqueta, height=0.62, edgecolor=FONDO, linewidth=1.5)
        for i, p in enumerate(partes):
            if p >= 7:
                ax.text(izquierda[i] + p / 2, i, f"{p:.0f}%", ha="center", va="center", color=tinta, fontsize=10)
        izquierda += partes
    ax.set_yticks(y, [n for n, _ in conf])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Porcentaje del tiempo en GPU")
    ax.grid(axis="y", visible=False)
    ax.set_title("Reparto del tiempo CUDA, 256 hilos por bloque")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=3)
    fig.savefig(GRAFICAS / "4_transferencias.png")
    plt.close(fig)


def grafica_hilos(crudo):
    m = crudo[(crudo.implementacion == "cuda") & (crudo.fase == "medicion") & (crudo.filtro == "media")
              & (crudo.resolucion == 4000) & (crudo.mascara == 9)]
    hilos = [128, 256, 512]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    aleatorio = np.random.default_rng(1)
    for i, h in enumerate(hilos):
        v = m[m.hilos_por_bloque == h].t_kernel.to_numpy()
        ax.scatter(i + aleatorio.uniform(-0.09, 0.09, len(v)), v, s=30, color=COLOR_KERNEL, alpha=0.55, linewidths=0,
                   label="Repetición" if i == 0 else None)
        mediana = np.median(v)
        ax.hlines(mediana, i - 0.22, i + 0.22, color=TINTA, linewidth=2, label="Mediana" if i == 0 else None)
        ax.annotate(f"{mediana:.2f} ms", (i + 0.24, mediana), va="center", color=TINTA, fontsize=10)
    ax.set_xticks(range(3), [f"{h} hilos" for h in hilos])
    ax.set_xlim(-0.5, 2.6)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Tiempo del kernel (ms)")
    ax.set_title("Hilos por bloque, media 9×9 en 4000×4000")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.savefig(GRAFICAS / "5_hilos.png")
    plt.close(fig)


def grafica_aceleracion(r):
    conf = configuraciones_cuda(r)
    y = np.arange(len(conf))
    computo = [f.aceleracion_computo for _, f in conf]
    con_transf = [f.aceleracion_con_transferencias for _, f in conf]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    a = 0.36
    b1 = ax.barh(y + a / 2 + 0.01, computo, a, color=COLOR_KERNEL, label="De cómputo (solo kernel)")
    b2 = ax.barh(y - a / 2 - 0.01, con_transf, a, color=COLOR_CUDA, label="Con transferencias")
    for barras, valores in ((b1, computo), (b2, con_transf)):
        for barra, v in zip(barras, valores):
            ax.annotate(f"{v:.1f}×", (v, barra.get_y() + barra.get_height() / 2), xytext=(4, 0),
                        textcoords="offset points", va="center", color=TINTA, fontsize=9)
    ax.set_yticks(y, [n for n, _ in conf])
    ax.grid(axis="y", visible=False)
    ax.axvline(1, color=TINTA_2, linewidth=1)
    ax.margins(x=0.1)
    ax.set_xlabel("Aceleración respecto a la CPU secuencial (1× = igual)")
    ax.set_title("Aceleración, 256 hilos por bloque")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.14), ncol=2)
    fig.savefig(GRAFICAS / "6_aceleracion.png")
    plt.close(fig)


def main():
    crudo = pd.read_csv(CARPETA / "crudo.csv")
    resumen = resumir(crudo)
    resumen.to_csv(CARPETA / "resumen.csv", index=False, float_format="%.4f")
    grafica_filtros(resumen)
    grafica_resolucion(resumen)
    grafica_mascara(resumen)
    grafica_transferencias(resumen)
    grafica_hilos(crudo)
    grafica_aceleracion(resumen)
    print(f"Resumen y graficas en {CARPETA}")


if __name__ == "__main__":
    main()
