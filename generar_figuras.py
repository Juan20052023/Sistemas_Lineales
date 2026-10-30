"""
generar_figuras.py
==================
Genera todas las figuras, la tabla de resultados y los audios reconstruidos
para el informe del Trabajo 1 (tema: canal de comunicaciones).

Uso:
    python generar_figuras.py                      # usa un audio sintético de prueba
    python generar_figuras.py --wav mi_audio.wav   # usa tu .wav sin comprimir
    python generar_figuras.py --wav mi_audio.wav --max-muestras 800 --sigma 0.1

Salida (carpeta --salida, por defecto ./salida):
    canal_h_H.png            h(t) y |H(f)| de ambos canales
    senales_<canal>.png      s(t) vs x(t) para cada n
    ojo_<canal>.png          diagramas de ojo (1 x 5)
    ojo_comparativa.png      diagramas de ojo de ambos canales (2 x 5)
    resumen_ber_apertura.png apertura del ojo y BER vs n
    tabla_resultados.csv     parámetros y métricas
    audio/*.wav              audio original cuantizado y audios recibidos
"""
from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from canal_pcm import (NOMBRE_CANAL, TIPOS_CANAL, VALORES_N, bits_a_audio,
                       cargar_fuente, guardar_wav, simular_todo)
from graficas import (dibujar_h, dibujar_H, dibujar_ojo, dibujar_resumen,
                      dibujar_senales)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", default=None, help="archivo .wav sin comprimir (opcional)")
    ap.add_argument("--max-muestras", type=int, default=500,
                    help="muestras de audio a transmitir (por defecto 500)")
    ap.add_argument("--inicio", type=int, default=0, help="muestra inicial del audio")
    ap.add_argument("--bits-por-muestra", type=int, default=8, choices=[8, 16])
    ap.add_argument("--sps", type=int, default=100, help="muestras de simulación por bit")
    ap.add_argument("--sigma", type=float, default=0.0, help="ruido gaussiano a la salida")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--salida", default="salida")
    a = ap.parse_args()

    os.makedirs(os.path.join(a.salida, "audio"), exist_ok=True)

    # ---- 1. Fuente: audio -> bits ------------------------------------------
    bits, fs, x = cargar_fuente(a.wav, a.max_muestras, a.inicio, a.bits_por_muestra)
    print(f"Fuente: {'demo sintético' if a.wav is None else a.wav}")
    print(f"  fs audio = {fs} Hz | muestras = {len(x)} | bits/muestra = "
          f"{a.bits_por_muestra} | bits totales = {len(bits)}")
    print(f"  sps = {a.sps} | sigma = {a.sigma}")

    # ---- 2. Simulación de los dos canales para n = 1,2,4,8,16 --------------
    todos = simular_todo(bits, sps=a.sps, sigma=a.sigma, semilla=a.semilla)

    # ---- 3. Figuras --------------------------------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for j, tipo in enumerate(TIPOS_CANAL):
        por_n = {n: todos[(tipo, n)] for n in VALORES_N}
        dibujar_h(axs[0, j], por_n, tipo)
        dibujar_H(axs[1, j], por_n, tipo)
    fig.tight_layout()
    fig.savefig(os.path.join(a.salida, "canal_h_H.png"), dpi=200)
    plt.close(fig)

    for tipo in TIPOS_CANAL:
        fig, axs = plt.subplots(len(VALORES_N), 1, figsize=(10, 12), sharex=True)
        for ax, n in zip(axs, VALORES_N):
            dibujar_senales(ax, todos[(tipo, n)], n_bits=12)
        axs[0].legend(fontsize=8, ncol=3)
        fig.tight_layout()
        fig.savefig(os.path.join(a.salida, f"senales_{tipo}.png"), dpi=200)
        plt.close(fig)

        fig, axs = plt.subplots(1, len(VALORES_N), figsize=(18, 3.6), sharey=True)
        for ax, n in zip(axs, VALORES_N):
            dibujar_ojo(ax, todos[(tipo, n)])
        fig.suptitle(f"Diagrama de ojo — {NOMBRE_CANAL[tipo]}")
        fig.tight_layout()
        fig.savefig(os.path.join(a.salida, f"ojo_{tipo}.png"), dpi=200)
        plt.close(fig)

    fig, axs = plt.subplots(2, len(VALORES_N), figsize=(18, 7), sharey=True)
    for i, tipo in enumerate(TIPOS_CANAL):
        for j, n in enumerate(VALORES_N):
            dibujar_ojo(axs[i, j], todos[(tipo, n)])
        axs[i, 0].annotate(NOMBRE_CANAL[tipo], xy=(0, 0.5), xytext=(-60, 0),
                           xycoords="axes fraction", textcoords="offset points",
                           rotation=90, va="center", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(a.salida, "ojo_comparativa.png"), dpi=200)
    plt.close(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    dibujar_resumen(ax1, ax2, todos)
    fig.tight_layout()
    fig.savefig(os.path.join(a.salida, "resumen_ber_apertura.png"), dpi=200)
    plt.close(fig)

    # ---- 4. Tabla de resultados -------------------------------------------
    ruta_csv = os.path.join(a.salida, "tabla_resultados.csv")
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["canal", "n", "T [Tb]", "W [1/Tb]", "instante_muestreo [Tb]",
                    "apertura_ojo", "errores", "bits", "BER"])
        print(f"\n{'canal':<12}{'n':>3}{'T':>8}{'W':>6}{'t_mu':>7}{'apertura':>10}"
              f"{'errores':>9}{'BER':>9}")
        for tipo in TIPOS_CANAL:
            for n in VALORES_N:
                r = todos[(tipo, n)]
                t_mu = r.offset / r.sps
                w.writerow([tipo, n, f"{r.T:.4f}", f"{r.W:.0f}", f"{t_mu:.2f}",
                            f"{r.apertura:.4f}", r.errores, len(bits), f"{r.ber:.6f}"])
                print(f"{tipo:<12}{n:>3}{r.T:>8.4f}{r.W:>6.0f}{t_mu:>7.2f}"
                      f"{r.apertura:>10.3f}{r.errores:>9d}{r.ber:>9.2%}")

    # ---- 5. Audios (lo que "se escucha" a la salida del receptor) ----------
    guardar_wav(os.path.join(a.salida, "audio", "original_cuantizado.wav"), fs,
                bits_a_audio(bits, a.bits_por_muestra))
    for (tipo, n), r in todos.items():
        guardar_wav(os.path.join(a.salida, "audio", f"rx_{tipo}_n{n}.wav"), fs,
                    bits_a_audio(r.bits_rx, a.bits_por_muestra))

    print(f"\nListo. Archivos guardados en ./{a.salida}/")


if __name__ == "__main__":
    main()
