"""
graficas.py
===========
Funciones de dibujo compartidas por generar_figuras.py (informe) y app.py (Streamlit).
Todas dibujan sobre ejes de Matplotlib que recibe el llamador.
"""
from __future__ import annotations

import numpy as np

from canal_pcm import (H_teorica, NOMBRE_CANAL, TB, VALORES_N, respuesta_frecuencia,
                       segmentos_ojo)

COLOR = {"exponencial": "tab:blue", "sinc": "tab:red"}


def dibujar_ojo(ax, res, n_trazas=300):
    """Diagrama de ojo: trazas de 2 Tb superpuestas, centradas en el muestreo."""
    t, segs = segmentos_ojo(res.y, res.sps, res.offset, n_trazas, res.descartar)
    ax.plot(t, segs.T, color=COLOR[res.tipo], alpha=0.15, lw=0.6)
    ax.axvline(0, ls="--", color="k", lw=0.8)          # instante de muestreo
    ax.axhline(0.5, ls=":", color="gray", lw=0.8)      # umbral de decisión
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel(r"$t - t_{muestreo}$  [$T_b$]")
    ax.set_ylabel("amplitud")
    ax.set_title(f"n = {res.n} (T = Tb/{res.n})\n"
                 f"apertura = {res.apertura:.2f} | BER = {res.ber:.2%}", fontsize=9)
    ax.grid(alpha=0.3)


def dibujar_senales(ax, res, n_bits=12):
    """s(t) transmitida vs. x(t) a la salida del canal, con instantes de muestreo."""
    N = n_bits * res.sps
    t = np.arange(N) * res.dt
    ax.plot(t, res.s[:N], color="gray", lw=1.2, drawstyle="steps-post", label="s(t) (PCM)")
    ax.plot(t, res.y[:N], color=COLOR[res.tipo], lw=1.5, label="x(t) = s(t) * h(t)")
    idx = np.arange(n_bits) * res.sps + res.offset
    ax.plot(idx * res.dt, res.y[idx], "ko", ms=4, label="muestreo")
    ax.axhline(0.5, ls=":", color="gray", lw=0.8)
    ax.set_xlabel(r"$t$  [$T_b$]")
    ax.set_ylabel("amplitud")
    ax.set_title(f"{NOMBRE_CANAL[res.tipo]} — n = {res.n}", fontsize=9)
    ax.grid(alpha=0.3)


def dibujar_h(ax, resultados, tipo):
    """Respuestas al impulso h(t) del canal para varios n (dict {n: Resultado})."""
    for n, res in resultados.items():
        ax.plot(res.t_h, res.h, lw=1.3, label=f"n = {n}")
    ax.set_xlim((-0.5, 5) if tipo == "exponencial" else (-5, 5))
    ax.set_xlabel(r"$t$  [$T_b$]")
    ax.set_ylabel("h(t)")
    ax.set_title(f"h(t) — {NOMBRE_CANAL[tipo]}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def dibujar_H(ax, resultados, tipo, fmax=None):
    """|H(f)| numérica (línea) y teórica (punteada) para varios n."""
    fmax = fmax or 0.75 * max(resultados)    # el corte de la sinc es n/2 (en 1/Tb)
    for n, res in resultados.items():
        f, H = respuesta_frecuencia(res.h, res.dt)
        linea, = ax.plot(f, H, lw=1.3, label=f"n = {n}")
        ax.plot(f, H_teorica(tipo, f, res.T), ls=":", lw=1, color=linea.get_color())
    ax.set_xlim(0, fmax)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel(r"$f$  [$1/T_b$]")
    ax.set_ylabel("|H(f)|")
    ax.set_title(f"|H(f)| — {NOMBRE_CANAL[tipo]} (punteada: teórica)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def dibujar_resumen(ax_ap, ax_ber, todos, valores_n=VALORES_N):
    """Apertura del ojo y BER en función de n, para ambos canales."""
    for tipo in ("exponencial", "sinc"):
        ap = [todos[(tipo, n)].apertura for n in valores_n]
        be = [todos[(tipo, n)].ber * 100 for n in valores_n]
        ax_ap.plot(valores_n, ap, "o-", color=COLOR[tipo], label=NOMBRE_CANAL[tipo])
        ax_ber.plot(valores_n, be, "o-", color=COLOR[tipo], label=NOMBRE_CANAL[tipo])
    ax_ap.axhline(0, color="k", lw=0.8)
    for ax, ylab, tit in ((ax_ap, "apertura del ojo", "Apertura vertical del ojo"),
                          (ax_ber, "BER [%]", "Tasa de error de bits")):
        ax.set_xscale("log", base=2)
        ax.set_xticks(valores_n)
        ax.set_xticklabels([str(n) for n in valores_n])
        ax.set_xlabel(r"n  ($T = T_b/n$,  $W = n/T_b$)")
        ax.set_ylabel(ylab)
        ax.set_title(tit, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
