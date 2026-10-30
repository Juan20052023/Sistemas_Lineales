"""
app.py — Interfaz interactiva (Streamlit) del Trabajo 1: canal de comunicaciones.

Ejecutar:
    pip install -r requirements.txt
    streamlit run app.py
"""
import io

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from canal_pcm import (NOMBRE_CANAL, TIPOS_CANAL, VALORES_N, bits_a_audio,
                       cargar_fuente, guardar_wav, simular_todo)
from graficas import (dibujar_h, dibujar_H, dibujar_ojo, dibujar_resumen,
                      dibujar_senales)

st.set_page_config(page_title="Canal PCM — Diagrama de ojo", layout="wide")
st.title("Transmisión PCM por un canal de ancho de banda limitado")
st.caption("Sistemas Lineales y Señales · Trabajo 1 · x(t) = s(t) ∗ h(t),  "
           "s(t) = Σ b_k g(t − k·Tb),  T = Tb/n,  W = 1/T")


# ----------------------------------------------------------------------------
# Caché de cálculos pesados
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Leyendo audio…")
def cargar(bytes_wav, max_muestras, inicio, bps):
    archivo = io.BytesIO(bytes_wav) if bytes_wav else None
    return cargar_fuente(archivo, max_muestras, inicio, bps)


@st.cache_data(show_spinner="Simulando los 10 casos (2 canales × 5 valores de n)…")
def correr(bits, sps, sigma, semilla):
    return simular_todo(bits, sps=sps, sigma=sigma, semilla=semilla)


def wav_bytes(fs, x):
    buf = io.BytesIO()
    guardar_wav(buf, fs, x)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# Barra lateral: parámetros
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Fuente de bits")
    subido = st.file_uploader("Archivo .wav sin comprimir", type=["wav"])
    max_muestras = st.slider("Muestras de audio a transmitir", 50, 2000, 400, 50,
                             help="Cada muestra aporta 8 o 16 bits. Más muestras = "
                                  "más bits = simulación más lenta.")
    inicio = st.number_input("Muestra inicial", 0, 10_000_000, 0, 1000)
    bps = st.radio("Bits por muestra", [8, 16], horizontal=True)

    st.header("Simulación")
    sps = st.slider("Muestras de simulación por bit", 40, 200, 100, 20)
    sigma = st.slider("Ruido gaussiano σ (a la salida del canal)", 0.0, 0.6, 0.0, 0.02)

    st.header("Caso a inspeccionar")
    tipo = st.selectbox("Canal", TIPOS_CANAL, format_func=lambda t: NOMBRE_CANAL[t])
    n = st.select_slider("n  (T = Tb/n)", options=list(VALORES_N), value=2)

bits, fs, x_audio = cargar(subido.getvalue() if subido else None,
                           max_muestras, int(inicio), bps)
todos = correr(bits, sps, sigma, 0)
res = todos[(tipo, n)]

if subido is None:
    st.info("Usando un audio sintético de prueba. Sube tu .wav en la barra lateral.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Bits transmitidos", f"{len(bits):,}")
c2.metric("T  /  W", f"{res.T:.4f} Tb  /  {res.W:.0f}/Tb")
c3.metric("Apertura del ojo", f"{res.apertura:.3f}")
c4.metric("BER", f"{res.ber:.2%}", f"{res.errores} errores", delta_color="off")

tab_sig, tab_ojo, tab_canal, tab_res, tab_audio = st.tabs(
    ["Señales", "Diagrama de ojo", "Canal h(t) y H(f)", "BER y apertura vs n", "Audio"])

# ---- Señales ----------------------------------------------------------------
with tab_sig:
    n_bits = st.slider("Bits a mostrar", 6, 40, 14)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    dibujar_senales(ax, res, n_bits)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Los puntos negros son los instantes de muestreo óptimos "
               f"(t = {res.offset / res.sps:.2f} Tb dentro de cada bit); "
               "la línea punteada es el umbral de decisión 0.5.")

# ---- Ojo ----------------------------------------------------------------------
with tab_ojo:
    n_trazas = st.slider("Trazas superpuestas", 50, 1000, 300, 50)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    dibujar_ojo(ax, res, n_trazas)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Comparación para todos los n")
    fig, axs = plt.subplots(2, len(VALORES_N), figsize=(18, 7), sharey=True)
    for i, t in enumerate(TIPOS_CANAL):
        for j, nn in enumerate(VALORES_N):
            dibujar_ojo(axs[i, j], todos[(t, nn)], 200)
        axs[i, 0].annotate(NOMBRE_CANAL[t], xy=(0, 0.5), xytext=(-60, 0),
                           xycoords="axes fraction", textcoords="offset points",
                           rotation=90, va="center", ha="center", fontsize=10)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ---- Canal --------------------------------------------------------------------
with tab_canal:
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    for j, t in enumerate(TIPOS_CANAL):
        por_n = {nn: todos[(t, nn)] for nn in VALORES_N}
        dibujar_h(axs[0, j], por_n, t)
        dibujar_H(axs[1, j], por_n, t)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    area = res.h.sum() * res.dt
    st.caption(f"Ganancia DC del caso seleccionado: ∫h(t)dt ≈ {area:.4f} "
               "(la h(t) discreta se normaliza para que sea 1).")

# ---- Resumen ------------------------------------------------------------------
with tab_res:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    dibujar_resumen(a1, a2, todos)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    filas = [{"Canal": NOMBRE_CANAL[t], "n": nn, "T [Tb]": round(todos[(t, nn)].T, 4),
              "W [1/Tb]": nn, "Apertura": round(todos[(t, nn)].apertura, 3),
              "Errores": todos[(t, nn)].errores,
              "BER [%]": round(todos[(t, nn)].ber * 100, 3)}
             for t in TIPOS_CANAL for nn in VALORES_N]
    st.dataframe(filas, width="stretch", hide_index=True)

# ---- Audio --------------------------------------------------------------------
with tab_audio:
    st.write("Con 8 o 16 bits por muestra, un error en un bit alto (MSB) produce un "
             "'clic' fuerte; el ojo cerrado se **escucha**.")
    ca, cb = st.columns(2)
    with ca:
        st.write("**Original (cuantizado)**")
        st.audio(wav_bytes(fs, bits_a_audio(bits, bps)), format="audio/wav")
    with cb:
        st.write(f"**Recibido** — {NOMBRE_CANAL[tipo]}, n = {n}")
        st.audio(wav_bytes(fs, bits_a_audio(res.bits_rx, bps)), format="audio/wav")
    m = min(len(x_audio), len(bits) // bps)
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(bits_a_audio(bits, bps)[:m], color="gray", label="original cuantizado")
    ax.plot(bits_a_audio(res.bits_rx, bps)[:m], color="tab:red", lw=0.9, label="recibido")
    ax.set_xlabel("muestra de audio")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
