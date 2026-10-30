"""
app.py — Interfaz interactiva (Streamlit) del Trabajo 1: canal de comunicaciones.

Los audios de la carpeta `audios/` se cargan automáticamente al abrir la app;
también se puede subir un .wav propio.  La simulación trabaja sobre un
fragmento corto del audio (el tiempo de bit es Tb = 1 s, así que simular el
audio completo requeriría millones de muestras por caso).

Ejecutar:
    pip install -r requirements.txt
    streamlit run app.py
"""
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from canal_pcm import (NOMBRE_CANAL, TIPOS_CANAL, VALORES_N, audio_a_bits,
                       bits_a_audio, cargar_audio, guardar_wav, simular,
                       simular_todo)
from graficas import (dibujar_h, dibujar_H, dibujar_ojo, dibujar_resumen,
                      dibujar_senales)

AUDIOS_DIR = Path(__file__).resolve().parent / "audios"
FS_OBJETIVO = 16000          # frecuencia a la que se simula (mono)
PICO = 0.9                   # margen de normalización del fragmento
BITS_ANALISIS = 8000         # bits usados para ojo, apertura y BER
SPS_AUDIO_MAX = 10           # muestras/bit para el audio (independiente del ojo)
SPS_AUDIO_MIN = 5
MUESTRAS_AUDIO_MAX = 6_000_000

st.set_page_config(page_title="Canal PCM — Diagrama de ojo", layout="wide")
st.title("Transmisión PCM por un canal de ancho de banda limitado")
st.caption("Sistemas Lineales y Señales · Trabajo 1 · x(t) = s(t) ∗ h(t),  "
           "s(t) = Σ b_k g(t − k·Tb),  T = Tb/n,  W = 1/T")


# ----------------------------------------------------------------------------
# Audios incluidos
# ----------------------------------------------------------------------------
def audios_disponibles():
    if not AUDIOS_DIR.is_dir():
        return []
    return sorted(p.name for p in AUDIOS_DIR.glob("*.wav"))


def info_audios():
    ruta = AUDIOS_DIR / "info.json"
    if not ruta.is_file():
        return {}
    try:
        with open(ruta, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


# ----------------------------------------------------------------------------
# Caché de cálculos pesados
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Cargando audio…")
def cargar(bytes_wav):
    archivo = io.BytesIO(bytes_wav) if bytes_wav else None
    return cargar_audio(archivo, fs_objetivo=FS_OBJETIVO)


@st.cache_resource(show_spinner="Simulando los 10 casos (2 canales × 5 valores de n)…")
def correr(bits, sps, sigma, semilla):
    return simular_todo(bits, sps=sps, sigma=sigma, semilla=semilla)


@st.cache_resource(show_spinner="Simulando el audio del fragmento…")
def correr_audio(bits, tipo, n, sps, sigma, semilla):
    return simular(bits, tipo, n, sps=sps, sigma=sigma, semilla=semilla)


def wav_bytes(fs, x):
    buf = io.BytesIO()
    guardar_wav(buf, fs, x)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# Barra lateral: fuente de bits y parámetros
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Fuente de bits")
    disponibles = audios_disponibles()
    modo = st.radio("Origen del audio", ["Audio incluido", "Subir .wav"],
                    horizontal=True)

    bytes_wav = None
    if modo == "Audio incluido":
        if disponibles:
            nombre = st.selectbox("Audio", disponibles)
            bytes_wav = (AUDIOS_DIR / nombre).read_bytes()
            st.audio(bytes_wav, format="audio/wav")
            detalle = next((a for a in info_audios().get("audios", [])
                            if a.get("archivo") == nombre), None)
            desde = (f" (desde {detalle['inicio_s']:.1f} s del original)"
                     if detalle else "")
            dur_inc = detalle["duracion_s"] if detalle else 5.0
            st.caption(f"Fragmento de {dur_inc:.0f} s incluido en la app "
                       f"(16 kHz, mono, PCM 16 bits){desde}.")
        else:
            st.warning("No hay audios en la carpeta `audios/`; se usará el "
                       "audio sintético de prueba.")
    else:
        subido = st.file_uploader("Archivo .wav sin comprimir", type=["wav"])
        if subido is not None:
            bytes_wav = subido.getvalue()
        else:
            st.info("Sin archivo: se usa el audio sintético de prueba.")

    fs, x_full = cargar(bytes_wav)
    dur_total = len(x_full) / fs
    if dur_total < 0.02:
        st.error("El audio es demasiado corto (mínimo 20 ms).")
        st.stop()

    st.header("Fragmento a simular")
    dur_max = max(0.02, min(5.0, dur_total))
    if "dur_s" not in st.session_state:
        st.session_state.dur_s = 1.0
    st.session_state.dur_s = float(
        np.clip(st.session_state.dur_s, 0.02, dur_max))
    if dur_max > 0.0201:
        dur_s = st.slider("Duración [s]", 0.02, float(round(dur_max, 3)),
                          step=0.01, key="dur_s",
                          help="El tiempo de bit es Tb = 1 s; fragmentos más largos "
                               "multiplican los bits y el costo de la simulación.")
    else:
        dur_s = dur_max
        st.caption("Duración: 20 ms (el audio es muy corto).")
    inicio_max = max(0.0, dur_total - dur_s)
    if "inicio_s" not in st.session_state:
        st.session_state.inicio_s = 0.0
    st.session_state.inicio_s = float(
        np.clip(st.session_state.inicio_s, 0.0, inicio_max))
    if inicio_max > 0.001:
        inicio_s = st.slider("Inicio [s]", 0.0, float(round(inicio_max, 3)),
                             step=0.01, key="inicio_s")
    else:
        inicio_s = 0.0
        st.caption("Inicio: 0 s (el fragmento ocupa todo el audio).")

    bps = st.radio("Bits por muestra", [8, 16], horizontal=True)
    normalizar = st.checkbox("Normalizar el fragmento al pico", value=True,
                             help="Usa todo el rango de cuantización. El pico se "
                                  f"fija en {PICO:.1f} para no saturar.")

    st.header("Simulación")
    sps = st.slider("Muestras por bit (ojo/BER)", 20, 200, 50, 10,
                    help="Resolución del diagrama de ojo y de las métricas. "
                         "El audio se simula aparte con menos muestras por bit.")
    sigma = st.slider("Ruido gaussiano σ (a la salida del canal)", 0.0, 0.6,
                      0.0, 0.02)

    st.header("Caso a inspeccionar")
    tipo = st.selectbox("Canal", TIPOS_CANAL,
                        format_func=lambda t: NOMBRE_CANAL[t])
    n = st.select_slider("n  (T = Tb/n)", options=list(VALORES_N), value=2)

# ---- Fragmento -> bits ------------------------------------------------------
i0 = int(round(inicio_s * fs))
i1 = min(len(x_full), i0 + int(round(dur_s * fs)))
x_frag = x_full[i0:i1].astype(np.float64).copy()
if normalizar and np.max(np.abs(x_frag)) > 0:
    x_frag = PICO * x_frag / np.max(np.abs(x_frag))
bits = audio_a_bits(x_frag, bps)

# El ojo y la BER se calculan sobre una ventana fija de bits (rápido y
# representativo); el audio se simula aparte sobre el fragmento completo con
# menos muestras por bit.  Así la duración del audio no encarece el análisis.
n_analisis = min(len(bits), BITS_ANALISIS)
bits_analisis = bits[:n_analisis]
todos = correr(bits_analisis, sps, sigma, 0)
res = todos[(tipo, n)]

sps_audio = int(np.clip(MUESTRAS_AUDIO_MAX // max(1, len(bits)),
                        SPS_AUDIO_MIN, SPS_AUDIO_MAX))
res_audio = correr_audio(bits, tipo, n, sps_audio, sigma, 0)

if bytes_wav is None:
    st.info("Usando un audio sintético de prueba. Elige un audio incluido o "
            "sube tu .wav en la barra lateral.")

st.caption(f"Audio: {fs} Hz · {dur_total:.2f} s · mono · "
           f"fragmento simulado: {dur_s:.2f} s ({len(bits):,} bits) · "
           f"ojo/BER sobre {n_analisis:,} bits · "
           f"audio con {sps_audio} muestras/bit")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Bits transmitidos", f"{len(bits):,}",
          f"{n_analisis:,} en el análisis", delta_color="off")
c2.metric("T  /  W", f"{res.T:.4f} Tb  /  {res.W:.0f}/Tb")
c3.metric("Apertura del ojo", f"{res.apertura:.3f}")
c4.metric("BER", f"{res.ber:.2%}", f"{res.errores} errores", delta_color="off")
st.caption("Apertura y BER corresponden a la ventana de análisis; el audio "
           "se simula sobre el fragmento completo.")

# ---- Forma de onda y fragmento seleccionado --------------------------------
fig, ax = plt.subplots(figsize=(11, 2.2))
t_full = np.arange(len(x_full)) / fs
ax.plot(t_full, x_full, color="0.55", lw=0.6)
ax.axvspan(inicio_s, inicio_s + dur_s, color="tab:blue", alpha=0.18,
           label="fragmento simulado")
ax.set_xlim(0, dur_total)
ax.set_xlabel("tiempo [s]")
ax.set_ylabel("amplitud")
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

tab_sig, tab_ojo, tab_canal, tab_res, tab_audio = st.tabs(
    ["Señales", "Diagrama de ojo", "Canal h(t) y H(f)", "BER y apertura vs n",
     "Audio"])

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
    filas = [{"Canal": NOMBRE_CANAL[t], "n": nn,
              "T [Tb]": round(todos[(t, nn)].T, 4),
              "W [1/Tb]": nn,
              "Apertura": round(todos[(t, nn)].apertura, 3),
              "Errores": todos[(t, nn)].errores,
              "BER [%]": round(todos[(t, nn)].ber * 100, 3)}
             for t in TIPOS_CANAL for nn in VALORES_N]
    st.dataframe(filas, width="stretch", hide_index=True)

# ---- Audio --------------------------------------------------------------------
with tab_audio:
    st.write(f"Fragmento simulado: **{dur_s:.2f} s** desde {inicio_s:.2f} s "
             f"({len(bits):,} bits, {sps_audio} muestras/bit). El ojo y la BER "
             f"se calculan sobre {n_analisis:,} bits.")
    st.write("Con 8 o 16 bits por muestra, un error en un bit alto (MSB) produce "
             "un 'clic' fuerte; el ojo cerrado se **escucha**.")
    ca, cb = st.columns(2)
    with ca:
        st.write("**Original (cuantizado)**")
        st.audio(wav_bytes(fs, bits_a_audio(bits, bps)), format="audio/wav")
    with cb:
        st.write(f"**Recibido** — {NOMBRE_CANAL[tipo]}, n = {n}")
        st.audio(wav_bytes(fs, bits_a_audio(res_audio.bits_rx, bps)),
                 format="audio/wav")
    m = min(len(x_frag), len(bits) // bps)
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.plot(bits_a_audio(bits, bps)[:m], color="gray",
            label="original cuantizado")
    ax.plot(bits_a_audio(res_audio.bits_rx, bps)[:m], color="tab:red", lw=0.9,
            label="recibido")
    ax.set_xlabel("muestra de audio")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
