"""
canal_pcm.py
============
Núcleo de la simulación del Trabajo 1 (Sistemas Lineales y Señales):
transmisión de una señal PCM a través de un canal LTI de ancho de banda limitado.

Modelo (todas las unidades de tiempo están normalizadas a Tb = 1 s):

    s(t) = sum_k b_k g(t - k Tb),      g(t) = u(t) - u(t - Tb)
    x(t) = s(t) * h(t)                 (convolución)

con dos modelos de canal:
    exponencial (RC):  h(t) = (1/T) exp(-t/T) u(t)
    sinc:              h(t) = (1/T) sinc(t/T),   sinc(x) = sin(pi x)/(pi x)

y T = Tb/n, con n = 1, 2, 4, 8, 16.  El ancho de banda aproximado es W = 1/T.

Discretización: cada bit se representa con `sps` muestras (samples per symbol),
por lo que el paso de simulación es dt = Tb/sps.  La convolución continua se
aproxima por  y[i] = dt * sum_j s[j] h[i-j].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve, resample_poly

TB = 1.0                                   # tiempo de bit [s]
VALORES_N = (1, 2, 4, 8, 16)               # valores de n pedidos en el enunciado
TIPOS_CANAL = ("exponencial", "sinc")
NOMBRE_CANAL = {
    "exponencial": "Exponencial (pasa-bajos RC)",
    "sinc": "Sinc (pasa-bajos ideal)",
}


# ----------------------------------------------------------------------------
# 1. Audio (.wav)  <->  bits
# ----------------------------------------------------------------------------
def leer_wav(archivo):
    """Lee un .wav sin comprimir (ruta o archivo en memoria).

    Devuelve (fs, x) con x en float64 dentro de [-1, 1] y en mono
    (si es estéreo se promedian los canales).
    """
    fs, datos = wavfile.read(archivo)
    if datos.dtype == np.uint8:
        x = (datos.astype(np.float64) - 128.0) / 128.0
    elif datos.dtype == np.int16:
        x = datos.astype(np.float64) / 32768.0
    elif datos.dtype == np.int32:
        x = datos.astype(np.float64) / 2147483648.0
    else:                                   # float32 / float64
        x = datos.astype(np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return fs, x


def guardar_wav(destino, fs, x):
    """Guarda x (float en [-1, 1]) como .wav de 16 bits (ruta o BytesIO)."""
    wavfile.write(destino, int(fs), (np.clip(x, -1, 1) * 32767).astype(np.int16))


def remuestrear_audio(x, fs_in, fs_out):
    """Baja la frecuencia de muestreo a `fs_out` conservando la duración.

    Solo remuestrea si fs_in > fs_out (no sobremuestrea: no aporta información
    y multiplicaría el costo de la simulación).  Se usa resample_poly, que
    aplica un filtro antialias antes de diezmar: sin él, al bajar de 48 kHz a
    16 kHz el contenido de alta frecuencia se plegaría sobre la banda útil
    (aliasing).
    """
    if fs_out is None or int(fs_in) <= int(fs_out):
        return x, fs_in
    g = int(np.gcd(int(fs_in), int(fs_out)))
    y = resample_poly(x, int(fs_out) // g, int(fs_in) // g)
    return y, int(fs_out)


def cargar_audio(archivo=None, fs_objetivo=None):
    """Carga un .wav (o el demo sintético) en mono, remuestreado y sin recortar.

    Devuelve (fs, x) con x en float64.  No normaliza: la normalización se hace
    sobre el fragmento que se va a simular para no alterar la escala relativa
    del audio completo.
    """
    if archivo is None:
        fs, x = generar_wav_demo()
    else:
        fs, x = leer_wav(archivo)
    x, fs = remuestrear_audio(x, fs, fs_objetivo)
    return fs, x


def generar_wav_demo(fs=8000, dur=0.5):
    """Audio sintético (dos tonos con envolvente) para probar sin archivo."""
    t = np.arange(int(fs * dur)) / fs
    x = 0.6 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    x *= np.sin(np.pi * t / dur) ** 0.5
    return fs, x


def audio_a_bits(x, bits_por_muestra=8):
    """Cuantiza x in [-1,1] a `bits_por_muestra` bits (binario natural, MSB primero)."""
    niveles = 2 ** bits_por_muestra - 1
    q = np.round((np.clip(x, -1, 1) + 1) / 2 * niveles).astype(np.uint32)
    shifts = np.arange(bits_por_muestra - 1, -1, -1, dtype=np.uint32)
    bits = (q[:, None] >> shifts[None, :]) & 1
    return bits.astype(np.uint8).ravel()


def bits_a_audio(bits, bits_por_muestra=8):
    """Operación inversa de audio_a_bits: bits -> muestras en [-1, 1]."""
    bps = bits_por_muestra
    n = len(bits) // bps
    b = np.asarray(bits[: n * bps], dtype=np.uint32).reshape(n, bps)
    pesos = (1 << np.arange(bps - 1, -1, -1)).astype(np.uint32)
    q = (b * pesos).sum(axis=1)
    return q / (2 ** bps - 1) * 2 - 1


def cargar_fuente(archivo=None, max_muestras=500, inicio=0, bits_por_muestra=8,
                  normalizar_pico=True, fs_objetivo=None):
    """Carga el audio (o el demo si archivo es None) y lo convierte a bits.

    Se toman solo `max_muestras` muestras a partir de `inicio`, porque simular
    todo un audio (decenas de millones de bits) no es práctico.  Si se indica
    `fs_objetivo`, el audio se remuestrea antes de recortar (así `inicio` y
    `max_muestras` se refieren a la frecuencia de destino).

    Devuelve (bits, fs, x_usado) donde x_usado son las muestras de audio
    (normalizadas al pico) antes de cuantizar.
    """
    fs, x = cargar_audio(archivo, fs_objetivo)
    x = x[inicio: inicio + max_muestras]
    if normalizar_pico and np.max(np.abs(x)) > 0:
        x = x / np.max(np.abs(x))
    return audio_a_bits(x, bits_por_muestra), fs, x


# ----------------------------------------------------------------------------
# 2. Señal PCM y respuestas al impulso
# ----------------------------------------------------------------------------
def senal_pcm(bits, sps):
    """s(t) = sum_k b_k g(t - k Tb): cada bit es un pulso rectangular de 1 Tb."""
    return np.repeat(np.asarray(bits, dtype=np.float64), sps)


def h_exponencial(T, dt, span=10.0, normalizar=True):
    """h(t) = (1/T) exp(-t/T) u(t), truncada a [0, span*T).

    Con normalizar=True se escala para que sum(h)*dt = 1 (ganancia DC unitaria).
    Sin esto, la suma de Riemann sobreestima el área en ~dt/(2T), lo cual es
    muy notorio para T pequeño (n = 16 -> ~8 %).
    """
    m = max(int(round(span * T / dt)), 2)
    t = np.arange(m) * dt
    h = np.exp(-t / T) / T
    if normalizar:
        h = h / (h.sum() * dt)
    return t, h


def h_sinc(T, dt, span=20.0, semi_min=3.0, normalizar=True):
    """h(t) = (1/T) sinc(t/T), truncada simétricamente a |t| <= max(span*T, semi_min*Tb).

    La sinc es no causal e infinita: hay que truncarla.  Se garantiza una
    semi-duración mínima de `semi_min` Tb para que el truncamiento no distorsione
    el filtro cuando T es muy pequeño.  El índice central corresponde a t = 0.
    """
    semi = max(span * T, semi_min * TB)
    m = int(round(semi / dt))
    t = np.arange(-m, m + 1) * dt
    h = np.sinc(t / T) / T                  # np.sinc(x) = sin(pi x)/(pi x)
    if normalizar:
        h = h / (h.sum() * dt)
    return t, h


def salida_canal(s, h, dt, causal):
    """y = s * h aproximada, devuelta alineada con el eje de tiempo de s.

    Si el canal es no causal (h simétrica centrada), se descarta el retardo de
    (len(h)-1)/2 muestras que introduce la implementación discreta, de modo que
    y[i] corresponde al instante t = i*dt, igual que s[i].
    """
    completa = fftconvolve(s, h) * dt       # factor dt: convolución continua
    retardo = 0 if causal else (len(h) - 1) // 2
    return completa[retardo: retardo + len(s)]


def H_teorica(tipo, f, T):
    """|H(f)| teórico de cada canal (ganancia DC unitaria)."""
    if tipo == "exponencial":
        return 1.0 / np.sqrt(1.0 + (2 * np.pi * f * T) ** 2)
    return (np.abs(f) <= 1.0 / (2.0 * T)).astype(float)


def respuesta_frecuencia(h, dt, nfft=2 ** 15):
    """|H(f)| numérica (DFT de h escalada por dt). f en múltiplos de 1/Tb."""
    H = np.fft.rfft(h, nfft) * dt
    f = np.fft.rfftfreq(nfft, dt)
    return f, np.abs(H)


# ----------------------------------------------------------------------------
# 3. Instante de muestreo, apertura del ojo, detección y BER
# ----------------------------------------------------------------------------
def apertura_ojo(y, bits, sps, offset, descartar=5):
    """Apertura vertical del ojo en el instante k*sps + offset.

    apertura = min(y | bit=1) - max(y | bit=0).   > 0: ojo abierto, <= 0: cerrado.
    Se descartan `descartar` bits en cada extremo (transitorios del filtro).
    """
    nb = len(bits)
    k = np.arange(descartar, nb - descartar)
    if len(k) == 0:
        return np.nan
    m = y[k * sps + offset]
    b = np.asarray(bits)[k]
    unos, ceros = m[b == 1], m[b == 0]
    if len(unos) == 0 or len(ceros) == 0:
        return np.nan
    return float(unos.min() - ceros.max())


def mejor_offset(y, bits, sps, descartar=5):
    """Instante de muestreo (dentro del bit) que maximiza la apertura del ojo."""
    aps = np.array([apertura_ojo(y, bits, sps, o, descartar) for o in range(sps)])
    if np.all(np.isnan(aps)):
        return sps // 2
    # Con canales de banda ancha el ojo está igual de abierto en una meseta de
    # instantes; se toma el centro de esa meseta (en vez del primer máximo).
    candidatos = np.flatnonzero(aps >= np.nanmax(aps) - 0.005)
    return int(candidatos[len(candidatos) // 2])


def detectar(y, sps, offset, nbits, umbral=0.5):
    """Decisión por umbral en el instante de muestreo de cada bit."""
    return (y[np.arange(nbits) * sps + offset] > umbral).astype(np.uint8)


def segmentos_ojo(y, sps, offset, n_trazas=300, descartar=5):
    """Trazas de 2 Tb centradas en el instante de muestreo (para el diagrama de ojo).

    Devuelve (t, segs): t en unidades de Tb relativas al instante de muestreo
    (t = 0), y segs de forma (n_trazas, 2*sps).
    """
    nb = len(y) // sps
    k = np.arange(max(descartar, 1), nb - max(descartar, 1) - 1)
    inicios = k * sps + offset - sps
    inicios = inicios[(inicios >= 0) & (inicios + 2 * sps <= len(y))]
    if len(inicios) > n_trazas:             # muestreo uniforme para cubrir todo el audio
        inicios = inicios[np.linspace(0, len(inicios) - 1, n_trazas).astype(int)]
    segs = np.stack([y[i: i + 2 * sps] for i in inicios])
    t = (np.arange(2 * sps) - sps) * (TB / sps)
    return t, segs


# ----------------------------------------------------------------------------
# 4. Simulación completa
# ----------------------------------------------------------------------------
@dataclass
class Resultado:
    tipo: str
    n: int
    T: float
    sps: int
    dt: float
    sigma: float
    descartar: int
    bits_tx: np.ndarray
    bits_rx: np.ndarray
    s: np.ndarray            # señal PCM transmitida
    y_limpia: np.ndarray     # salida del canal sin ruido
    y: np.ndarray            # salida del canal (con ruido si sigma > 0)
    t_h: np.ndarray
    h: np.ndarray
    offset: int              # instante de muestreo dentro del bit [muestras]
    apertura: float
    errores: int
    ber: float

    @property
    def W(self):
        return 1.0 / self.T


def simular(bits, tipo, n, sps=100, sigma=0.0, semilla=0, descartar=5, normalizar=True):
    """Transmite `bits` por el canal `tipo` con T = Tb/n y devuelve un Resultado.

    sigma: desviación estándar de ruido gaussiano aditivo a la salida del canal
           (en unidades de la amplitud del bit). sigma = 0 -> sin ruido.
    El instante de muestreo se elige sobre la señal sin ruido.
    """
    if tipo not in TIPOS_CANAL:
        raise ValueError(f"tipo de canal desconocido: {tipo}")
    bits = np.asarray(bits, dtype=np.uint8)
    T, dt = TB / n, TB / sps

    s = senal_pcm(bits, sps)
    if tipo == "exponencial":
        t_h, h = h_exponencial(T, dt, normalizar=normalizar)
    else:
        t_h, h = h_sinc(T, dt, normalizar=normalizar)

    y_limpia = salida_canal(s, h, dt, causal=(tipo == "exponencial"))
    offset = mejor_offset(y_limpia, bits, sps, descartar)
    apertura = apertura_ojo(y_limpia, bits, sps, offset, descartar)

    if sigma > 0:
        ruido = np.random.default_rng(semilla).standard_normal(len(y_limpia))
        y = y_limpia + sigma * ruido
    else:
        y = y_limpia

    bits_rx = detectar(y, sps, offset, len(bits))
    errores = int(np.count_nonzero(bits_rx != bits))
    return Resultado(tipo, n, T, sps, dt, sigma, descartar, bits, bits_rx, s,
                     y_limpia, y, t_h, h, offset, apertura, errores,
                     errores / len(bits))


def simular_todo(bits, sps=100, sigma=0.0, semilla=0, valores_n=VALORES_N,
                 descartar=5, normalizar=True):
    """Simula los dos canales para todos los n. Devuelve {(tipo, n): Resultado}."""
    return {
        (tipo, n): simular(bits, tipo, n, sps, sigma, semilla, descartar, normalizar)
        for tipo in TIPOS_CANAL for n in valores_n
    }
