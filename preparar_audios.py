"""
preparar_audios.py
==================
Prepara los audios que se incluyen en la aplicación (`audios/`) a partir de
una carpeta con archivos `.wav` sin comprimir.

Para cada archivo:
  1. Se mezcla a mono (promedio de canales, sin sumar para no saturar).
  2. Se remuestrea a `--fs` (16 kHz por defecto) con un filtro antialias.
  3. Se elige la ventana de `--duracion` segundos con mayor energía (RMS).
  4. Se aplican fundidos de entrada/salida para evitar clics.
  5. Se normaliza el pico a `--pico` (0.9 por defecto, deja margen).
  6. Se guarda como PCM de 16 bits, mono.

Uso:
    python preparar_audios.py --origen "C:\\ruta\\a\\mis\\audios"
    python preparar_audios.py --origen "C:\\ruta\\a\\mis\\audios" --fs 16000 --duracion 5

Los archivos resultantes son pequeños (unos 160 kB por cada 5 s a 16 kHz) y se
versionan en el repositorio para que la app los cargue al abrirse.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

FS_DEFECTO = 16000
DURACION_DEFECTO = 5.0
PICO_DEFECTO = 0.9
FUNDIDO_S = 0.01


def leer_mono(ruta):
    """Lee un .wav (8/16/24/32 bits o float) y lo devuelve en mono float64."""
    with warnings.catch_warnings():
        # Algunos .wav traen chunks de metadatos que scipy ignora.
        warnings.simplefilter("ignore", wavfile.WavFileWarning)
        fs, datos = wavfile.read(ruta)

    if datos.dtype == np.uint8:
        x = (datos.astype(np.float64) - 128.0) / 128.0
    elif datos.dtype == np.int16:
        x = datos.astype(np.float64) / 32768.0
    elif datos.dtype == np.int32:
        x = datos.astype(np.float64) / 2147483648.0
    else:
        x = datos.astype(np.float64)

    if x.ndim == 2:
        x = x.mean(axis=1)
    return int(fs), x


def remuestrear(x, fs_in, fs_out):
    """Remuestrea conservando la duración (factor racional exacto)."""
    if fs_out == fs_in:
        return x
    g = int(np.gcd(fs_in, fs_out))
    return resample_poly(x, fs_out // g, fs_in // g)


def ventana_mas_energetica(x, n, fs, paso_s=0.25):
    """Inicio de la ventana de n muestras con mayor RMS (paso grueso + fino)."""
    if len(x) <= n:
        return 0
    paso = max(1, int(paso_s * fs))
    inicios = np.arange(0, len(x) - n + 1, paso)
    energia = np.array([np.mean(x[i:i + n] ** 2) for i in inicios])
    k = int(inicios[int(np.argmax(energia))])

    # Refinamiento muestra a muestra alrededor del mejor bloque.
    ini = max(0, k - paso)
    fin = min(len(x) - n, k + paso)
    if fin > ini:
        e = np.array([np.mean(x[i:i + n] ** 2) for i in range(ini, fin + 1)])
        k = ini + int(np.argmax(e))
    return int(k)


def fundir(y, fs, dur_s=FUNDIDO_S):
    """Fundido de entrada y salida (coseno alzado) para evitar clics."""
    n = min(int(dur_s * fs), len(y) // 2)
    if n <= 0:
        return y
    rampa = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n))
    y = y.copy()
    y[:n] *= rampa
    y[-n:] *= rampa[::-1]
    return y


def normalizar(y, pico):
    m = float(np.max(np.abs(y)))
    return y if m == 0 else y * (pico / m)


def preparar_audio(ruta, fs_obj, dur_s, pico):
    fs, x = leer_mono(ruta)
    x = remuestrear(x, fs, fs_obj)
    n = int(round(dur_s * fs_obj))
    ini = ventana_mas_energetica(x, n, fs_obj)
    y = x[ini:ini + n]
    y = fundir(y, fs_obj)
    y = normalizar(y, pico)
    return fs_obj, y, ini / fs_obj


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origen", required=True,
                    help="carpeta con los .wav originales")
    ap.add_argument("--destino", default="audios",
                    help="carpeta de salida (por defecto ./audios)")
    ap.add_argument("--fs", type=int, default=FS_DEFECTO,
                    help=f"frecuencia de muestreo de salida (por defecto {FS_DEFECTO})")
    ap.add_argument("--duracion", type=float, default=DURACION_DEFECTO,
                    help=f"duración del fragmento en segundos (por defecto {DURACION_DEFECTO})")
    ap.add_argument("--pico", type=float, default=PICO_DEFECTO,
                    help=f"pico de normalización, 0 < pico <= 1 (por defecto {PICO_DEFECTO})")
    a = ap.parse_args()

    if not (0 < a.pico <= 1):
        ap.error("--pico debe estar en (0, 1]")

    origen = Path(a.origen).expanduser().resolve()
    destino = Path(a.destino).expanduser().resolve()
    if not origen.is_dir():
        ap.error(f"no existe la carpeta de origen: {origen}")

    archivos = sorted(origen.glob("*.wav"))
    if not archivos:
        ap.error(f"no se encontraron archivos .wav en {origen}")

    destino.mkdir(parents=True, exist_ok=True)
    info = {
        "fs": a.fs,
        "duracion_s": a.duracion,
        "pico": a.pico,
        "audios": [],
    }

    for ruta in archivos:
        fs, y, offset = preparar_audio(ruta, a.fs, a.duracion, a.pico)
        salida = destino / ruta.name
        wavfile.write(salida, fs, np.round(y * 32767).astype(np.int16))
        info["audios"].append({
            "archivo": salida.name,
            "origen": ruta.name,
            "inicio_s": round(offset, 3),
            "duracion_s": round(len(y) / fs, 3),
            "pico": round(float(np.max(np.abs(y))), 4),
        })
        print(f"{ruta.name:16s} -> {salida.name:16s} "
              f"({len(y) / fs:.2f} s @ {fs} Hz, desde {offset:.2f} s)")

    with open(destino / "info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"\n{len(archivos)} audios preparados en {destino}")


if __name__ == "__main__":
    main()
