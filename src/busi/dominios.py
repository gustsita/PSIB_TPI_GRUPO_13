"""Parte 3 – Filtrado en dominio espacial vs. frecuencial y efecto sobre la textura.

Pregunta guía: ¿qué le hace cada filtro a la textura de la imagen? No alcanza con
medir cuánto ruido sacó: hay que ver qué particularidades texturales conserva o
destruye cada dominio (bordes, granularidad del speckle, patrón interno de la lesión).

Pasos:
1. Filtros espaciales: media, gaussiano, mediana (varios tamaños de ventana).
2. Filtros frecuenciales: pasabajos ideal, Butterworth y gaussiano sobre la FFT
   (varias frecuencias de corte).
3. Para cada imagen filtrada medir:
   - SNR / ENL (número equivalente de looks) en un parche homogéneo,
   - preservación de bordes (por ejemplo, gradiente medio sobre el contorno de la máscara),
   - descriptores GLCM dentro y fuera de la lesión,
   - espectro de potencia (qué frecuencias se van).
4. Comparar dominios: tablas y figuras lado a lado.
"""

import numpy as np
from scipy import ndimage
from skimage import filters


# ---- Dominio espacial ------------------------------------------------------

def filtro_media(img, tam=5):
    return ndimage.uniform_filter(img.astype(float), size=tam)


def filtro_gaussiano(img, sigma=1.5):
    return filters.gaussian(img.astype(float), sigma=sigma, preserve_range=True)


def filtro_mediana(img, tam=5):
    return ndimage.median_filter(img, size=tam)


# ---- Dominio frecuencial ---------------------------------------------------

def espectro(img):
    """FFT centrada de la imagen (np.fft.fftshift(np.fft.fft2(img)))."""
    return np.fft.fftshift(np.fft.fft2(img.astype(float)))


def mascara_pasabajos(shape, corte, tipo="butterworth", orden=2):
    """Función de transferencia H(u, v) centrada.

    tipo: "ideal", "butterworth" o "gaussiano". `corte` en píxeles de frecuencia (D0).
    TODO: construir la distancia D(u, v) al centro y aplicar la fórmula de cada filtro.
    """
    raise NotImplementedError


def filtrar_frecuencia(img, H):
    """Multiplica el espectro por H y vuelve al dominio espacial (parte real)."""
    G = espectro(img) * H
    return np.real(np.fft.ifft2(np.fft.ifftshift(G)))


# ---- Métricas de comparación -----------------------------------------------

def enl(parche):
    """Número equivalente de looks: (μ/σ)². Más alto = menos speckle."""
    parche = parche.astype(float)
    return (parche.mean() / parche.std()) ** 2


def preservacion_bordes(img_filtrada, mascara):
    """Gradiente medio sobre el contorno de la lesión.

    TODO: contorno = mascara XOR erosión(mascara); gradiente con filters.sobel.
    """
    raise NotImplementedError
