"""Parte 4 – Mapas de características (antes de definir la segmentación).

Idea: en vez de elegir un método de segmentación a ciegas, calcular mapas píxel a
píxel (con ventana deslizante) y medir con las máscaras manuales cuál separa mejor
la lesión del tejido. La segmentación (Parte 5) se hace sobre los mapas ganadores.

Mapas propuestos:
- intensidad filtrada (salida de la Parte 3),
- media y varianza local, coeficiente de variación local (σ/μ),
- GLCM local: contraste, homogeneidad, energía, correlación, entropía,
- magnitud del gradiente.
"""

import numpy as np
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops


def reducir_niveles(img, niveles=16):
    """Cuantiza a `niveles` grises para que la GLCM sea manejable (como en la clase)."""
    img = img.astype(float)
    img = (img - img.min()) / (img.max() - img.min() + 1e-12)
    return np.minimum((img * niveles).astype(np.uint8), niveles - 1)


def mapa_media_varianza(img, ventana=9):
    """Devuelve (media_local, varianza_local)."""
    img = img.astype(float)
    media = ndimage.uniform_filter(img, ventana)
    media2 = ndimage.uniform_filter(img ** 2, ventana)
    return media, np.maximum(media2 - media ** 2, 0)


def glcm_ventana(ventana_img, niveles=16, distancias=(1,), angulos=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    """Propiedades GLCM promediadas en las 4 direcciones para una ventana."""
    P = graycomatrix(ventana_img, distancias, angulos, levels=niveles, symmetric=True, normed=True)
    props = {p: graycoprops(P, p).mean() for p in ("contrast", "homogeneity", "energy", "correlation")}
    p = P.mean(axis=(2, 3))
    props["entropy"] = -np.sum(p[p > 0] * np.log2(p[p > 0]))
    return props


def mapas_glcm(img, ventana=15, paso=4, niveles=16):
    """Mapas GLCM locales.

    Calcular la GLCM en cada píxel es lento; se calcula cada `paso` píxeles y después
    se interpola al tamaño original.
    TODO: recorrer la imagen con glcm_ventana y armar un dict {propiedad: mapa}.
    """
    raise NotImplementedError


def separabilidad(mapa, mascara, margen=15):
    """Qué tan bien separa `mapa` la lesión del tejido que la rodea.

    Compara los valores dentro de la máscara contra los de un anillo de `margen` px
    alrededor (dilatación − máscara). Devuelve AUC y/o razón de Fisher:
        F = (μ_in − μ_out)² / (σ²_in + σ²_out)
    TODO: implementar; usar sklearn.metrics.roc_auc_score para el AUC.
    """
    raise NotImplementedError
