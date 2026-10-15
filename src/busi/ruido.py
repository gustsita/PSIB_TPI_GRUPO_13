"""Parte 2 – Caracterización del ruido speckle.

Idea: el speckle es ruido multiplicativo, g = f · n. En zonas de tejido homogéneo
f es casi constante, así que la distribución de g refleja la de n.

Pasos:
1. Elegir parches homogéneos de tejido (lejos de la lesión y de las marcas del ecógrafo).
2. Estudiar el histograma de cada parche y ajustar distribuciones candidatas
   (Rayleigh, Gamma, Nakagami); compararlas (log-verosimilitud / test KS).
3. Calcular la relación media/desvío (para speckle completamente desarrollado con
   amplitud Rayleigh, μ/σ ≈ 1,91) y ver si el dataset la cumple.
4. Verificar que log(g) = log(f) + log(n): el ruido pasa a ser aditivo y su varianza
   ya no depende del brillo local.

Ojo para la discusión: las imágenes del ecógrafo vienen con compresión logarítmica y
post-procesamiento, así que no hay que esperar que cumplan exactamente la teoría.
"""

import numpy as np
from scipy import stats


def extraer_parches(img, mascara, tam=32, n=10, margen=20, rng=None):
    """Devuelve n parches tam×tam de tejido, alejados al menos `margen` px de la lesión.

    TODO: descartar parches con poca varianza (fondo negro fuera del sector de la
    sonda) o con marcas del ecógrafo (píxeles saturados).
    """
    raise NotImplementedError


def razon_media_desvio(parche):
    """μ/σ del parche. Comparar contra 1,91 (Rayleigh)."""
    parche = parche.astype(float)
    return parche.mean() / parche.std()


def ajustar_distribuciones(valores, candidatas=("rayleigh", "gamma", "nakagami")):
    """Ajusta cada distribución de scipy.stats a `valores`.

    Devuelve dict {nombre: {"params": ..., "loglik": ..., "ks": ...}}.
    TODO: implementar con getattr(stats, nombre).fit(valores) y stats.kstest.
    """
    raise NotImplementedError


def varianza_vs_media(img, ventana=7):
    """Media y varianza local en cada píxel.

    Graficar varianza contra media: si el ruido es multiplicativo la varianza crece
    con la media; después de aplicar log, debería quedar aproximadamente constante.
    TODO: usar scipy.ndimage.uniform_filter sobre img y sobre img**2.
    """
    raise NotImplementedError
