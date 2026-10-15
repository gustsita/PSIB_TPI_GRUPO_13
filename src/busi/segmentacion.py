"""Parte 5 – Segmentación de la lesión.

Se segmenta sobre los mapas elegidos en la Parte 4. Enfoque semiautomático: el
usuario marca una semilla dentro de la lesión.

Métodos (vistos en clase):
- region growing desde la semilla,
- k-means sobre el vector de mapas (intensidad + textura),
- contornos activos (Chan-Vese, skimage.segmentation.chan_vese / morphological_chan_vese).
Refinamiento: apertura, cierre, relleno de huecos, y quedarse con la componente
conexa que contiene la semilla.
Validación: Dice e IoU contra la máscara manual.
"""

import numpy as np
from scipy import ndimage
from skimage import measure, morphology


def region_growing(mapa, semilla, tolerancia):
    """Crece desde `semilla` (fila, col) agregando vecinos con |valor − media_región| <= tolerancia.

    TODO: implementar con una cola (BFS) y vecindad de 4 u 8.
    """
    raise NotImplementedError


def kmeans_mapas(mapas, k=2, seed=13):
    """Clusteriza píxeles usando como features los mapas apilados (normalizados).

    mapas: lista de arrays 2D del mismo tamaño. Devuelve imagen de etiquetas.
    TODO: sklearn.cluster.KMeans sobre np.stack(mapas, -1).reshape(-1, len(mapas)).
    """
    raise NotImplementedError


def refinar(mascara, semilla, radio=3):
    """Apertura + cierre + relleno de huecos, y deja solo la componente que contiene la semilla."""
    se = morphology.disk(radio)
    m = morphology.binary_closing(morphology.binary_opening(mascara, se), se)
    m = ndimage.binary_fill_holes(m)
    etiquetas = measure.label(m)
    etiqueta = etiquetas[semilla]
    return etiquetas == etiqueta if etiqueta else np.zeros_like(m)


def semilla_desde_mascara(mascara):
    """Para evaluar sin clics: punto interior de la máscara manual (el más alejado del borde)."""
    dist = ndimage.distance_transform_edt(mascara)
    return np.unravel_index(np.argmax(dist), dist.shape)


def dice(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    return 2 * np.logical_and(pred, gt).sum() / (pred.sum() + gt.sum() + 1e-12)


def iou(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    return np.logical_and(pred, gt).sum() / (np.logical_or(pred, gt).sum() + 1e-12)
