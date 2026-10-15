"""Parte 6 – Descriptores de la lesión y análisis benigno vs. maligno.

Descriptores (calculados sobre la máscara manual Y sobre la segmentación propia,
para separar errores de segmentación de la capacidad de los descriptores):
- Forma (BI-RADS): área, relación alto/ancho (orientación), circularidad,
  solidez (área / área del convex hull), irregularidad del contorno.
- Intensidad: media, desvío, ecogenicidad relativa (lesión vs. anillo de tejido),
  sombra acústica posterior (región debajo de la lesión vs. tejido lateral).
- Textura: GLCM dentro de la lesión (4 direcciones, varias distancias).

Estadística: Mann-Whitney U por descriptor, clasificador simple (regresión logística)
con validación cruzada estratificada y class_weight="balanced".
"""

import numpy as np
from skimage import measure


def descriptores_forma(mascara):
    """Dict con área, alto/ancho, circularidad, solidez, excentricidad."""
    props = measure.regionprops(mascara.astype(np.uint8))[0]
    alto = props.bbox[2] - props.bbox[0]
    ancho = props.bbox[3] - props.bbox[1]
    return {
        "area": props.area,
        "alto_ancho": alto / ancho,
        "circularidad": 4 * np.pi * props.area / props.perimeter ** 2,
        "solidez": props.solidity,
        "excentricidad": props.eccentricity,
    }


def descriptores_intensidad(img, mascara, margen=15):
    """Media, desvío, ecogenicidad relativa y sombra acústica posterior.

    TODO: anillo = dilatación(mascara, margen) − mascara;
          sombra = región de igual ancho debajo del bbox de la lesión.
    """
    raise NotImplementedError


def descriptores_textura(img, mascara, niveles=16, distancias=(1, 2, 4)):
    """GLCM restringida a la lesión.

    TODO: recortar el bbox, poner fuera de la máscara un nivel extra que se descarta
    de la matriz (o usar solo pares dentro de la máscara) y usar graycoprops.
    """
    raise NotImplementedError


def tabla_descriptores(casos, usar_mascara_manual=True):
    """Recorre los casos (DataFrame de datos.listar_casos) y arma un DataFrame de descriptores.

    TODO: implementar cuando estén las funciones anteriores.
    """
    raise NotImplementedError


def test_mann_whitney(tabla, columna_clase="clase"):
    """p-valor de Mann-Whitney U (benign vs malignant) para cada descriptor.

    TODO: scipy.stats.mannwhitneyu; corregir por comparaciones múltiples (Bonferroni/Holm).
    """
    raise NotImplementedError
