"""Parte 1 – Reconstrucción de las marcas del ecógrafo (calipers, textos, escalas).

No reduce el speckle: saca estructuras que no son tejido para que no contaminen
la caracterización del ruido, los mapas de textura ni la segmentación.

Pasos:
1. Detección: las marcas son pequeñas y mucho más brillantes que su entorno.
   Top-hat blanco (img − apertura(img)) + umbral alto + etiquetado, filtrando las
   componentes por tamaño y forma. Dilatar un poco la máscara para cubrir el halo.
2. Reconstrucción: inpainting de los píxeles detectados a partir de la vecindad
   (cv2.inpaint con INPAINT_TELEA o INPAINT_NS, o skimage.restoration.inpaint_biharmonic).
3. Validación: pegar marcas sintéticas sobre regiones limpias, reconstruir y comparar
   la región con la original (MSE, PSNR).
"""

import cv2
import numpy as np
from skimage import measure, morphology, restoration


def tophat(img, radio=3):
    """Top-hat blanco: img − apertura(img) con un disco de radio `radio`.

    La apertura borra las estructuras claras más finas que el disco; al restarla
    quedan solo esas estructuras. Las marcas miden 3 px de grosor, así que un disco
    de radio 3 (diámetro 7) las borra por completo, incluso donde son más anchas
    (centro de las cruces, números).
    """
    return morphology.white_tophat(img, morphology.disk(radio))


def es_fibra(region, elongacion_min=3, largo_min=8, angulo_max=35):
    """True si la componente parece una fibra del tejido y no una marca.

    Las fibras son alargadas y aproximadamente horizontales (el tejido mamario se
    dispone en capas paralelas a la piel). Las marcas son cruces, puntos y letras:
    compactas, o alargadas pero verticales (el "1", la "I").

    region: un elemento de skimage.measure.regionprops.
    angulo_max: máxima inclinación respecto de la horizontal, en grados.
    """
    largo = region.axis_major_length
    elongacion = largo / max(region.axis_minor_length, 1)
    # orientation de skimage es el ángulo entre el eje de las filas y el eje mayor:
    # vale ±90° para una componente horizontal y 0° para una vertical
    inclinacion = 90 - abs(np.degrees(region.orientation))
    return elongacion >= elongacion_min and largo >= largo_min and inclinacion <= angulo_max


def detectar_marcas(img, radio_tophat=3, umbral=100, dilatar=1, descartar_fibras=True):
    """Máscara booleana con las marcas del ecógrafo.

    1. Top-hat con disco de radio `radio_tophat` y umbral `umbral`: candidatos.
    2. Etiquetado de componentes conexas y descarte de las que parecen fibras.
    3. Dilatación de `dilatar` px para cubrir el borde suavizado de las marcas
       (en los perfiles de intensidad se ve 1 px de transición a cada lado).
    """
    candidatos = tophat(img, radio_tophat) >= umbral

    marcas = candidatos
    if descartar_fibras:
        etiquetas = measure.label(candidatos, connectivity=2)
        fibras = [r.label for r in measure.regionprops(etiquetas) if es_fibra(r)]
        marcas = candidatos & ~np.isin(etiquetas, fibras)

    if dilatar:
        marcas = morphology.dilation(marcas, morphology.disk(dilatar))
    return marcas


def reconstruir(img, mascara_marcas, radio=3, metodo="telea"):
    """Rellena los píxeles de `mascara_marcas` a partir de su vecindad (inpainting).

    metodo:
    - "telea": avanza desde el borde del hueco hacia adentro; cada píxel nuevo es un
      promedio ponderado de los píxeles conocidos a menos de `radio` px.
    - "ns" (Navier-Stokes): prolonga hacia adentro del hueco las líneas de igual
      nivel de gris (isofotas), para continuar los bordes que llegan al hueco.
    - "biharmonico": rellena con la superficie más suave que coincide con el borde
      del hueco (resuelve la ecuación biharmónica). `radio` no se usa.
    """
    if metodo == "biharmonico":
        rec = restoration.inpaint_biharmonic(img, mascara_marcas)   # devuelve float en [0, 1]
        return np.clip(rec * 255, 0, 255).round().astype(np.uint8)
    flag = cv2.INPAINT_TELEA if metodo == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(img.astype(np.uint8), mascara_marcas.astype(np.uint8), radio, flag)


def agregar_marcas_sinteticas(img, n=3, rng=None):
    """Copia de img con n cruces brillantes dibujadas, y la máscara de dónde se dibujaron.

    Sirve para validar: como se conoce la imagen original debajo de la marca, se puede
    medir qué tan bien la reconstruye el inpainting.
    TODO: dibujar cruces con cv2.drawMarker (MARKER_CROSS, color 255) en posiciones al azar.
    """
    raise NotImplementedError


def mse_psnr(original, reconstruida, mascara):
    """MSE y PSNR calculados solo dentro de `mascara`."""
    dif = original.astype(float)[mascara] - reconstruida.astype(float)[mascara]
    mse = np.mean(dif ** 2)
    psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else np.inf
    return mse, psnr
