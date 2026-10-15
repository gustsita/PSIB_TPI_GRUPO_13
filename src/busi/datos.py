"""Carga del dataset BUSI.

Estructura esperada (como viene de Kaggle):

    DATA_DIR/
        benign/     benign (1).png, benign (1)_mask.png, benign (1)_mask_1.png, ...
        malignant/  malignant (1).png, malignant (1)_mask.png, ...
        normal/     normal (1).png, normal (1)_mask.png, ...

Algunas imágenes tienen más de una máscara (_mask_1, _mask_2); se unen con OR.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io

from .config import load_config

CLASES = ("benign", "malignant", "normal")
_PATRON_MASCARA = re.compile(r"_mask(_\d+)?$")


def listar_casos(data_dir=None):
    """Devuelve un DataFrame con una fila por imagen: id, clase, ruta y rutas de máscaras."""
    data_dir = Path(data_dir) if data_dir else load_config()["DATA_DIR"]
    if not data_dir.exists():
        raise FileNotFoundError(
            f"No existe {data_dir}. Bajar BUSI y descomprimirlo ahí (ver README)."
        )

    filas = []
    for clase in CLASES:
        carpeta = data_dir / clase
        for ruta in sorted(carpeta.glob("*.png")):
            if _PATRON_MASCARA.search(ruta.stem):
                continue
            mascaras = sorted(carpeta.glob(f"{ruta.stem}_mask*.png"))
            filas.append({
                "id": ruta.stem,
                "clase": clase,
                "ruta": ruta,
                "mascaras": mascaras,
                "n_mascaras": len(mascaras),
            })

    return pd.DataFrame(filas)


def cargar_imagen(ruta):
    """Imagen en escala de grises, uint8 [0, 255].

    Algunas PNG de BUSI vienen en RGB con los tres canales iguales; nos quedamos con uno.
    """
    img = io.imread(ruta)
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.uint8)


def cargar_mascara(rutas_mascaras, shape=None):
    """Une todas las máscaras de un caso en una sola máscara booleana."""
    mascara = None
    for ruta in rutas_mascaras:
        m = io.imread(ruta)
        if m.ndim == 3:
            m = m[..., 0]
        m = m > 0
        mascara = m if mascara is None else (mascara | m)

    if mascara is None:
        if shape is None:
            raise ValueError("El caso no tiene máscaras y no se pasó shape.")
        mascara = np.zeros(shape, dtype=bool)
    return mascara


def cargar_caso(fila):
    """A partir de una fila de listar_casos(), devuelve (imagen, mascara)."""
    img = cargar_imagen(fila["ruta"])
    mascara = cargar_mascara(fila["mascaras"], shape=img.shape)
    return img, mascara
