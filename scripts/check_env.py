"""Verifica dependencias y que el dataset esté donde dice config.yaml.

Uso (desde la raíz del repo):  python scripts/check_env.py
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PAQUETES = ["numpy", "scipy", "pandas", "cv2", "skimage", "sklearn", "matplotlib", "yaml"]

ok = True
for nombre in PAQUETES:
    try:
        mod = importlib.import_module(nombre)
        print(f"[OK]    {nombre} {getattr(mod, '__version__', '')}")
    except ImportError:
        print(f"[FALTA] {nombre}  ->  pip install -r requirements.txt")
        ok = False

if ok:
    from busi.datos import listar_casos

    try:
        casos = listar_casos()
    except FileNotFoundError as e:
        print(f"[FALTA] dataset: {e}")
        ok = False
    else:
        print(f"[OK]    dataset: {len(casos)} imágenes")
        print(casos["clase"].value_counts().to_string())
        print(f"        con más de una máscara: {(casos['n_mascaras'] > 1).sum()}")

sys.exit(0 if ok else 1)
