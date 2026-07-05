import json
from pathlib import Path

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_CONFIG_DEFAULT = RAIZ_PROYECTO / "config" / "config.json"


def cargar_config(ruta=RUTA_CONFIG_DEFAULT):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)
