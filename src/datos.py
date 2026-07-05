import time

import numpy as np
import pandas as pd
import yfinance as yf

from .config import RAIZ_PROYECTO

RUTA_PRECIOS = RAIZ_PROYECTO / "data" / "precios_crudos.parquet"
RUTA_BENCHMARK = RAIZ_PROYECTO / "data" / "benchmark.parquet"


def _serie_cierre(df):
    cierre = df["Close"]
    if isinstance(cierre, pd.DataFrame):
        cierre = cierre.iloc[:, 0]
    return cierre


def descargar_precios(tickers, inicio, fin=None, pausa=0.5):
    datos_por_ticker = {}

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=inicio, end=fin, auto_adjust=True, progress=False)
            if not df.empty:
                datos_por_ticker[ticker] = _serie_cierre(df)
            time.sleep(pausa)
        except Exception:
            continue

    return pd.concat(datos_por_ticker, axis=1)


def obtener_precios(config, forzar_descarga=False):
    if RUTA_PRECIOS.exists() and not forzar_descarga:
        return pd.read_parquet(RUTA_PRECIOS)

    tickers = config["tickers"]
    inicio = config["descarga"]["fecha_inicio_default"]
    pausa = config["descarga"]["pausa_segundos"]

    precios = descargar_precios(tickers, inicio=inicio, pausa=pausa)

    RUTA_PRECIOS.parent.mkdir(parents=True, exist_ok=True)
    precios.to_parquet(RUTA_PRECIOS)

    return precios


def obtener_benchmark(config, forzar_descarga=False):
    if RUTA_BENCHMARK.exists() and not forzar_descarga:
        return pd.read_parquet(RUTA_BENCHMARK).iloc[:, 0]

    ticker = config["benchmark"]
    inicio = config["descarga"]["fecha_inicio_default"]

    df_benchmark = yf.download(ticker, start=inicio, auto_adjust=True, progress=False)
    precios_benchmark = _serie_cierre(df_benchmark)
    precios_benchmark.name = ticker

    RUTA_BENCHMARK.parent.mkdir(parents=True, exist_ok=True)
    precios_benchmark.to_frame().to_parquet(RUTA_BENCHMARK)

    return precios_benchmark


def perfilar_faltantes(precios):
    faltantes_columna = precios.isna().mean()
    faltantes_fila = precios.isna().mean(axis=1)
    filas_totales = len(precios)
    filas_completas = int(precios.notna().all(axis=1).sum())

    return {
        "faltantes_por_columna": faltantes_columna,
        "faltantes_por_fila": faltantes_fila,
        "filas_totales": filas_totales,
        "filas_completas": filas_completas,
        "porcentaje_filas_completas": filas_completas / filas_totales if filas_totales else 0.0,
    }


def eliminar_columnas_incompletas(precios, umbral=0.10):
    faltantes_columna = precios.isna().mean()
    columnas_a_eliminar = faltantes_columna[faltantes_columna > umbral].index.tolist()
    return precios.drop(columns=columnas_a_eliminar)


def encontrar_nans_bloques(series):
    es_nan = series.isna().values
    bloques = []
    n = len(series)
    i = 0

    while i < n:
        if es_nan[i]:
            inicio = i
            while i < n and es_nan[i]:
                i += 1
            fin = i - 1
            bloques.append((inicio, fin, fin - inicio + 1))
        else:
            i += 1

    return bloques


def imputar_corto(series, inicio, fin, ventana=5):
    s = series.copy()

    for idx in range(inicio, fin + 1):
        previos = s.iloc[max(0, idx - ventana):idx].dropna()
        if len(previos) > 0:
            s.iloc[idx] = previos.mean()

    return s


def imputar_largo(series, inicio, fin):
    s = series.copy()
    idx_previo = inicio - 1
    idx_siguiente = fin + 1

    if idx_previo >= 0 and idx_siguiente < len(s):
        valor_previo = s.iloc[idx_previo]
        valor_siguiente = s.iloc[idx_siguiente]

        if pd.notna(valor_previo) and pd.notna(valor_siguiente):
            tamanio_ventana = fin - inicio + 1
            for k, idx in enumerate(range(inicio, fin + 1), start=1):
                s.iloc[idx] = valor_previo + (valor_siguiente - valor_previo) * (k / (tamanio_ventana + 1))

    return s


def imputar_serie(series, umbral_racha_corta=3, ventana=5):
    s = series.copy()
    bloques = encontrar_nans_bloques(s)

    for inicio, fin, longitud in bloques:
        if longitud <= umbral_racha_corta:
            s = imputar_corto(s, inicio, fin, ventana=ventana)
        else:
            s = imputar_largo(s, inicio, fin)

    return s


def imputar_precios(precios, umbral_racha_corta=3, ventana=5):
    precios_imputados = precios.copy()

    for columna in precios_imputados.columns:
        precios_imputados[columna] = imputar_serie(
            precios_imputados[columna], umbral_racha_corta=umbral_racha_corta, ventana=ventana)

    return precios_imputados


def calcular_rendimientos_log(precios):
    return np.log(precios / precios.shift(1)).dropna()
