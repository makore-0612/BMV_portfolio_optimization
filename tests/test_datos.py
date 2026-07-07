import numpy as np
import pandas as pd
import pytest

from src import datos


def test_encontrar_nans_bloques(precios_con_huecos):
    bloques = datos.encontrar_nans_bloques(precios_con_huecos)
    assert bloques == [(5, 6, 2), (12, 16, 5)]


def test_imputar_corto(precios_con_huecos):
    resultado = datos.imputar_corto(precios_con_huecos, 5, 6, ventana=5)

    assert resultado.iloc[5] == pytest.approx(102.0)
    assert resultado.iloc[6] == pytest.approx(102.4)
    assert resultado.iloc[:5].equals(precios_con_huecos.iloc[:5])
    assert resultado.iloc[12:17].isna().all()


def test_imputar_largo(precios_con_huecos):
    resultado = datos.imputar_largo(precios_con_huecos, 12, 16)

    esperado = [112.0, 113.0, 114.0, 115.0, 116.0]
    assert resultado.iloc[12:17].tolist() == pytest.approx(esperado)
    assert resultado.iloc[5:7].isna().all()


def test_imputar_serie_despacha_por_longitud_de_racha(precios_con_huecos):
    resultado = datos.imputar_serie(precios_con_huecos, umbral_racha_corta=3, ventana=5)

    assert not resultado.isna().any()
    assert resultado.iloc[5] == pytest.approx(102.0)
    assert resultado.iloc[6] == pytest.approx(102.4)
    assert resultado.iloc[12:17].tolist() == pytest.approx([112.0, 113.0, 114.0, 115.0, 116.0])


def test_eliminar_columnas_incompletas(precios_incompletos):
    resultado = datos.eliminar_columnas_incompletas(precios_incompletos, umbral=0.10)

    assert list(resultado.columns) == ["A", "B"]


def test_perfilar_faltantes(precios_incompletos):
    reporte = datos.perfilar_faltantes(precios_incompletos)

    assert reporte["filas_totales"] == 10
    assert reporte["filas_completas"] == 8
    assert reporte["porcentaje_filas_completas"] == pytest.approx(0.8)
    assert reporte["faltantes_por_columna"]["A"] == pytest.approx(0.0)
    assert reporte["faltantes_por_columna"]["B"] == pytest.approx(0.1)
    assert reporte["faltantes_por_columna"]["C"] == pytest.approx(0.2)


def test_calcular_rendimientos_log():
    precios = pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [50.0, 45.0, 40.5]})

    rendimientos = datos.calcular_rendimientos_log(precios)

    assert len(rendimientos) == len(precios) - 1
    assert rendimientos["A"].iloc[0] == pytest.approx(np.log(1.1))
    assert rendimientos["A"].iloc[1] == pytest.approx(np.log(1.1))
    assert rendimientos["B"].iloc[0] == pytest.approx(np.log(0.9))
