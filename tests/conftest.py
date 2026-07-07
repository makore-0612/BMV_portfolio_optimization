import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def precios_con_huecos():
    fechas = pd.date_range("2024-01-01", periods=20, freq="D")
    valores = np.array([
        100, 101, 102, 103, 104,
        np.nan, np.nan,
        107, 108, 109, 110, 111,
        np.nan, np.nan, np.nan, np.nan, np.nan,
        117, 118, 119,
    ], dtype=float)
    return pd.Series(valores, index=fechas)


@pytest.fixture
def precios_incompletos():
    fechas = pd.date_range("2024-01-01", periods=10, freq="D")
    a = [100.0] * 10
    b = [50.0] * 10
    c = [25.0] * 10

    b[3] = np.nan
    c[3] = np.nan
    c[7] = np.nan

    return pd.DataFrame({"A": a, "B": b, "C": c}, index=fechas)


@pytest.fixture
def rendimientos_dispersos():
    rng = np.random.default_rng(42)
    fechas = pd.date_range("2024-01-01", periods=500, freq="B")
    medias_diarias = np.array([0.0000, 0.0005, 0.0010, 0.0020])
    ruido = rng.normal(scale=0.01, size=(500, 4))
    valores = medias_diarias + ruido

    return pd.DataFrame(valores, index=fechas, columns=["A", "B", "C", "D"])


@pytest.fixture
def rendimientos_medias_iguales():
    rng = np.random.default_rng(7)
    T, N = 100, 4
    m = 0.001

    R = rng.normal(scale=0.01, size=(T, N))
    R = R - R.mean(axis=0, keepdims=True) + m

    fechas = pd.date_range("2024-01-01", periods=T, freq="B")
    return pd.DataFrame(R, index=fechas, columns=["A", "B", "C", "D"])


@pytest.fixture
def mercado_y_activos_capm():
    rng = np.random.default_rng(3)
    T = 500
    fechas = pd.date_range("2024-01-01", periods=T, freq="B")
    mercado = pd.Series(rng.normal(loc=0.0003, scale=0.01, size=T), index=fechas, name="mercado")

    betas_verdaderos = np.array([0.5, 1.0, 1.5, 2.0])
    activos = pd.DataFrame(
        np.outer(mercado.values, betas_verdaderos),
        index=fechas, columns=["A", "B", "C", "D"],
    )
    return activos, mercado, betas_verdaderos


@pytest.fixture
def rendimientos_cuasi_singulares():
    rng = np.random.default_rng(11)
    T, N = 20, 18
    fechas = pd.date_range("2024-01-01", periods=T, freq="B")
    columnas = [f"activo_{i}" for i in range(N)]
    valores = rng.normal(scale=0.01, size=(T, N))

    return pd.DataFrame(valores, index=fechas, columns=columnas)


@pytest.fixture
def mu_sigma_sinteticos():
    rng = np.random.default_rng(21)
    activos = ["A", "B", "C", "D", "E"]
    n = len(activos)

    mu = pd.Series(rng.uniform(0.03, 0.15, size=n), index=activos)

    L = rng.normal(scale=0.05, size=(n, n))
    Sigma_arr = L @ L.T + 0.01 * np.eye(n)
    Sigma = pd.DataFrame(Sigma_arr, index=activos, columns=activos)

    return mu, Sigma
