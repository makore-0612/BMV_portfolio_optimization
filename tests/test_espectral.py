import numpy as np
import pytest

from src import espectral, optimizacion


def test_descomposicion_espectral_orden_descendente_y_reconstruye_sigma(mu_sigma_sinteticos):
    mu, Sigma = mu_sigma_sinteticos

    eigenvalues, eigenvectors = espectral.descomposicion_espectral(Sigma)

    assert (np.diff(eigenvalues) <= 0).all()

    reconstruida = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    assert reconstruida == pytest.approx(Sigma.values, abs=1e-8)


def test_num_componentes_umbral_dentro_de_rango(mu_sigma_sinteticos):
    mu, Sigma = mu_sigma_sinteticos
    eigenvalues, _ = espectral.descomposicion_espectral(Sigma)
    n = len(mu)

    for umbral in [0.5, 0.9, 0.99]:
        k = espectral.num_componentes_umbral(eigenvalues, umbral=umbral)
        assert 1 <= k <= n


def test_modelo_6_respeta_cota_superior(mu_sigma_sinteticos):
    mu, Sigma = mu_sigma_sinteticos
    mu_P = float(mu.mean())
    n = len(mu)

    pesos_exacto = optimizacion.pesos_frontera(mu, Sigma, mu_P)
    V_optimo = float(pesos_exacto.values @ Sigma.values @ pesos_exacto.values)

    for I in range(1, n + 1):
        cotas = espectral.cotas_modelo_6(mu, Sigma, mu_P, I)
        assert cotas["cota_superior"] >= V_optimo - 1e-6


def test_convergencia_modelo_6_recupera_optimo_exacto_con_todos_los_componentes(mu_sigma_sinteticos):
    mu, Sigma = mu_sigma_sinteticos
    mu_P = float(mu.mean())
    n = len(mu)

    resultado = espectral.convergencia_modelo_6(mu, Sigma, mu_P, rango_I=[n])

    assert abs(resultado["error_relativo_pct"][0]) < 1.0
