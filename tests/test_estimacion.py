import numpy as np
import pandas as pd
import pytest

from src import estimacion


def test_estimar_mu_historico(rendimientos_dispersos):
    mu = estimacion.estimar_mu_historico(rendimientos_dispersos, factor_anual=252)
    esperado = rendimientos_dispersos.mean() * 252

    pd.testing.assert_series_equal(mu, esperado)


def test_estimar_mu_bayes_stein_reduce_dispersion(rendimientos_dispersos):
    mu_historico = estimacion.estimar_mu_historico(rendimientos_dispersos, factor_anual=252)
    mu_bayes_stein = estimacion.estimar_mu_bayes_stein(rendimientos_dispersos, factor_anual=252)

    assert mu_bayes_stein.std() < mu_historico.std()


def test_estimar_mu_bayes_stein_sin_shrinkage_si_medias_iguales(rendimientos_medias_iguales):
    mu_historico = estimacion.estimar_mu_historico(rendimientos_medias_iguales, factor_anual=252)
    mu_bayes_stein = estimacion.estimar_mu_bayes_stein(rendimientos_medias_iguales, factor_anual=252)

    pd.testing.assert_series_equal(mu_bayes_stein, mu_historico, atol=1e-8)


def test_regresion_capm_recupera_beta_conocido(mercado_y_activos_capm):
    activos, mercado, betas_verdaderos = mercado_y_activos_capm

    resultado = estimacion.regresion_capm(activos, mercado, tasa_libre_riesgo_anual=0.0)

    assert resultado["beta"].values == pytest.approx(betas_verdaderos, abs=1e-8)
    assert resultado["alpha_diario"].values == pytest.approx(np.zeros(4), abs=1e-10)


def test_estimar_sigma_ledoit_wolf_eleva_eigenvalor_minimo(rendimientos_cuasi_singulares):
    Sigma_muestral = estimacion.estimar_sigma_muestral(rendimientos_cuasi_singulares, factor_anual=1)
    Sigma_lw = estimacion.estimar_sigma_ledoit_wolf(rendimientos_cuasi_singulares, factor_anual=1)

    min_eig_muestral = np.linalg.eigvalsh(Sigma_muestral.values).min()
    min_eig_lw = np.linalg.eigvalsh(Sigma_lw.values).min()

    assert min_eig_lw > min_eig_muestral


def test_prueba_fdr_alphas_es_monotonica():
    rng = np.random.default_rng(5)
    activos = [f"activo_{i}" for i in range(20)]
    p_valores = pd.Series(rng.uniform(0, 1, size=20), index=activos)

    resultado_capm = {
        "alpha_diario": pd.Series(rng.normal(size=20), index=activos),
        "p_valor_alpha": p_valores,
    }

    tabla = estimacion.prueba_fdr_alphas(resultado_capm)

    assert (tabla["p_valor_ajustado_fdr"] >= tabla["p_valor"] - 1e-12).all()
    assert tabla["p_valor_ajustado_fdr"].between(0.0, 1.0).all()
