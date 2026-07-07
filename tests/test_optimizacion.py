import numpy as np
import pytest

from src import optimizacion


def test_pesos_gmv_suman_uno(mu_sigma_sinteticos):
    mu, Sigma = mu_sigma_sinteticos
    pesos = optimizacion.pesos_gmv(Sigma)

    assert pesos.sum() == pytest.approx(1.0)


def test_pesos_gmv_cerrado_coincide_con_numerico(mu_sigma_sinteticos):
    mu, Sigma = mu_sigma_sinteticos
    pesos_cerrado = optimizacion.pesos_gmv(Sigma)
    pesos_numerico = optimizacion.pesos_gmv_restringido(Sigma, permitir_cortos=True)

    var_cerrado = float(pesos_cerrado.values @ Sigma.values @ pesos_cerrado.values)
    var_numerico = float(pesos_numerico.values @ Sigma.values @ pesos_numerico.values)

    # SLSQP con tolerancia default no siempre converge a pesos idénticos al cierre
    # analítico (ver Etapa 3); la varianza lograda sí debe coincidir de cerca.
    assert var_numerico == pytest.approx(var_cerrado, abs=1e-4)


def test_pesos_frontera_logra_retorno_objetivo(mu_sigma_sinteticos):
    mu, Sigma = mu_sigma_sinteticos
    r_objetivo = float(mu.mean())

    pesos = optimizacion.pesos_frontera(mu, Sigma, r_objetivo)

    assert pesos.sum() == pytest.approx(1.0)
    assert float(mu.values @ pesos.values) == pytest.approx(r_objetivo, abs=1e-8)


def test_gmv_restringido_tiene_varianza_mayor_o_igual(mu_sigma_sinteticos):
    mu, Sigma = mu_sigma_sinteticos

    pesos_restringido = optimizacion.pesos_gmv_restringido(Sigma, permitir_cortos=False)
    var_restringido = float(pesos_restringido.values @ Sigma.values @ pesos_restringido.values)

    pesos_libre = optimizacion.pesos_gmv(Sigma)
    var_libre = float(pesos_libre.values @ Sigma.values @ pesos_libre.values)

    assert var_restringido >= var_libre - 1e-10


def test_tangencia_maximiza_sharpe_sobre_la_frontera(mu_sigma_sinteticos):
    mu, Sigma = mu_sigma_sinteticos
    r_f = 0.01

    pesos_tan = optimizacion.pesos_tangencia(mu, Sigma, r_f)
    resumen_tan = optimizacion.resumen_portafolio(pesos_tan, mu, Sigma, r_f=r_f)

    for r_objetivo in np.linspace(mu.min(), mu.max(), 15):
        pesos = optimizacion.pesos_frontera(mu, Sigma, r_objetivo)
        resumen = optimizacion.resumen_portafolio(pesos, mu, Sigma, r_f=r_f)
        assert resumen["sharpe"] <= resumen_tan["sharpe"] + 1e-8


def test_resumen_portafolio_pesos_pct_suman_100(mu_sigma_sinteticos):
    mu, Sigma = mu_sigma_sinteticos
    pesos = optimizacion.pesos_gmv(Sigma)

    resumen = optimizacion.resumen_portafolio(pesos, mu, Sigma)

    assert resumen["pesos_pct"].sum() == pytest.approx(100.0)


def test_optimizar_portafolio_dispatch_tipos_validos(mu_sigma_sinteticos):
    mu, Sigma = mu_sigma_sinteticos

    casos = [
        ("gmv", {}),
        ("tangencia", {"r_f": 0.01}),
        ("objetivo", {"r_objetivo": float(mu.mean())}),
    ]

    for tipo, kwargs in casos:
        resumen = optimizacion.optimizar_portafolio(mu, Sigma, tipo=tipo, **kwargs)
        assert set(resumen.keys()) >= {"pesos_pct", "mu", "sigma"}


def test_optimizar_portafolio_tipo_desconocido_lanza_error(mu_sigma_sinteticos):
    mu, Sigma = mu_sigma_sinteticos

    with pytest.raises(ValueError):
        optimizacion.optimizar_portafolio(mu, Sigma, tipo="no_existe")


def test_optimizar_portafolio_objetivo_requiere_r_objetivo(mu_sigma_sinteticos):
    mu, Sigma = mu_sigma_sinteticos

    with pytest.raises(ValueError):
        optimizacion.optimizar_portafolio(mu, Sigma, tipo="objetivo")
