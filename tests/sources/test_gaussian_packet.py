"""Test Gaussian-packet profiles, fields, spectrum, and propagation."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.constants import c

from wakis.sources import GaussianPacket


def test_source_updates_component(unit_solver):
    source = GaussianPacket(
        zs=2,
        sigmaz=0.02,
        sigmaxy=0.03,
        tinj=0.0,
        amplitude=2.0,
    )

    source.update(unit_solver, 0.0)

    assert np.all(unit_solver.H[:, :, 2, "y"] < 0.0)
    assert not np.any(unit_solver.E.toarray())
    assert not np.any(unit_solver.H[:, :, 2, "x"])


def test_gaussian_packet_can_inject_electric_field(unit_solver):
    source = GaussianPacket(
        field="E",
        zs=2,
        sigmaz=0.02,
        sigmaxy=0.03,
        tinj=0.0,
        amplitude=2.0,
    )

    source.update(unit_solver, 0.0)

    assert np.all(unit_solver.E[:, :, 2, "x"] > 0.0)
    assert not np.any(unit_solver.H.toarray())
    assert not np.any(unit_solver.E[:, :, 2, "y"])


def test_gaussian_packet_envelope_peak_and_width(unit_solver):
    source = GaussianPacket(
        zs=2,
        sigmaz=0.01,
        sigmaxy=0.02,
        tinj=0.03,
        beta=0.5,
    )
    peak_time = source.tinj / (source.beta * c)
    sigma_time = source.sigmaz / (source.beta * c)
    center = unit_solver.Nx // 2

    source.update(unit_solver, peak_time)
    peak = -unit_solver.H[center, center, 2, "y"]
    one_sigma_x = -unit_solver.H[center + 1, center, 2, "y"]
    source.update(unit_solver, peak_time + sigma_time)
    one_sigma_t = -unit_solver.H[center, center, 2, "y"]

    np.testing.assert_allclose(one_sigma_x / peak, np.exp(-0.5))
    np.testing.assert_allclose(one_sigma_t / peak, np.exp(-0.5))


def test_gaussian_packet_spatial_and_spectral_widths_are_reciprocal():
    from_space = GaussianPacket(sigmaz=0.02)
    from_frequency = GaussianPacket(sigmaf=from_space.sigmaf)

    np.testing.assert_allclose(from_space.sigmaf, c / (2 * np.pi * 0.02))
    np.testing.assert_allclose(from_frequency.sigmaz, 0.02)


def test_gaussian_packet_resolves_mesh_dependent_defaults(unit_solver):
    source = GaussianPacket()

    source.update(unit_solver, 0.0)

    expected_sigmaz = 10 * np.mean(unit_solver.dz)
    assert source.sigmaz == pytest.approx(expected_sigmaz)
    assert source.sigmaf == pytest.approx(c / (2 * np.pi * expected_sigmaz))
    assert source.tinj == pytest.approx(6 * expected_sigmaz)
    assert source.sigmaxy == pytest.approx(
        5 * np.mean([np.mean(unit_solver.dx), np.mean(unit_solver.dy)])
    )


def test_gaussian_packet_rejects_unknown_field():
    with pytest.raises(ValueError, match="field"):
        GaussianPacket(field="J")


@pytest.mark.parametrize(
    "field, label, expected_sign", [("E", "Ex", 1.0), ("H", "Hy", -1.0)]
)
def test_gaussian_packet_plot_shows_injected_component(
    monkeypatch, field, label, expected_sign
):
    source = GaussianPacket(
        field=field, sigmaz=0.01, sigmaxy=0.02, tinj=0.0, amplitude=2.0
    )
    monkeypatch.setattr(plt, "show", lambda: None)

    source.plot(np.array([0.0]))

    line = plt.gcf().axes[0].lines[0]
    assert line.get_label() == label
    assert line.get_ydata()[0] == pytest.approx(expected_sign * source.amplitude)
    plt.close()


def test_gaussian_packet_plot_spectrum(monkeypatch):
    source = GaussianPacket(sigmaz=0.01)
    time = np.linspace(0.0, 12 * source.sigmaz / c, 256)
    monkeypatch.setattr(plt, "show", lambda: None)

    frequency, spectrum = source.plot_spectrum(time)

    assert frequency.shape == spectrum.shape
    assert np.all(frequency >= 0.0)
    assert np.isfinite(spectrum).all()
    assert frequency[np.argmax(spectrum)] == pytest.approx(0.0)
    assert plt.gcf().axes[0].get_xlim()[1] == pytest.approx(source.sigmaf * 3e-9)
    plt.close()


def test_gaussian_packet_propagates_with_its_sign_and_transverse_profile(
    propagation_solver, simulate_source
):
    source = GaussianPacket(
        zs=8,
        sigmaz=4 * c * propagation_solver.dt,
        sigmaxy=0.02,
        tinj=12 * c * propagation_solver.dt,
    )

    result = simulate_source(
        propagation_solver, source, ("H", "y"), ("E", "x"), ("H", "y")
    )

    assert np.max(result["injected"]) <= 0.0
    peak = np.argmax(np.abs(result["observed"]))
    assert result["observed"][peak] * result["companion"][peak] > 0.0
    assert abs(result["profile_x"][peak, propagation_solver.Nx // 2]) > 1.1 * abs(
        result["profile_x"][peak, 0]
    )
