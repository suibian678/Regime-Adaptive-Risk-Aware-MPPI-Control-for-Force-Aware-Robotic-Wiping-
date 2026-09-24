from types import SimpleNamespace

import pytest

from forcewipe.threshold_sensitive_actor_to_mppi import (
    configure_threshold_sensitive_actor_to_mppi,
)


def test_activation_fraction_is_installed_after_base_configuration(monkeypatch):
    monkeypatch.setattr(
        "forcewipe.threshold_sensitive_actor_to_mppi.configure_actor_to_mppi_contact_mode",
        lambda cfg: cfg,
    )
    cfg = SimpleNamespace()
    configured = configure_threshold_sensitive_actor_to_mppi(
        cfg, activation_fraction=0.75
    )
    assert configured.mppi_tracking_activation_fraction == pytest.approx(0.75)


@pytest.mark.parametrize("value", [-0.1, 0.0, 1.0, 1.1])
def test_activation_fraction_must_be_open_unit_interval(value):
    with pytest.raises(ValueError):
        configure_threshold_sensitive_actor_to_mppi(
            SimpleNamespace(), activation_fraction=value
        )
