import pytest
from blackbox.app.app import CONFIG_NAME_MAPPER, create_app


@pytest.mark.parametrize(
    "config,check,value",
    [
        ("development", "DEBUG", True),
        ("production", "DEBUG", False),
        ("testing", "TESTING", True),
    ],
)
def test_create_app_with_config(config, check, value):
    """
    Tests the creation of the Flask app with specified configuration.

    Args:
        config (str): Flask app config.
        check (str): Flask app config key to check.
        value (bool): the expected value for check.
    """
    app = create_app(config=config)
    assert app.config[check] == value


def test_create_app_with_env_config(monkeypatch):
    """Tests the creation of the Flask app with specified configuration in FLASK_ENV
    environmental variable."""
    monkeypatch.setenv("FLASK_ENV", "development")
    app = create_app()
    assert app.config["DEBUG"]


def test_create_app_with_non_existant_config():
    """Tests that KeyError is raised with a non existant configuration."""
    with pytest.raises(KeyError):
        create_app("this-config-does-not-exist")


def test_create_app_with_broken_config():
    """Test that ImportError is raised when a configuration cannot be imported."""
    CONFIG_NAME_MAPPER["broken"] = "broken"
    with pytest.raises(ImportError):
        create_app("broken")
    del CONFIG_NAME_MAPPER["broken"]
