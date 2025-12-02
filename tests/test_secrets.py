import os

import pytest

import llm_wrappers.secrets as secrets


def test_get_secret_from_env(monkeypatch):
    monkeypatch.setenv("TEST_SECRET_ENV", "env-value-123")
    # Should pick up from environment
    assert secrets.get_secret("TEST_SECRET_ENV") == "env-value-123"


def test_get_secret_from_keychain(monkeypatch):
    # Ensure env var absent
    monkeypatch.delenv("TEST_SECRET_KC", raising=False)

    # Patch subprocess.check_output used by the keychain fallback
    def fake_check_output(args, stderr=None):
        # emulate macOS security CLI returning the secret bytes
        return b"kc-value-xyz"

    monkeypatch.setattr(secrets.subprocess, "check_output", fake_check_output)

    val = secrets.get_secret("TEST_SECRET_KC")
    assert val == "kc-value-xyz"


def test_get_secret_not_found(monkeypatch):
    # No env, keychain raises -> get_secret returns None
    monkeypatch.delenv("TEST_NOTHING", raising=False)

    def raise_err(*args, **kwargs):
        raise Exception("not found")

    monkeypatch.setattr(secrets.subprocess, "check_output", raise_err)
    assert secrets.get_secret("TEST_NOTHING") is None


def test_set_env_from_secret(monkeypatch):
    # Patch get_secret to return a value
    monkeypatch.setattr(secrets, "get_secret", lambda name: "from-get-secret")
    # Ensure var not present
    if "SOME_SECRET" in os.environ:
        del os.environ["SOME_SECRET"]
    v = secrets.set_env_from_secret("SOME_SECRET")
    assert v == "from-get-secret"
    assert os.environ.get("SOME_SECRET") == "from-get-secret"
