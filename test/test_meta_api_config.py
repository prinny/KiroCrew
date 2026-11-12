"""Config-surface tests for the native Meta Model API provider.

Covers the section 3.2 surfaces that are pure config: the widened
``agent.provider`` enum (round-trip + degrade), the plaintext-refusal
floor for ``agent.meta_api_secret`` at write time, credential resolution
precedence (with the key value asserted absent from logs), and the
factory dispatch (meta_api builds the native provider; member/worker
sessions are refused; unknown values fail closed).
"""

from __future__ import annotations

import json
import logging
from dataclasses import fields

import pytest
from config_loader_helpers import _load_from_dict

from kiro_crew.config.loader import AgentConfig
from kiro_crew.config.loader_document import write_config_atomically

FAKE_KEY = "sk-TESTKEY-DO-NOT-LOG"


def test_provider_enum_is_two_values_acp_first():
    provider = next(f for f in fields(AgentConfig) if f.name == "provider")
    assert provider.default == "acp"
    assert provider.metadata["enum"] == ["acp", "meta_api"]


def test_meta_api_fields_have_safe_defaults():
    by_name = {f.name: f for f in fields(AgentConfig)}
    assert by_name["meta_api_secret"].default == "secret://MODEL_API_KEY"
    assert by_name["meta_api_base_url"].default == ""
    assert by_name["meta_api_monthly_budget_usd"].default == 0.0


def test_meta_api_config_round_trips():
    cfg = _load_from_dict(
        {
            "agent": {
                "provider": "meta_api",
                "meta_api_secret": "secret://MODEL_API_KEY",
                "meta_api_base_url": "https://api.meta.ai/v1",
                "meta_api_monthly_budget_usd": 25.0,
            }
        }
    )
    assert cfg.agent.provider == "meta_api"
    td = cfg.to_dict()
    assert td["agent"]["provider"] == "meta_api"
    assert td["agent"]["meta_api_monthly_budget_usd"] == 25.0


def test_unknown_provider_value_degrades_to_acp():
    cfg = _load_from_dict({"agent": {"provider": "not-a-provider"}})
    assert cfg.agent.provider == "acp"


# -- plaintext floor -----------------------------------------------------------


def test_plaintext_meta_api_secret_refused_at_write(tmp_path):
    path = tmp_path / "config.json"
    with pytest.raises(ValueError, match="meta_api_secret"):
        write_config_atomically(path, {"agent": {"meta_api_secret": FAKE_KEY}})
    assert not path.exists()


def test_secret_reference_writes_fine(tmp_path):
    path = tmp_path / "config.json"
    write_config_atomically(path, {"agent": {"meta_api_secret": "secret://MODEL_API_KEY"}})
    assert json.loads(path.read_text())["agent"]["meta_api_secret"] == ("secret://MODEL_API_KEY")


def test_preexisting_plaintext_not_rewritten_passes(tmp_path):
    """The floor refuses what a write INTRODUCES (mirrors the vault-env
    rule): a hand-edited plaintext already on disk does not brick every
    later unrelated save."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"agent": {"meta_api_secret": FAKE_KEY}}))
    write_config_atomically(
        path,
        {"agent": {"meta_api_secret": FAKE_KEY}, "dashboard": {"theme_mode": "dark"}},
    )
    # But CHANGING the plaintext to another plaintext is a new introduction.
    with pytest.raises(ValueError, match="meta_api_secret"):
        write_config_atomically(path, {"agent": {"meta_api_secret": FAKE_KEY + "2"}})


def test_changing_plaintext_to_a_reference_is_allowed(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"agent": {"meta_api_secret": FAKE_KEY}}))
    write_config_atomically(path, {"agent": {"meta_api_secret": "secret://MODEL_API_KEY"}})
    assert "secret://" in path.read_text()


# -- credential resolution -----------------------------------------------------

from kiro_crew.providers.meta_api import resolve_meta_api_key  # noqa: E402


def test_resolve_prefers_vault_then_env(monkeypatch):
    monkeypatch.setattr(
        "kiro_crew.config.loader_credentials.resolve_env_credentials",
        lambda mapping: {"MODEL_API_KEY": "vault-key"},
    )
    monkeypatch.setenv("MODEL_API_KEY", "env-key")
    key, source = resolve_meta_api_key("secret://MODEL_API_KEY")
    assert key == "vault-key"
    assert "secret://" in source


def test_resolve_falls_back_to_env_when_vault_empty(monkeypatch):
    monkeypatch.setattr(
        "kiro_crew.config.loader_credentials.resolve_env_credentials",
        lambda mapping: {"MODEL_API_KEY": ""},
    )
    monkeypatch.setenv("MODEL_API_KEY", "env-key")
    key, _source = resolve_meta_api_key("secret://MODEL_API_KEY")
    assert key == "env-key"


def test_resolve_env_only(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "env-key")
    key, source = resolve_meta_api_key("")
    assert key == "env-key"
    assert "environment" in source


def test_resolve_plaintext_ref_fails_closed_and_never_logs_the_value(monkeypatch, caplog):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with caplog.at_level(logging.DEBUG, logger="kiro_crew.providers.meta_api"):
        key, _source = resolve_meta_api_key(FAKE_KEY)
    assert key == ""
    assert FAKE_KEY not in caplog.text


def test_resolve_nothing_configured_fails_closed(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setattr(
        "kiro_crew.config.loader_credentials.resolve_env_credentials",
        lambda mapping: {"MODEL_API_KEY": ""},
    )
    assert resolve_meta_api_key("secret://MODEL_API_KEY") == ("", "")


# -- factory dispatch ------------------------------------------------------------


def test_factory_builds_the_native_provider():
    from kiro_crew.providers.meta_api import MetaApiProvider

    cfg = _load_from_dict({"agent": {"provider": "meta_api"}})
    factory = cfg.create_provider_factory()
    provider = factory(session_key="meta-api-test-session")
    assert isinstance(provider, MetaApiProvider)
    assert provider.is_meta_api_backend is True
    assert provider.context_provider_type == "meta_api"


def test_factory_refuses_member_sessions():
    cfg = _load_from_dict({"agent": {"provider": "meta_api"}})
    factory = cfg.create_provider_factory()
    with pytest.raises(ValueError, match="member or worker"):
        factory(session_key="some-session", member_context=True)


def test_factory_fails_closed_on_unknown_provider():
    cfg = _load_from_dict({"agent": {"provider": "acp"}})
    cfg.agent.provider = "bogus"
    with pytest.raises(ValueError, match="Unknown agent.provider"):
        cfg.create_provider_factory()
