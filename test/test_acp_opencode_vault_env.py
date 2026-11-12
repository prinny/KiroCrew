"""``agent.opencode_env``: provider keys for the OpenCode harness, from the vault.

The map is ``agent.deepseek_env``'s shape pointed at a different harness: a
provider key lives in Kiro Crew's secret vault and reaches the OpenCode child
as an environment variable at spawn, so a custom provider declared in
OpenCode's own configuration (``apiKey: "{env:MODEL_API_KEY}"``) can resolve
its credential without the key ever being written to ``config.json`` or to an
``opencode.json``.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from kiro_crew.acp import client as client_module
from kiro_crew.acp.harness import opencode as opencode_mod
from kiro_crew.acp.transport_errors import AcpToolGateUnroutable


def _mapping(name: str = "MODEL_API_KEY", value: str = "secret://meta-model-api") -> dict:
    return {name: value}


# ── The spawn-time validator ──


def test_a_provider_key_mapping_the_harness_honours_is_accepted() -> None:
    """The happy shape passes every rule, so the refusals are not vacuous."""
    client_module._validate_opencode_env_mapping(_mapping())
    # Any provider credential name the operator's OpenCode config interpolates
    # works; nothing here is Meta-specific.
    client_module._validate_opencode_env_mapping({"OPENAI_API_KEY": "secret://other"})
    client_module._validate_opencode_env_mapping({"CUSTOM_PROVIDER_TOKEN": "secret://other"})


@pytest.mark.parametrize(
    "mapping",
    [
        {"MODEL_API_KEY": "sk-live-plaintext"},  # a literal value, not a reference
        {"OPENCODE_CONFIG_CONTENT": "secret://x"},  # the harness's own namespace
        {"OPENCODE_API_KEY": "secret://x"},  # the harness's own namespace
        {"KIROCREW_SESSION_KEY": "secret://x"},  # Crew's identity namespace
        {"KIRO_API_KEY": "secret://x"},  # kiro-cli's own model credential
        {"XDG_DATA_HOME": "secret://x"},  # the credential floor's override
        {"AWS_SECRET_ACCESS_KEY": "secret://x"},  # Crew's spawn scrub strips it
        {"1MODEL_API_KEY": "secret://x"},  # not a POSIX identifier
        {"MODEL-API-KEY": "secret://x"},  # not a POSIX identifier
    ],
)
def test_a_provider_key_mapping_the_harness_would_not_honour_is_refused(mapping) -> None:
    with pytest.raises(ValueError) as refused:
        client_module._validate_opencode_env_mapping(mapping)
    name = next(iter(mapping))
    assert repr(name) in str(refused.value), "the refusal must name the operator's own key"


def test_a_refusal_never_carries_the_value_or_the_vault_name() -> None:
    with pytest.raises(ValueError) as refused:
        opencode_mod._validate_opencode_env_mapping({"MODEL_API_KEY": "sk-live-plaintext"})
    assert "sk-live-plaintext" not in str(refused.value)
    with pytest.raises(ValueError) as refused:
        opencode_mod._validate_opencode_env_mapping({"OPENCODE_API_KEY": "secret://my-vault-name"})
    assert "my-vault-name" not in str(refused.value)


def test_on_windows_a_case_variant_of_a_reserved_name_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(opencode_mod.os, "name", "nt")
    with pytest.raises(ValueError):
        opencode_mod._validate_opencode_env_mapping({"opencode_api_key": "secret://x"})


# ── The publish floor ──


def test_a_plaintext_value_never_reaches_disk_through_the_locked_write(tmp_path) -> None:
    """The publish floor: ``config set`` cannot persist a provider key in plaintext."""
    from kiro_crew.config.loader import ConfigWriteRefused, update_config_locked

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"agent": {"yolo": True}}), encoding="utf-8")
    before = path.read_bytes()

    def plant_plaintext(existing: dict) -> dict:
        existing.setdefault("agent", {})["opencode_env"] = {"MODEL_API_KEY": "sk-live-plaintext"}
        return existing

    with pytest.raises(ConfigWriteRefused) as refused:
        update_config_locked(path, mutate=plant_plaintext)

    message = str(refused.value)
    assert "'MODEL_API_KEY'" in message, "the refusal must name the operator's own env-var key"
    assert "holds a literal value" in message
    assert "sk-live-plaintext" not in message, "the refusal must never carry the value"
    assert path.read_bytes() == before, "a refused write must leave the file byte-identical"
    assert not list(tmp_path.glob("*.tmp*")), "no temp file may survive a refused publish"


def test_a_secret_reference_passes_the_publish_floor(tmp_path) -> None:
    from kiro_crew.config.loader import update_config_locked

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"agent": {}}), encoding="utf-8")

    def plant_reference(existing: dict) -> dict:
        existing.setdefault("agent", {})["opencode_env"] = _mapping()
        return existing

    update_config_locked(path, mutate=plant_reference)
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["agent"]["opencode_env"] == _mapping()


# ── Resolution and spawn injection ──


def _seed_config_and_vault(tmp_path, monkeypatch, mapping, secret=None) -> None:
    """Point Kiro Crew at a scratch home holding *mapping* and, optionally, its secret."""
    from kiro_crew.config.paths import config_dir
    from kiro_crew.secrets import SecretVault

    monkeypatch.setenv("KIROCREW_HOME", str(tmp_path / "home"))
    config_dir().mkdir(parents=True, exist_ok=True)
    (config_dir() / "config.json").write_text(
        json.dumps({"agent": {"opencode_env": mapping}}), encoding="utf-8"
    )
    if secret is not None:
        SecretVault(config_dir()).set_sync(*secret)


def test_an_empty_mapping_resolves_to_no_env(tmp_path, monkeypatch) -> None:
    _seed_config_and_vault(tmp_path, monkeypatch, {})
    resolved, secret_keys = opencode_mod._opencode_vault_env()
    assert resolved == {}
    assert secret_keys == ()


def test_a_mapped_key_resolves_from_the_vault(tmp_path, monkeypatch) -> None:
    _seed_config_and_vault(
        tmp_path, monkeypatch, _mapping(), secret=("meta-model-api", "sk-test-value")
    )
    resolved, secret_keys = opencode_mod._opencode_vault_env()
    assert resolved == {"MODEL_API_KEY": "sk-test-value"}
    assert secret_keys == ("MODEL_API_KEY",)


def test_a_missing_vault_secret_refuses_rather_than_starting_without_a_key(
    tmp_path, monkeypatch
) -> None:
    _seed_config_and_vault(tmp_path, monkeypatch, _mapping(), secret=None)
    with pytest.raises(ValueError):
        opencode_mod._opencode_vault_env()


def test_prepare_spawn_env_places_the_key_and_keeps_the_routing_seed(
    tmp_path, monkeypatch
) -> None:
    _seed_config_and_vault(
        tmp_path, monkeypatch, _mapping(), secret=("meta-model-api", "sk-test-value")
    )
    adapter = opencode_mod.OpencodeLaunch()
    adapter._config_content = '{"permission": "ask"}'
    env: dict[str, str] = {}
    asyncio.run(adapter.prepare_spawn_env(env, to_thread=asyncio.to_thread))
    assert env["MODEL_API_KEY"] == "sk-test-value"
    assert env[opencode_mod._ENV_OPENCODE_CONFIG_CONTENT] == '{"permission": "ask"}'


def test_prepare_spawn_env_with_no_mapping_adds_nothing(tmp_path, monkeypatch) -> None:
    _seed_config_and_vault(tmp_path, monkeypatch, {})
    adapter = opencode_mod.OpencodeLaunch()
    env: dict[str, str] = {}
    asyncio.run(adapter.prepare_spawn_env(env, to_thread=asyncio.to_thread))
    assert env == {}


def test_prepare_spawn_env_refuses_a_plaintext_mapping_closed(tmp_path, monkeypatch) -> None:
    """A hand-edited plaintext mapping refuses the session; it never spawns keyless."""
    _seed_config_and_vault(tmp_path, monkeypatch, {"MODEL_API_KEY": "sk-live-plaintext"})
    adapter = opencode_mod.OpencodeLaunch()
    env: dict[str, str] = {}
    with pytest.raises(AcpToolGateUnroutable):
        asyncio.run(adapter.prepare_spawn_env(env, to_thread=asyncio.to_thread))
    assert "MODEL_API_KEY" not in env
