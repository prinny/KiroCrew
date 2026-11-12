"""Credential-chain tests for the native Bedrock provider.

The maintainer's hard requirement: resolution is the FULL AWS SDK
chain, gateway-side — not a bearer key, not env-only. These tests pin
the chain members the provider must inherit by construction (a boto3
Session with at most a profile name): environment credentials, shared
files, a named profile, and a credential_process helper, each resolved
by the REAL SDK in a hermetic HOME. Plus the Crew-side surfaces:
per-session profile binding reaches the session factory, the vault
triple is applied only when no profile is selected (§2.3 precedence),
the account pin refuses a mismatched identity, and an empty chain
fails closed at start().
"""

from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path

import pytest
from bedrock_stub import TEST_ACCOUNT, FakeSession, make_provider

from kiro_crew.providers.bedrock_client import (
    SOURCE_ENVIRONMENT,
    SOURCE_PROFILE,
    classify_source_class,
    make_boto_session,
    verify_identity,
)
from kiro_crew.providers.meta_api_client import AppServerError

FAKE_KEY_ID = "AKIAEXAMPLE00000000"
FAKE_SECRET = "example-secret-value-never-real"


def _hermetic_env(monkeypatch, tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    return home


def test_env_credentials_resolve_through_the_real_chain(monkeypatch, tmp_path):
    _hermetic_env(monkeypatch, tmp_path)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", FAKE_KEY_ID)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", FAKE_SECRET)
    monkeypatch.setenv("AWS_SESSION_TOKEN", "example-session-token")
    session = make_boto_session(None)
    credentials = session.get_credentials()
    assert credentials is not None
    assert credentials.access_key == FAKE_KEY_ID
    assert credentials.token == "example-session-token"
    assert classify_source_class(session, None) == SOURCE_ENVIRONMENT


def test_shared_file_profile_resolves_through_the_real_chain(monkeypatch, tmp_path):
    home = _hermetic_env(monkeypatch, tmp_path)
    aws_dir = home / ".aws"
    aws_dir.mkdir()
    (aws_dir / "credentials").write_text(
        f"[crewtest]\naws_access_key_id = {FAKE_KEY_ID}\naws_secret_access_key = {FAKE_SECRET}\n",
        encoding="utf-8",
    )
    (aws_dir / "config").write_text("[profile crewtest]\nregion = us-west-2\n", encoding="utf-8")
    session = make_boto_session("crewtest")
    credentials = session.get_credentials()
    assert credentials is not None and credentials.access_key == FAKE_KEY_ID
    assert classify_source_class(session, "crewtest") == SOURCE_PROFILE


def test_credential_process_profile_resolves_through_the_real_chain(monkeypatch, tmp_path):
    home = _hermetic_env(monkeypatch, tmp_path)
    aws_dir = home / ".aws"
    aws_dir.mkdir()
    helper = tmp_path / "helper.py"
    helper.write_text(
        "import json\n"
        f"print(json.dumps({{'Version': 1, 'AccessKeyId': '{FAKE_KEY_ID}', "
        f"'SecretAccessKey': '{FAKE_SECRET}', 'SessionToken': 'proc-token'}}))\n",
        encoding="utf-8",
    )
    helper.chmod(helper.stat().st_mode | stat.S_IEXEC)
    (aws_dir / "config").write_text(
        f"[profile crewproc]\nregion = us-west-2\ncredential_process = python3 {helper}\n",
        encoding="utf-8",
    )
    session = make_boto_session("crewproc")
    credentials = session.get_credentials()
    assert credentials is not None
    assert credentials.access_key == FAKE_KEY_ID
    assert credentials.token == "proc-token"
    assert classify_source_class(session, "crewproc") == SOURCE_PROFILE


def test_empty_chain_yields_no_credentials(monkeypatch, tmp_path):
    _hermetic_env(monkeypatch, tmp_path)
    session = make_boto_session(None)
    assert session.get_credentials() is None


def test_verify_identity_announces_account_and_source():
    identity = verify_identity(FakeSession())
    assert identity.account_id == TEST_ACCOUNT
    assert identity.arn.endswith("user/kirocrew-test")
    assert identity.source_class  # a class is always announced


def test_session_factory_receives_the_bound_profile(tmp_path):
    async def main():
        seen: list = []

        def factory(profile):
            seen.append(profile)
            return FakeSession()

        provider, _session = make_provider(
            tmp_path, session_factory=factory, bedrock_profile="crew-bedrock"
        )
        await provider.start()
        assert seen == ["crew-bedrock"]
        assert provider.is_alive()

    asyncio.run(main())


class _FakeSecret:
    def __init__(self, value):
        self._value = value

    def reveal(self):
        return self._value


class _FakeVault:
    """Stands in for SecretVault: holds the two static-key entries."""

    def __init__(self, _home):
        pass

    def get_many(self, names):
        values = {
            "AWS_ACCESS_KEY_ID": FAKE_KEY_ID,
            "AWS_SECRET_ACCESS_KEY": FAKE_SECRET,
        }
        return {name: _FakeSecret(values[name]) if name in values else None for name in names}


def test_resolve_secret_refs_never_passes_the_literal_through(tmp_path):
    """Regression pin for the resolver trap: ``resolve_env_credentials``
    only resolves references under CREDENTIAL_KEYS names, so a reference
    filed under an AWS name comes back as the literal ``secret://…``
    text. The provider's own helper must resolve the VALUE — and omit
    what it cannot resolve, never pass the literal through."""
    import kiro_crew.secrets as secrets_mod
    from kiro_crew.providers.bedrock_client import resolve_secret_refs

    original = secrets_mod.SecretVault
    secrets_mod.SecretVault = _FakeVault
    try:
        resolved = resolve_secret_refs(
            {
                "AWS_ACCESS_KEY_ID": "secret://AWS_ACCESS_KEY_ID",
                "AWS_SESSION_TOKEN": "secret://AWS_SESSION_TOKEN",
                "AWS_SECRET_ACCESS_KEY": "plaintext-is-refused",
            },
            tmp_path,
        )
    finally:
        secrets_mod.SecretVault = original
    assert resolved == {"AWS_ACCESS_KEY_ID": FAKE_KEY_ID}
    assert all(not v.startswith("secret://") for v in resolved.values())


def test_vault_triple_applied_only_without_profile(tmp_path, monkeypatch):
    async def main():
        from kiro_crew.providers import bedrock as bedrock_mod

        monkeypatch.setattr("kiro_crew.secrets.SecretVault", _FakeVault)
        for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
            monkeypatch.delenv(name, raising=False)
        refs = {
            "AWS_ACCESS_KEY_ID": "secret://AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY": "secret://AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN": "secret://AWS_SESSION_TOKEN",
        }
        provider, _ = make_provider(tmp_path, credential_refs=refs)
        await provider.start()
        # The env carries the resolved VALUES, not the reference text.
        assert os.environ.get("AWS_ACCESS_KEY_ID") == FAKE_KEY_ID
        assert os.environ.get("AWS_SECRET_ACCESS_KEY") == FAKE_SECRET
        assert "AWS_SESSION_TOKEN" not in os.environ  # unresolvable: omitted
        assert bedrock_mod.SOURCE_VAULT_TRIPLE == provider._identity_source

        for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
            monkeypatch.delenv(name, raising=False)
        profiled, _ = make_provider(tmp_path, credential_refs=refs, bedrock_profile="crew-bedrock")
        await profiled.start()
        # §2.3 precedence: with a profile bound, the triple is NOT applied.
        assert "AWS_ACCESS_KEY_ID" not in os.environ

    asyncio.run(main())


def test_account_pin_refuses_mismatched_identity(tmp_path):
    async def main():
        provider, _ = make_provider(tmp_path, expected_account_id="999999999999")
        with pytest.raises(AppServerError, match="account"):
            await provider.start()
        assert provider.startup_notice is not None
        assert provider.startup_notice[0] == "bedrock_account_mismatch"

    asyncio.run(main())


def test_no_credentials_fails_closed_at_start(tmp_path):
    async def main():
        import botocore.exceptions

        class NoCredSession(FakeSession):
            def client(self, name, region_name=None):
                if name == "sts":
                    raise botocore.exceptions.NoCredentialsError()
                return super().client(name, region_name)

        provider, _ = make_provider(tmp_path, session_factory=lambda profile: NoCredSession())
        with pytest.raises(AppServerError):
            await provider.start()
        assert provider.startup_notice is not None
        assert provider.startup_notice[0] == "bedrock_identity_failed"

    asyncio.run(main())


def test_bearer_token_in_gateway_env_refuses_start(tmp_path, monkeypatch):
    async def main():
        monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "bearer-token-not-a-source")
        provider, _ = make_provider(tmp_path)
        with pytest.raises(AppServerError, match="bearer token"):
            await provider.start()
        assert provider.startup_notice[0] == "bedrock_bearer_token_present"

    asyncio.run(main())


def test_region_fails_closed_when_nowhere_configured(tmp_path):
    async def main():
        session = FakeSession()
        session.region_name = None
        provider, _ = make_provider(tmp_path, session_factory=lambda profile: session)
        with pytest.raises(AppServerError, match="Region"):
            await provider.start()
        assert provider.startup_notice[0] == "bedrock_region_missing"

    asyncio.run(main())
