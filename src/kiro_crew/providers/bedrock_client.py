"""The boto3 seam for the native Bedrock provider: session factory,
STS identity verification, classified errors, throttling retry.

Everything AWS-shaped that is not pure translation lives behind this
module so the provider and its tests never touch boto3 directly: tests
substitute a fake factory, and the one live surface is small enough to
audit. Credential resolution is the AWS SDK's own default chain,
unmodified — a ``boto3.Session`` built with at most a profile name and
a Region inherits the full chain (shared files, credential_process,
SSO, environment, instance metadata) exactly as the SDK resolves it,
because that inheritance is the maintainer's hard requirement: a
provider that re-implements a subset of the chain is the Strands
anti-example this provider exists to replace.

No function here returns, logs or formats a credential value. The
identity surface is the STS announcement (account id, principal ARN)
and a coarse SOURCE CLASS naming which chain member supplied the
credentials — the class, never the material.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: The credential-reference env names the vault triple maps onto. These
#: are Crew configuration names for the gateway's OWN environment; the
#: values behind them are written into the process environment before
#: the boto3 session is built, where the SDK chain's environment member
#: picks them up — no bespoke credential plumbing anywhere else.
ENV_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
ENV_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
ENV_SESSION_TOKEN = "AWS_SESSION_TOKEN"

#: Source classes for the identity announcement (design §2.5). A class
#: names WHERE the chain resolved credentials, never their value.
SOURCE_VAULT_TRIPLE = "configured secret:// references (gateway environment)"
SOURCE_PROFILE = "named profile"
SOURCE_SESSION_PROFILE = "session profile"
SOURCE_ENVIRONMENT = "gateway environment"
SOURCE_SHARED_FILES = "shared credential files"
SOURCE_SSO = "SSO token provider"
SOURCE_CONTAINER = "container credentials"
SOURCE_INSTANCE_METADATA = "instance metadata"
SOURCE_UNKNOWN = "aws sdk credential chain"

#: Client-error codes that mean the credentials themselves were refused:
#: after one of these the provider re-verifies identity before reporting
#: (design §2.5), because a rotated credential mid-session must not be
#: reported as a model refusal.
AUTH_ERROR_CODES = frozenset(
    {
        "AccessDeniedException",
        "UnrecognizedClientException",
        "InvalidClientTokenException",
        "SignatureDoesNotMatch",
        "AuthFailure",
    }
)


class BedrockUnavailableError(RuntimeError):
    """boto3 is not importable in this environment.

    Raised at provider start (fail closed), never at import time: the
    module itself imports cleanly without the optional dependency so
    every non-Bedrock session on a boto3-less install is unaffected.
    """


def boto3_available() -> bool:
    try:
        import boto3  # noqa: F401
    except Exception:
        return False
    return True


def make_boto_session(profile_name: str | None) -> Any:
    """Build the boto3 Session the provider resolves credentials through.

    Only the profile name is ever passed. Region is deliberately NOT
    set on the session: the provider's configured Region is applied per
    client, so the session's own Region resolution (profile, env) never
    silently overrides the operator's explicit ``agent.bedrock_region``.
    """
    try:
        import boto3
    except Exception as exc:  # noqa: BLE001 - re-raised as the typed error
        raise BedrockUnavailableError(
            "boto3 is not installed; the native Bedrock provider needs "
            "the kirocrew[bedrock] extra. No session was started."
        ) from exc
    if profile_name:
        return boto3.Session(profile_name=profile_name)
    return boto3.Session()


def classify_source_class(session: Any, profile_name: str | None) -> str:
    """The source class of the session's resolved credentials.

    Reads only the TYPE of the chain member botocore resolved (its
    provider class name) — never a credential value.
    """
    try:
        credentials = session.get_credentials()
    except Exception:
        return SOURCE_UNKNOWN
    if credentials is None:
        return SOURCE_UNKNOWN
    method = str(getattr(credentials, "method", "") or "")
    if method == "env":
        return SOURCE_ENVIRONMENT
    if method == "shared-credentials-file":
        return SOURCE_PROFILE if profile_name else SOURCE_SHARED_FILES
    if method == "custom-process":
        return SOURCE_PROFILE if profile_name else SOURCE_UNKNOWN
    if method == "sso":
        return SOURCE_SSO
    if method in ("ecs/container", "ec2-credentials-file"):
        return SOURCE_CONTAINER
    if method == "iam-role":
        return SOURCE_INSTANCE_METADATA
    if profile_name:
        return SOURCE_PROFILE
    return SOURCE_UNKNOWN


@dataclass
class Identity:
    """The STS announcement: who the gateway acts as on AWS."""

    account_id: str
    arn: str
    source_class: str


class BedrockCallError(RuntimeError):
    """A classified AWS call failure: code + retryability, no payload."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.retryable = retryable


def classify_client_error(exc: BaseException) -> BedrockCallError:
    """Classify a botocore ClientError (or connection failure).

    Only the error CODE and the service's own message are carried; a
    response body is never formatted into the error, so nothing an echo
    could contain travels further.
    """
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error") or {}
        code = str(error.get("Code") or type(exc).__name__)
        message = str(error.get("Message") or "")
        retryable = code in ("ThrottlingException", "ServiceUnavailableException")
        return BedrockCallError(code, message, retryable=retryable)
    return BedrockCallError(type(exc).__name__, str(exc)[:300], retryable=False)


def verify_identity(session: Any) -> Identity:
    """STS GetCallerIdentity through the session's resolved credentials.

    Fails closed: any error propagates as a classified BedrockCallError
    and the caller starts no session.
    """
    try:
        sts = session.client("sts")
        response = sts.get_caller_identity()
    except Exception as exc:  # noqa: BLE001 - classified, then raised
        if is_no_credentials_error(exc):
            raise BedrockCallError(
                "NoCredentialsError",
                "the AWS credential chain resolved no credentials",
            ) from exc
        raise classify_client_error(exc) from exc
    return Identity(
        account_id=str(response.get("Account") or ""),
        arn=str(response.get("Arn") or ""),
        source_class=classify_source_class(session, getattr(session, "profile_name", None)),
    )


def resolve_secret_refs(refs: dict[str, str], vault_home: Any) -> dict[str, str]:
    """Resolve ``secret://`` credential references against the vault.

    Deliberately NOT ``config.loader_credentials.resolve_env_credentials``:
    that resolver only resolves references filed under ``CREDENTIAL_KEYS``
    names (it is the ``.env`` reader's resolver), and the AWS environment
    names are intentionally absent from that tuple — the Docker
    entrypoint scrub covers exactly the tuple, so registering the AWS
    names would scrub the gateway's own environment and break the
    chain's environment member. A reference filed under an AWS name
    would come back from that resolver as the LITERAL ``secret://…``
    text, which a boto3 session would then present as the credential
    itself. This helper resolves the same vault directly: a reference
    that cannot be resolved is OMITTED from the result (fails closed),
    and a non-reference value is refused (config write already refuses
    plaintext in these fields; a hand-edited one must not become a
    credential either). Values are returned only to the caller that
    writes them into the process environment; nothing here logs one.
    """
    from kiro_crew.mcp_gateway.secret_uri import SECRET_URI_PREFIX, _is_valid_secret_name

    pending: dict[str, str] = {}
    for key, value in refs.items():
        value = (value or "").strip()
        if not value:
            continue
        if not value.startswith(SECRET_URI_PREFIX):
            logger.error(
                "bedrock: credential reference for %s is not a secret:// "
                "reference; refusing to use it as a credential",
                key,
            )
            continue
        name = value[len(SECRET_URI_PREFIX) :]
        if not _is_valid_secret_name(name):
            logger.error("bedrock: malformed secret:// reference for %s", key)
            continue
        pending[key] = name
    if not pending:
        return {}
    from kiro_crew.secrets import SecretVault

    try:
        fetched = SecretVault(vault_home).get_many(list(pending.values()))
    except Exception:  # noqa: BLE001 - any vault fault fails closed
        fetched = None
    resolved: dict[str, str] = {}
    for key, name in pending.items():
        secret = fetched.get(name) if fetched else None
        if secret is None:
            logger.error(
                "bedrock: the vault holds no resolvable secret for %s; "
                "that credential is treated as unset",
                key,
            )
            continue
        resolved[key] = secret.reveal()
    return resolved


def is_no_credentials_error(exc: BaseException) -> bool:
    """True when botocore's chain resolved NO credentials at all."""
    return type(exc).__name__ in ("NoCredentialsError", "PartialCredentialsError")


def converse_with_retry(
    call: Callable[[], Any],
    *,
    max_retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[float, float], float] = random.uniform,
) -> Any:
    """Run one Converse-family call with the design's bounded retry.

    Only ThrottlingException is retried (design Q8): at most
    ``max_retries`` retries with exponential backoff plus jitter. Every
    other failure, and a throttle that outlives the bound, raises the
    CLASSIFIED error (BedrockCallError) as the floor — a throttled call
    is reported as throttled, never silently swapped or hidden.
    """
    attempt = 0
    while True:
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - classified below
            if is_no_credentials_error(exc):
                raise BedrockCallError(
                    "NoCredentialsError",
                    "the AWS credential chain resolved no credentials",
                ) from exc
            classified = classify_client_error(exc)
            if classified.code == "ThrottlingException" and attempt < max_retries:
                attempt += 1
                sleep(min(2.0**attempt, 8.0) + jitter(0.0, 0.5))
                continue
            raise classified from exc
