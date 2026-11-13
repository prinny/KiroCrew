# kirocrew-bedrock-acp

An ACP agent that drives Amazon Bedrock's Converse API. It speaks ACP v1
(newline-delimited JSON-RPC over stdio) to Kiro Crew and, inside, calls
Bedrock through boto3's default credential chain — profiles,
`credential_process`, static keys, session tokens, instance roles — resolved
in the adapter's own process. Crew never hands it a credential: it is spawned
because an operator registered it under `agent.custom_backends`, and that
registration's env rules make a credential binding impossible to declare in
plaintext.

## Install

```sh
pip install kirocrew-bedrock-acp
```

This provides the `kirocrew-bedrock-acp` command.

## Register it with Kiro Crew

Add one entry to `agent.custom_backends` in `config.json`:

```json
{
  "agent": {
    "custom_backends": {
      "bedrock": {
        "label": "Bedrock (ACP adapter)",
        "command": "kirocrew-bedrock-acp",
        "args": ["--region", "us-west-2"],
        "install_hint": "pip install kirocrew-bedrock-acp",
        "env": { "AWS_PROFILE": "default" },
        "permission": {
          "kind": "session_config",
          "option_id": "approval_mode",
          "value": "promptUnmatched"
        },
        "mcp": "array",
        "credentials": { "posture": "adapter_chain", "reads": ["~/.aws"] },
        "capabilities": { "model_option_id": "model", "report_usage": true }
      }
    },
    "acp_backend": "bedrock"
  }
}
```

That entry is the entire integration. No fork code names Bedrock.

## Behaviour

- **Models.** `session/new` offers a `model` config option built from
  `ListFoundationModels` joined with `ListInferenceProfiles` on the model
  ARN: bare on-demand foundation-model ids and ACTIVE inference-profile ids,
  verbatim. A listed model is not promised invocable on the account; the
  Converse error says so when it is not. `--models a,b` replaces the
  catalogue read; `--model` picks the starting model.
- **Credentials.** boto3's default chain, in this process. Select a profile
  with the ordinary `AWS_PROFILE` (as in the registration above) or pass
  `--profile`. Nothing about the chain is Crew configuration.
- **Tools.** MCP servers from the session's `mcpServers` array (stdio
  elements) are connected by the adapter and offered to the model as
  Converse tools, named `<server>__<tool>`. Before any tool runs, the
  adapter asks through `session/request_permission` — Crew's approval flow
  decides. The `approval_mode` config option selects `promptUnmatched`
  (ask; the default) or `denyUnmatched` (deny without asking).
- **Sessions.** In-memory only: `loadSession` is false and `session/load`
  answers method-not-found. Usage from Converse is returned with each
  prompt's stop reason.
