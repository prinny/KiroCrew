"""One-time pruning of command entries whose legacy owner cannot be recovered."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from pathlib import Path

from kiro_crew import agent as agent_mod
from kiro_crew.agent_materialization import guide_platform, mcp_sources
from kiro_crew.mcp_provenance import APP_KEY
from kiro_crew.mcp_utils import mcp_server_alias
from kiro_crew.template_mcp_additions import MERGED_MARK

MIGRATION_MARKER = "legacy_mcp_launches_pruned.json"


def pending(*, path: Path | None = None) -> bool:
    """An absent, unreadable or unrecognized record cannot attest completion.

    Only the default spec participates: other output paths never commit this
    record and must not repeatedly adopt transports as if they were migrating.
    """
    if path is not None:
        from kiro_crew.apps.bridges import _mcp_json_path

        try:
            if path.resolve() != _mcp_json_path().resolve():
                return False
        except OSError:
            return False
    try:
        record = json.loads(guide_platform._marker(MIGRATION_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return True
    return record != {"version": 1}


def mark_complete() -> None:
    """Called under the spec lock after its successful write; failures retry next time."""
    try:
        guide_platform._mark(MIGRATION_MARKER)
    except OSError:
        agent_mod.logger.warning(
            "Legacy MCP migration completion could not be recorded; will retry"
        )


def is_unmarked_launch(name: str, entry: object) -> bool:
    """Whether a carried entry is a launch whose legacy ownership needs a source."""
    return (
        "/" not in name
        and ":" not in name
        and isinstance(entry, dict)
        and APP_KEY not in entry
        and MERGED_MARK not in entry
        and isinstance(entry.get("command"), str)
        and bool(entry["command"])
    )


def prune_unowned_launches(
    servers: dict,
    on_disk: dict,
    *,
    sources: mcp_sources.McpSources,
    mounted_aliases: Mapping[str, str],
    app_claimed: Collection[str],
    unresolved: Collection[str] = (),
) -> set[str]:
    """Remove unmarked launches from both candidates so the locked merge cannot restore them.

    Normalization maps carried entries to themselves too. Only a source's own
    transport can vouch through that map, not mere presence in the map.
    Collected apps carry APP_KEY and card additions carry MERGED_MARK.
    """
    platform = sources.managed_names | set(agent_mod._extra_mcp_servers())
    declared = mcp_sources.declared_mcp_sources(
        (sources.provider_global, sources.kiro_global, sources.kirocrew), app_claimed
    )
    vouched = platform | {
        target for source, target in mounted_aliases.items() if source in declared | platform
    }
    # Resolution omits these sources before normalization can record a target.
    vouched.update(mcp_server_alias(name) for name in declared if name in unresolved)
    removed = set()
    for name, entry in {**on_disk, **servers}.items():
        if name in vouched or not is_unmarked_launch(name, entry):
            continue
        servers.pop(name, None)
        on_disk.pop(name, None)
        removed.add(name)
        agent_mod.logger.warning(
            "Removed %r as an unowned legacy MCP launch; re-add it from the MCP servers page",
            name,
        )
    return removed
