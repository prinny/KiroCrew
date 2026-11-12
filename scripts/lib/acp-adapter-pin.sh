# shellcheck shell=sh
# Shared by install.sh, setup.sh and minimal_install.sh: the npm spec
# (`<name>@<exact version>`) for an optional ACP adapter backend.
#
# The versions live in packaging/acp-adapters/package.json so Dependabot can
# bump them; this helper only reads them. Sourced (setup.sh is itself sourced
# by bash or zsh), so POSIX sh only.
#
# Usage: acp_adapter_spec <repo-root> [package-name]
# The package defaults to claude-agent-acp; a caller naming another package
# pinned in the same manifest (muse-acp) gets that adapter's spec instead.
# Prints the spec, or nothing (exit 1) when the pinned version cannot be read.
# Callers then skip the install instead of falling back to an unpinned one.

ACP_NPM_PKG="@agentclientprotocol/claude-agent-acp"

acp_adapter_spec() {
    _acp_pkg_json="$1/packaging/acp-adapters/package.json"
    _acp_pkg_name="${2:-$ACP_NPM_PKG}"
    [ -f "$_acp_pkg_json" ] || return 1
    command -v node >/dev/null 2>&1 || return 1
    _acp_ver="$(ACP_PKG_JSON="$_acp_pkg_json" ACP_PKG_NAME="$_acp_pkg_name" node -p \
        'require(process.env.ACP_PKG_JSON).dependencies[process.env.ACP_PKG_NAME] || ""' \
        2>/dev/null)" || return 1
    # Exact x.y.z only. A range, a tag or an empty value is refused.
    case "$_acp_ver" in
        ''|*[!0-9.]*) return 1 ;;
    esac
    printf '%s@%s\n' "$_acp_pkg_name" "$_acp_ver"
}
