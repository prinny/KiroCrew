import { api } from '../../api/client'
import type {
  ProviderAdapter,
  ProviderCapabilities,
  ProviderLabels,
  AgentBinding,
  NormalizedUsage,
  NormalizedProviderHook,
  NormalizedPlugin,
  ModelInfo,
  PermissionMode,
} from '../types'

// The native Meta Model API provider's catalogue, mirroring the backend's
// static registry (src/kiro_crew/providers/meta_api_models.py). The picker
// reads THIS list for a meta_api session: the loop's own model/list never
// carries Muse ids (the backend's probe P-6), so there is no live list to
// prefer. Windows are the published 1,048,576 for every Muse model.
const MUSE_MODELS: ModelInfo[] = [
  { name: 'muse-spark-1.3', description: "Meta's flagship agentic model (Standard tier).", contextWindow: 1_048_576 },
  { name: 'muse-spark-1.3-contributor', description: 'Contributor tier: far cheaper because it grants the API operator training rights over the traffic. Never a default.', contextWindow: 1_048_576 },
  { name: 'muse-wonder-1.0', description: "Meta's highest-capability model.", contextWindow: 1_048_576 },
  { name: 'muse-ember-1.0', description: "Meta's fast, low-cost model; long-context pricing steps up past 200k input tokens.", contextWindow: 1_048_576 },
]

const MUSE_WINDOWS: Record<string, number> = Object.fromEntries(
  MUSE_MODELS.map(m => [m.name, m.contextWindow ?? 1_048_576]),
)
const DEFAULT_CONTEXT = 200_000

interface KirocrewAgentConfig {
  agent?: { model?: string; reasoning_effort?: string }
}

/**
 * Adapter for the fork's native Meta Model API provider
 * (`agent.provider = "meta_api"`). A deliberately REDUCED subset of the ACP
 * adapter, matching the provider's declared capabilities: no hooks, no
 * plugin registry, no agent templates, no subagents — the surfaces that
 * remain (models, context windows, per-turn USD usage, reasoning effort,
 * session resume, compaction) are the ones the provider actually serves.
 */
export class MetaApiAdapter implements ProviderAdapter {
  readonly id = 'meta_api' as const
  readonly displayName = 'Meta Model API'

  readonly capabilities: ProviderCapabilities = {
    hooks: false,
    pluginRegistry: false,
    agentTemplates: false,
    usageBilling: true,
    warmPool: false,
    modelResolution: true,
    toolExecution: true,
    subagents: false,
    sandbox: true,
    contextWindow: true,
    modelSwitching: true,
    permissionModes: true,
    sessionResume: true,
    compaction: true,
    reasoningEffort: true,
  }

  readonly labels: ProviderLabels = {
    sessionProcess: 'Codex app-server process',
    processCountLabel: 'app_server',
    configFile: 'kirocrew.json',
    pluginRegistryName: 'None',
    hooksSection: 'Hooks (not served by this provider)',
  }

  resolveAgentTemplate(agent: AgentBinding): string {
    // Native sessions have no kiro agent template; the binding name is the
    // identity.
    return agent.name
  }

  async resolveModel(_agentName: string, _app?: string): Promise<string> {
    return this.resolveDefaultModel()
  }

  async resolveDefaultModel(): Promise<string> {
    try {
      const c = (await api.kirocrewConfig()) as KirocrewAgentConfig
      const m = c?.agent?.model || ''
      return m === 'auto' ? '' : m
    } catch {
      return ''
    }
  }

  async resolveDefaultEffort(readConfig: () => Promise<unknown>): Promise<string> {
    try {
      const c = (await readConfig()) as KirocrewAgentConfig
      return c?.agent?.reasoning_effort || ''
    } catch {
      return ''
    }
  }

  async fetchUsage(): Promise<NormalizedUsage> {
    // Per-turn USD cost rides the turn-stats surface for this provider;
    // there is no plan/credit aggregate to report, so the sessions half is
    // an honest zero shape rather than a fabricated total.
    const period = { sessions: 0, messages: 0, toolCalls: 0 }
    return {
      sessions: {
        total: 0,
        today: period,
        thisWeek: period,
        thisMonth: period,
        avgMsgsPerSession: 0,
        refusedTranscripts: 0,
        unreadableSessionDirs: 0,
        dailyHistory: [],
      },
      billing: null,
    }
  }

  async fetchProviderHooks(): Promise<Record<string, NormalizedProviderHook[]>> {
    return {}
  }

  async listPlugins(): Promise<NormalizedPlugin[]> {
    return []
  }

  async installPlugin(_pkg: string, _type: 'agent' | 'skill' | 'mcp') {
    return { ok: false as const, error: 'plugins are not served by the Meta Model API provider' }
  }

  async uninstallPlugin(_pkg: string, _type: 'agent' | 'skill' | 'mcp') {
    return { ok: false as const, error: 'plugins are not served by the Meta Model API provider' }
  }

  async updatePlugins(_type: 'agent' | 'skill' | 'mcp') {
    return { ok: false as const, error: 'plugins are not served by the Meta Model API provider' }
  }

  async fetchAvailableModels(_backend?: string): Promise<ModelInfo[]> {
    return [...MUSE_MODELS]
  }

  getContextWindow(model: string): number {
    return MUSE_WINDOWS[model] ?? DEFAULT_CONTEXT
  }

  getDefaultModel(): string {
    return 'auto'
  }

  getPermissionModes(): PermissionMode[] {
    // The loop runs one approval policy (on-request); Crew's grant
    // machinery (always-allow, withdraw) layers on top of it unchanged.
    return [
      { id: 'normal', label: 'Normal', description: 'Ask before each tool execution' },
    ]
  }
}
