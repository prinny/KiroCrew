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

// The native Bedrock provider's catalogue is DYNAMIC — discovered live
// from the operator's AWS account (ListFoundationModels joined with
// ListInferenceProfiles) — so this adapter carries NO baked-in model
// list. Rows come from the same /api/models surface every provider
// reads; a row's reported window comes from the provider's capability
// overlay, and a row with no reported window renders none rather than
// a guessed one. Baking a Bedrock model list in here would freeze the
// catalogue at bundle time and lie about accounts whose discovery
// differs.
const DEFAULT_CONTEXT = 200_000

interface RawModel {
  model_name: string
  display_name?: string
  description?: string
  context_window?: number
}

interface KirocrewAgentConfig {
  agent?: { model?: string; reasoning_effort?: string }
}

/**
 * Adapter for the fork's native Amazon Bedrock provider
 * (`agent.provider = "bedrock"`). A deliberately REDUCED subset of the ACP
 * adapter, matching the provider's declared capabilities: no hooks, no
 * plugin registry, no agent templates, no subagents, no reasoning effort
 * — the surfaces that remain (models, context windows, per-turn USD
 * usage, session resume, compaction) are the ones the provider serves.
 */
export class BedrockAdapter implements ProviderAdapter {
  readonly id = 'bedrock' as const
  readonly displayName = 'Amazon Bedrock'

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
    reasoningEffort: false,
  }

  readonly labels: ProviderLabels = {
    sessionProcess: 'In-process (gateway)',
    processCountLabel: 'in_process',
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

  async resolveDefaultEffort(_readConfig: () => Promise<unknown>): Promise<string> {
    // Effort is not served by this provider.
    return ''
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
    return { ok: false as const, error: 'plugins are not served by the Bedrock provider' }
  }

  async uninstallPlugin(_pkg: string, _type: 'agent' | 'skill' | 'mcp') {
    return { ok: false as const, error: 'plugins are not served by the Bedrock provider' }
  }

  async updatePlugins(_type: 'agent' | 'skill' | 'mcp') {
    return { ok: false as const, error: 'plugins are not served by the Bedrock provider' }
  }

  async fetchAvailableModels(_backend?: string): Promise<ModelInfo[]> {
    try {
      const models = (await api.models(undefined)) as RawModel[]
      if (!Array.isArray(models)) return []
      return models
        .filter(m => m && typeof m.model_name === 'string')
        .map(m => ({
          name: m.model_name,
          displayName:
            m.display_name && m.display_name !== m.model_name ? m.display_name : undefined,
          description: m.description || '',
          contextWindow: m.context_window || undefined,
        }))
    } catch {
      return []
    }
  }

  getContextWindow(_model: string): number {
    // Windows come from the live rows; without one the surface default
    // applies — never a per-model guess baked into the bundle.
    return DEFAULT_CONTEXT
  }

  getDefaultModel(): string {
    return 'auto'
  }

  getPermissionModes(): PermissionMode[] {
    // Tool calls are gated by the provider's approval registry; Crew's
    // grant machinery layers on top of it unchanged.
    return [
      { id: 'normal', label: 'Normal', description: 'Ask before each tool execution' },
    ]
  }
}
