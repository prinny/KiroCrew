import type { ProviderAdapter, ProviderId } from './types'
import { AcpAdapter } from './adapters/acp'
import { MetaApiAdapter } from './adapters/meta_api'

// The adapter registry: 'acp' (kiro-cli over ACP) is the default provider;
// 'meta_api' is this fork's native Meta Model API provider
// (agent.provider enum is ["acp", "meta_api"]). Kept as a thin indirection
// so the 15+ pages that consume useProvider()/the adapter interface for
// labels/capabilities/model-windows stay unchanged.
const ADAPTERS: Record<ProviderId, ProviderAdapter> = {
  acp: new AcpAdapter(),
  meta_api: new MetaApiAdapter(),
}

export function getAdapter(id?: ProviderId): ProviderAdapter {
  // Honour the argument: a caller asking for a provider gets THAT
  // provider's adapter. (This previously ignored the parameter and always
  // returned the ACP adapter, which was invisible while 'acp' was the only
  // member of the union.)
  return id ? ADAPTERS[id] : ADAPTERS.acp
}
