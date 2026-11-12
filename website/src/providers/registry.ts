import type { ProviderAdapter, ProviderId } from './types'
import { AcpAdapter } from './adapters/acp'
import { MetaApiAdapter } from './adapters/meta_api'
import { BedrockAdapter } from './adapters/bedrock'

// The adapter registry: 'acp' (kiro-cli over ACP) is the default provider;
// 'meta_api' and 'bedrock' are this fork's native providers
// (agent.provider enum is ["acp", "meta_api", "bedrock"]). Kept as a thin
// indirection so the 15+ pages that consume useProvider()/the adapter
// interface for labels/capabilities/model-windows stay unchanged.
const ADAPTERS: Record<ProviderId, ProviderAdapter> = {
  acp: new AcpAdapter(),
  meta_api: new MetaApiAdapter(),
  bedrock: new BedrockAdapter(),
}

export function getAdapter(id?: ProviderId): ProviderAdapter {
  // Honour the argument: a caller asking for a provider gets THAT
  // provider's adapter. (This previously ignored the parameter and always
  // returned the ACP adapter, which was invisible while 'acp' was the only
  // member of the union.)
  return id ? ADAPTERS[id] : ADAPTERS.acp
}
