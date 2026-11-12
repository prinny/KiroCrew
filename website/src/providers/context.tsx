import { createContext, useContext, type ReactNode } from 'react'
import { getAdapter } from './registry'
import type { ProviderAdapter } from './types'

// The context serves the DEFAULT provider's adapter (ACP). The registry
// also carries the fork's native Meta Model API adapter, which call sites
// that know a session's provider reach via getAdapter('meta_api'); making
// this context itself provider-switchable is follow-up work. The context
// is retained (rather than inlining the adapter at each call site) so the
// many useProvider() consumers stay unchanged.
const acpAdapter = getAdapter()

const ProviderContext = createContext<ProviderAdapter>(acpAdapter)

export function ProviderProvider({ children }: { children: ReactNode }) {
  return <ProviderContext.Provider value={acpAdapter}>{children}</ProviderContext.Provider>
}

export function useProvider(): ProviderAdapter {
  return useContext(ProviderContext)
}
