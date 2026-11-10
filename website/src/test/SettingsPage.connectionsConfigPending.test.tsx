/**
 * Settings → Connections waits for the config before choosing its layout.
 *
 * `connections_ui` decides between the OAuth-apps-only forms (off) and the
 * gallery (on), and an unread config reads as off. Mounting the page then
 * would show the off forms first and swap them for the gallery when the read
 * lands, dropping a Client ID or secret typed into them in between.
 */
import { describe, it, expect, vi } from 'vitest'
import { render, screen, waitFor, act } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

let resolveConfig: (value: unknown) => void = () => {}
vi.mock('../api/client', async importOriginal => {
  const mod = await importOriginal<typeof import('../api/client')>()
  return {
    ...mod,
    api: {
      ...mod.api,
      kirocrewConfig: vi.fn(() => new Promise(resolve => { resolveConfig = resolve })),
    },
  }
})

// The page itself is its own unit; this probe records which layout it got.
vi.mock('../pages/connections/ConnectionsPage', () => ({
  default: ({ servicesEnabled }: { servicesEnabled: boolean }) => (
    <div data-testid="connections-page">{servicesEnabled ? 'gallery' : 'oauth-only'}</div>
  ),
}))

vi.mock('../store', () => ({ useAppSelector: () => '1.0.0' }))
vi.mock('../hooks/useIsMobile', () => ({ useIsMobile: () => false }))

import SettingsPage from '../pages/SettingsPage'

describe('Settings → Connections while the config is still loading', () => {
  it('mounts the page only once the config is read, so the off layout never shows first', async () => {
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={qc}>
        <MemoryRouter initialEntries={['/settings/connections']}>
          <Routes>
            <Route path="/settings/*" element={<SettingsPage />} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>,
    )
    // Let the lazy tab chunk load while the config read is held open.
    await act(async () => { await new Promise(r => setTimeout(r, 50)) })
    expect(screen.queryByTestId('connections-page')).toBeNull()
    await act(async () => { resolveConfig({}) })
    await waitFor(() => expect(screen.getByTestId('connections-page')).toHaveTextContent('gallery'), { timeout: 10_000 })
  })
})
