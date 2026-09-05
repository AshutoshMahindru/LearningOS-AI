import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { AppRoutes } from '../App';
import { setAuthToken } from '../api/client';
import { AuthProvider } from '../context/AuthContext';
import { jsonResponse } from '../test/http';

const MISSIONS = [
  {
    id: 'catalog.generic',
    title: 'Catalog fixture',
    description: 'Map the whole system before descending.',
  },
  {
    id: 'catalog.second',
    title: 'Second fixture',
    description: 'Another generic package.',
  },
];

const SESSION = {
  session_id: 'session-1',
  mission_id: 'catalog.generic',
  learner_id: 'learner-1',
  current_stage_id: 'stage_orientation',
};

const MISSION_SPEC = {
  id: 'catalog.generic',
  title: 'Catalog fixture',
  stages: [
    {
      id: 'stage_orientation',
      title: 'Fixture orientation',
      type: 'orientation',
      assistance_policy: 'UNRESTRICTED',
    },
  ],
};

function installCatalogMock() {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const method = (init?.method ?? 'GET').toUpperCase();
    if (url.includes('/auth/bootstrap')) {
      return jsonResponse({ token: 'loopback-token', token_type: 'bearer' });
    }
    if (url.includes('/learners/')) {
      return jsonResponse({ id: 'learner-1', username: 'ada', display_name: 'Ada' });
    }
    if (url.includes('/missions/catalog.generic')) {
      return jsonResponse(MISSION_SPEC);
    }
    if (url.endsWith('/missions') || url.includes('/missions?')) {
      return jsonResponse({ missions: MISSIONS });
    }
    if (url.includes('/sessions') && method === 'POST') {
      return jsonResponse(SESSION);
    }
    if (url.includes('/sessions/session-1') && method === 'GET') {
      return jsonResponse(SESSION);
    }
    if (url.includes('/stages/') && url.endsWith('/enter')) {
      return jsonResponse({ status: 'ENTERED' });
    }
    return jsonResponse({ error: { code: 'NOT_FOUND', message: 'missing', details: {} } }, 404);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

function renderCatalog() {
  return render(
    <AuthProvider>
      <MemoryRouter>
        <AppRoutes />
      </MemoryRouter>
    </AuthProvider>,
  );
}

describe('catalog accessibility', () => {
  beforeEach(() => {
    setAuthToken(null);
    sessionStorage.setItem(
      'learningos.learner',
      JSON.stringify({ id: 'learner-1', username: 'ada', display_name: 'Ada' }),
    );
  });

  it('exposes unique labelled start controls for each mission', async () => {
    installCatalogMock();
    renderCatalog();

    expect(await screen.findByRole('heading', { name: 'Catalog' })).toBeInTheDocument();
    expect(await screen.findByRole('heading', { name: 'Catalog fixture' })).toBeInTheDocument();
    const first = screen.getByRole('button', { name: 'Start session: Catalog fixture' });
    const second = screen.getByRole('button', { name: 'Start session: Second fixture' });
    expect(first).toBeEnabled();
    expect(second).toBeEnabled();
    expect(screen.getByRole('list', { name: 'Catalog' })).toBeInTheDocument();
  });

  it('starts a session from the keyboard', async () => {
    const user = userEvent.setup();
    const fetchMock = installCatalogMock();
    renderCatalog();

    const start = await screen.findByRole('button', { name: 'Start session: Catalog fixture' });
    start.focus();
    expect(start).toHaveFocus();
    await user.keyboard('{Enter}');

    await waitFor(() => {
      const call = fetchMock.mock.calls.find(([input, init]) => {
        return String(input).includes('/sessions') && (init as RequestInit | undefined)?.method === 'POST';
      });
      expect(call).toBeDefined();
      const init = call?.[1] as RequestInit | undefined;
      expect(JSON.parse(String(init?.body))).toMatchObject({
        mission_id: 'catalog.generic',
        learner_id: 'learner-1',
      });
    });
  });

  it('sets a catalog document title in the application shell', async () => {
    installCatalogMock();
    renderCatalog();

    expect(await screen.findByRole('heading', { name: 'Catalog' })).toBeInTheDocument();
    expect(document.title).toMatch(/catalog/i);
    expect(screen.getByRole('main', { name: 'Main content' })).toBeInTheDocument();
  });
});
