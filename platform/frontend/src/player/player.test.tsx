import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { AppRoutes } from '../App';
import { setAuthToken } from '../api/client';
import { AuthProvider } from '../context/AuthContext';
import { jsonResponse } from '../test/http';
import { CANONICAL_STAGE_TYPES } from '../api/types';
import { FallbackStage } from './stages';
import { STAGE_REGISTRY, resolveStageRenderer } from './stageRegistry';

const GENERIC_SPEC = {
  id: 'catalog.generic',
  title: 'Catalog fixture',
  core_invariant: 'Map the whole system before descending.',
  competencies: ['comp.generic.trace'],
  stages: [
    {
      id: 'stage_orientation',
      title: 'Fixture orientation',
      type: 'orientation',
      assistance_policy: 'UNRESTRICTED',
      instructions: 'Read the mission invariant before tracing.',
    },
    {
      id: 'stage_experiment',
      title: 'Fixture experiment',
      type: 'experiment',
      assistance_policy: 'RESTRICTED_HINTS',
      instructions: 'Predict before you run.',
    },
  ],
};

function installPlayerMock(currentStageId: string) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = (init?.method ?? 'GET').toUpperCase();
      if (url.includes('/auth/bootstrap')) {
        return jsonResponse({ token: 'loopback-token', token_type: 'bearer' });
      }
      if (url.includes('/learners/')) {
        return jsonResponse({ id: 'learner-1', username: 'ada', display_name: 'Ada' });
      }
      if (url.includes('/missions/catalog.generic')) {
        return jsonResponse(GENERIC_SPEC);
      }
      if (url.includes('/sessions/session-restore-1') && !url.includes('/stages/') && method === 'GET') {
        return jsonResponse({
          session_id: 'session-restore-1',
          mission_id: 'catalog.generic',
          learner_id: 'learner-1',
          current_stage_id: currentStageId,
        });
      }
      if (url.includes('/stages/') && url.endsWith('/enter')) {
        return jsonResponse({ status: 'ENTERED' });
      }
      if (url.includes('/stages/') && url.endsWith('/predict')) {
        return jsonResponse({ status: 'SEALED', prediction_hash: 'pred-1' });
      }
      if (url.includes('/stages/') && url.endsWith('/execute')) {
        return jsonResponse({
          status: 'SUCCESS',
          execution_id: 'exec-1',
          blocks: [{ type: 'metric', title: 'latency', payload: { p95_ms: 18 } }],
        });
      }
      if (url.includes('/stages/') && url.endsWith('/submit')) {
        return jsonResponse({ status: 'SUBMITTED' });
      }
      return jsonResponse({ error: { code: 'NOT_FOUND', message: 'missing', details: {} } }, 404);
    }),
  );
}

function renderRoute(path: string) {
  return render(
    <AuthProvider>
      <MemoryRouter initialEntries={[path]}>
        <AppRoutes />
      </MemoryRouter>
    </AuthProvider>,
  );
}

describe('generic mission player', () => {
  beforeEach(() => {
    setAuthToken(null);
    sessionStorage.setItem(
      'learningos.learner',
      JSON.stringify({ id: 'learner-1', username: 'ada', display_name: 'Ada' }),
    );
  });

  it('renders a stage from a fixture specification', async () => {
    installPlayerMock('stage_orientation');
    renderRoute('/sessions/session-restore-1');

    expect(await screen.findByTestId('mission-player')).toBeInTheDocument();
    expect(await screen.findByRole('heading', { name: 'Catalog fixture' })).toBeInTheDocument();
    expect(await screen.findByRole('heading', { name: 'Fixture orientation' })).toBeInTheDocument();
    expect(screen.getByText('Read the mission invariant before tracing.')).toBeInTheDocument();
    expect(screen.getByTestId('stage-frame')).toHaveAttribute('data-stage-type', 'orientation');
    expect(document.body.innerHTML).not.toContain('/missions/demo');
  });

  it('restores current_stage_id when opening /sessions/:id', async () => {
    installPlayerMock('stage_experiment');
    renderRoute('/sessions/session-restore-1');

    expect(await screen.findByLabelText('Hypothesis')).toBeInTheDocument();
    expect(screen.getByTestId('stage-frame')).toHaveAttribute('data-stage-type', 'experiment');
    expect(screen.getByText('Predict before you run.')).toBeInTheDocument();
  });

  it('restores current_stage_id when opening /player?session=', async () => {
    installPlayerMock('stage_experiment');
    renderRoute('/player?session=session-restore-1');

    expect(await screen.findByLabelText('Hypothesis')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Catalog fixture' })).toBeInTheDocument();
  });

  it('shows request diagnostics without leaking the auth token', async () => {
    installPlayerMock('stage_orientation');
    const user = userEvent.setup();
    renderRoute('/sessions/session-restore-1');
    expect(await screen.findByTestId('stage-frame')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Diagnostics' }));
    const drawer = await screen.findByTestId('diagnostics-drawer');
    expect(drawer).toHaveTextContent('/api/v1/sessions/session-restore-1');
    expect(drawer.textContent).not.toContain('loopback-token');
    expect(drawer.textContent).not.toMatch(/Bearer /);
  });
});

describe('stage registry', () => {
  it('maps every canonical stage type to a generic renderer', () => {
    expect(Object.keys(STAGE_REGISTRY).sort()).toEqual([...CANONICAL_STAGE_TYPES].sort());
    for (const type of CANONICAL_STAGE_TYPES) {
      expect(resolveStageRenderer(type)).toBe(STAGE_REGISTRY[type]);
    }
    expect(resolveStageRenderer('custom_extension')).toBe(FallbackStage);
  });
});

describe('player empty state', () => {
  beforeEach(() => {
    setAuthToken(null);
    sessionStorage.setItem(
      'learningos.learner',
      JSON.stringify({ id: 'learner-1', username: 'ada', display_name: 'Ada' }),
    );
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes('/auth/bootstrap')) {
          return jsonResponse({ token: 'loopback-token', token_type: 'bearer' });
        }
        if (url.includes('/learners/')) {
          return jsonResponse({ id: 'learner-1', username: 'ada' });
        }
        return jsonResponse({ error: { code: 'NOT_FOUND', message: 'missing', details: {} } }, 404);
      }),
    );
  });

  it('shows an empty state when /player has no session', async () => {
    renderRoute('/player');
    await waitFor(() => {
      expect(screen.getByTestId('empty-state')).toHaveTextContent('No session selected');
    });
  });
});
