import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { setAuthToken } from '../api/client';
import { errorEnvelope, jsonResponse } from '../test/http';
import { TutorSurface } from './TutorSurface';

const SESSION = {
  session_id: 'session-1',
  mission_id: 'catalog.generic',
  learner_id: 'learner-1',
  current_stage_id: 'stage_orientation',
};

const MISSION = {
  id: 'catalog.generic',
  title: 'Catalog fixture',
  stages: [
    {
      id: 'stage_orientation',
      title: 'Fixture orientation',
      type: 'orientation',
      assistance_policy: 'UNRESTRICTED',
    },
    {
      id: 'stage_transfer',
      title: 'Fixture transfer',
      type: 'transfer_assessment',
      assistance_policy: 'NO_AI_REQUIRED',
    },
  ],
};

function installFetch(handler: (url: string, init?: RequestInit) => Response | Promise<Response>) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    return handler(String(input), init);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

describe('Tutor surface', () => {
  beforeEach(() => {
    setAuthToken('loopback-token');
    window.history.pushState({}, '', '/tutor');
  });

  afterEach(() => {
    setAuthToken(null);
    window.history.pushState({}, '', '/tutor');
  });

  it('does not call tutor chat when the loaded stage is no-AI locked', async () => {
    window.history.pushState({}, '', '/tutor?session=session-1&stage=stage_transfer');
    const fetchMock = installFetch((url) => {
      if (url.includes('/sessions/session-1') && !url.includes('/stages/')) {
        return jsonResponse({ ...SESSION, current_stage_id: 'stage_transfer' });
      }
      if (url.includes('/missions/catalog.generic')) {
        return jsonResponse(MISSION);
      }
      if (url.includes('/tutor/chat')) {
        return jsonResponse({ role: 'SOCRATIC', reply: 'should not run', provider: 'heuristic' });
      }
      return jsonResponse({ error: { code: 'NOT_FOUND', message: 'missing', details: {} } }, 404);
    });

    render(<TutorSurface />);

    await userEvent.click(screen.getByRole('button', { name: 'Load session' }));
    expect(await screen.findByText(/No-AI lock/i)).toBeInTheDocument();
    expect(screen.getByTestId('tutor-stage-policy')).toHaveTextContent('NO_AI_REQUIRED');
    expect(screen.getByRole('button', { name: 'Ask tutor' })).toBeDisabled();
    expect(screen.getByLabelText('Prompt')).toBeDisabled();
    expect(fetchMock.mock.calls.some(([input]) => String(input).includes('/tutor/chat'))).toBe(false);
  });

  it('posts to tutor chat and shows a 501 provider-unavailable state', async () => {
    const user = userEvent.setup();
    const fetchMock = installFetch((url, init) => {
      if (url.includes('/sessions/session-1') && !url.includes('/stages/')) {
        return jsonResponse(SESSION);
      }
      if (url.includes('/missions/catalog.generic')) {
        return jsonResponse(MISSION);
      }
      if (url.includes('/tutor/chat') && (init?.method ?? 'GET').toUpperCase() === 'POST') {
        return errorEnvelope('TUTOR_NOT_AVAILABLE', 'No tutor provider is configured', {}, 501);
      }
      return jsonResponse({ error: { code: 'NOT_FOUND', message: 'missing', details: {} } }, 404);
    });

    render(<TutorSurface />);

    await user.type(screen.getByLabelText('Session id'), 'session-1');
    await user.click(screen.getByRole('button', { name: 'Load session' }));
    expect(await screen.findByTestId('tutor-stage-policy')).toHaveTextContent('UNRESTRICTED');

    await user.type(screen.getByLabelText('Prompt'), 'What is the next action?');
    await user.click(screen.getByRole('button', { name: 'Ask tutor' }));

    expect(await screen.findByText('Tutor is not available')).toBeInTheDocument();
    const chatCall = fetchMock.mock.calls.find(([input]) => String(input).includes('/tutor/chat'));
    expect(chatCall).toBeDefined();
    const headers = new Headers(chatCall?.[1]?.headers);
    expect(headers.get('Authorization')).toBe('Bearer loopback-token');
    expect(JSON.parse(String(chatCall?.[1]?.body))).toMatchObject({
      session_id: 'session-1',
      stage_id: 'stage_orientation',
      role: 'SOCRATIC',
      prompt: 'What is the next action?',
    });
  });

  it('renders heuristic guidance without storing provider credentials', async () => {
    const user = userEvent.setup();
    installFetch((url, init) => {
      if (url.includes('/sessions/session-1') && !url.includes('/stages/')) {
        return jsonResponse(SESSION);
      }
      if (url.includes('/missions/catalog.generic')) {
        return jsonResponse(MISSION);
      }
      if (url.includes('/tutor/chat') && (init?.method ?? 'GET').toUpperCase() === 'POST') {
        return jsonResponse({
          role: 'DEBUGGER',
          reply: 'DEBUGGER guidance: reproduce the symptom before changing code.',
          provider: 'heuristic',
          assistance_policy: 'UNRESTRICTED',
        });
      }
      return jsonResponse({ error: { code: 'NOT_FOUND', message: 'missing', details: {} } }, 404);
    });

    render(<TutorSurface />);

    await user.type(screen.getByLabelText('Session id'), 'session-1');
    await user.click(screen.getByRole('button', { name: 'Load session' }));
    await user.selectOptions(screen.getByLabelText('Tutor role'), 'DEBUGGER');
    await user.type(screen.getByLabelText('Prompt'), 'The run failed.');
    await user.click(screen.getByRole('button', { name: 'Ask tutor' }));

    await waitFor(() => {
      expect(screen.getByTestId('tutor-transcript')).toHaveTextContent('The run failed.');
    });
    expect(screen.getByTestId('tutor-transcript')).toHaveTextContent('reproduce the symptom');
    expect(screen.getByTestId('tutor-transcript')).toHaveTextContent('DEBUGGER');
  });
});
