import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { AppRoutes } from './App';
import { setAuthToken } from './api/client';
import { AuthProvider } from './context/AuthContext';
import { jsonResponse } from './test/http';

function installApiMock() {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/auth/bootstrap')) {
        return jsonResponse({ token: 'loopback-token', token_type: 'bearer' });
      }
      if (url.includes('/learners/')) {
        return jsonResponse({ id: 'learner-1', username: 'ada', display_name: 'Ada' });
      }
      if (url.endsWith('/missions') || url.includes('/missions?')) {
        return jsonResponse({ missions: [] });
      }
      if (url.includes('/system/health')) {
        return jsonResponse({
          status: 'HEALTHY',
          version: '3.0.0',
          worker_alive: false,
          database_path: '/tmp/learningos.db',
        });
      }
      return jsonResponse({ error: { code: 'NOT_FOUND', message: 'missing', details: {} } }, 404);
    }),
  );
}

describe('application shell', () => {
  beforeEach(() => {
    setAuthToken(null);
    sessionStorage.setItem(
      'learningos.learner',
      JSON.stringify({ id: 'learner-1', username: 'ada', display_name: 'Ada' }),
    );
    installApiMock();
  });

  it('renders skip link, header, nav, and main landmarks', async () => {
    render(
      <AuthProvider>
        <MemoryRouter>
          <AppRoutes />
        </MemoryRouter>
      </AuthProvider>,
    );

    expect(screen.getByRole('link', { name: /skip to main content/i })).toHaveAttribute(
      'href',
      '#main-content',
    );

    await waitFor(() => {
      expect(screen.getByRole('banner')).toBeInTheDocument();
    });

    expect(screen.getByRole('navigation', { name: 'Primary' })).toBeInTheDocument();
    expect(screen.getByRole('main')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Dashboard' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Player' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Artifacts' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Settings' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Diagnostics' })).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: /demo/i })).not.toBeInTheDocument();
    expect(document.body.innerHTML).not.toContain('/missions/demo');
  });
});
