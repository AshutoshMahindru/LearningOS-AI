import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { AppRoutes } from './App';
import { setAuthToken } from './api/client';
import { EmptyState } from './components/EmptyState';
import { StatusBanner } from './components/StatusBanner';
import { AuthProvider } from './context/AuthContext';
import { errorEnvelope, jsonResponse } from './test/http';

describe('empty and error states', () => {
  it('renders a reusable empty state', () => {
    render(<EmptyState title="Nothing here" message="The catalog has no missions." />);
    expect(screen.getByTestId('empty-state')).toHaveTextContent('Nothing here');
    expect(screen.getByText('The catalog has no missions.')).toBeInTheDocument();
  });

  it('renders an error banner as an alert', () => {
    render(
      <StatusBanner tone="error" title="Unable to load catalog">
        Worker socket missing
      </StatusBanner>,
    );
    expect(screen.getByRole('alert')).toHaveTextContent('Unable to load catalog');
    expect(screen.getByRole('alert')).toHaveTextContent('Worker socket missing');
  });

  it('shows the dashboard empty state when the catalog is empty', async () => {
    setAuthToken(null);
    sessionStorage.setItem(
      'learningos.learner',
      JSON.stringify({ id: 'learner-1', username: 'ada' }),
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
        if (url.includes('/missions')) {
          return jsonResponse({ missions: [] });
        }
        return jsonResponse({ error: { code: 'NOT_FOUND', message: 'missing', details: {} } }, 404);
      }),
    );

    render(
      <AuthProvider>
        <MemoryRouter>
          <AppRoutes />
        </MemoryRouter>
      </AuthProvider>,
    );

    await waitFor(() => {
      expect(screen.getByTestId('empty-state')).toHaveTextContent('No missions loaded');
    });
  });

  it('shows the dashboard error state when the catalog request fails', async () => {
    setAuthToken(null);
    sessionStorage.setItem(
      'learningos.learner',
      JSON.stringify({ id: 'learner-1', username: 'ada' }),
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
        if (url.includes('/missions')) {
          return errorEnvelope('INTERNAL', 'Catalog unavailable', {}, 500);
        }
        return jsonResponse({ error: { code: 'NOT_FOUND', message: 'missing', details: {} } }, 404);
      }),
    );

    render(
      <AuthProvider>
        <MemoryRouter>
          <AppRoutes />
        </MemoryRouter>
      </AuthProvider>,
    );

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent('Unable to load catalog');
      expect(screen.getByRole('alert')).toHaveTextContent('Catalog unavailable');
    });
  });
});

describe('bootstrap failure', () => {
  beforeEach(() => {
    setAuthToken(null);
  });

  it('surfaces a local API error instead of a fake local identity', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => errorEnvelope('INTERNAL', 'Loopback bootstrap failed', {}, 500)),
    );

    render(
      <AuthProvider>
        <MemoryRouter>
          <AppRoutes />
        </MemoryRouter>
      </AuthProvider>,
    );

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent('Local API unavailable');
      expect(screen.getByRole('alert')).toHaveTextContent('Loopback bootstrap failed');
    });
    expect(screen.queryByText(/learner_default/i)).not.toBeInTheDocument();
  });
});
