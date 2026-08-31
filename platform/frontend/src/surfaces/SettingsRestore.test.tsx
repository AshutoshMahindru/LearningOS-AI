import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { setAuthToken } from '../api/client';
import { jsonResponse } from '../test/http';
import { SettingsSurface } from './SettingsSurface';

describe('Settings restore', () => {
  beforeEach(() => {
    setAuthToken('loopback-token');
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.includes('/system/health')) {
          return jsonResponse({
            status: 'HEALTHY',
            version: '3.0.0',
            worker_alive: true,
            database_path: '/tmp/live-home/learningos.db',
          });
        }
        if (url.includes('/system/version')) {
          return jsonResponse({ version: '3.0.0' });
        }
        if (url.includes('/system/config')) {
          return jsonResponse({
            data_home: '/tmp/live-home',
            database_path: '/tmp/live-home/learningos.db',
            worker_socket: '/tmp/live-home/run/worker.sock',
            bind_host: '127.0.0.1',
            api_prefix: '/api/v1',
          });
        }
        if (url.includes('/curriculum/packages') && (!init || init.method !== 'POST')) {
          return jsonResponse({ packages: [] });
        }
        if (url.includes('/system/restore')) {
          return jsonResponse({
            status: 'RESTORED',
            path: '/tmp/backup.tar.gz',
            dest_home: '/tmp/clean-home',
          });
        }
        return jsonResponse({ error: { code: 'NOT_FOUND', message: 'missing', details: {} } }, 404);
      }),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    setAuthToken(null);
  });

  it('requires dest_home and includes it in the restore request', async () => {
    const user = userEvent.setup();
    render(<SettingsSurface />);

    const restoreButton = await screen.findByRole('button', { name: 'Restore' });
    expect(restoreButton).toBeDisabled();

    await user.type(screen.getByLabelText('Backup id or path'), '/tmp/backup.tar.gz');
    expect(restoreButton).toBeDisabled();

    await user.type(screen.getByLabelText(/destination home/i), '/tmp/clean-home');
    expect(restoreButton).toBeEnabled();

    await user.click(restoreButton);

    await waitFor(() => {
      const fetchMock = globalThis.fetch as unknown as ReturnType<typeof vi.fn>;
      const restoreCall = fetchMock.mock.calls.find(([input]) => String(input).includes('/system/restore'));
      expect(restoreCall).toBeDefined();
      const init = restoreCall?.[1] as RequestInit | undefined;
      expect(JSON.parse(String(init?.body))).toEqual({
        path: '/tmp/backup.tar.gz',
        dest_home: '/tmp/clean-home',
      });
    });

    expect(await screen.findByText(/restore requested into \/tmp\/clean-home/i)).toBeInTheDocument();
  });
});
