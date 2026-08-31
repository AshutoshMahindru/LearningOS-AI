import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { errorEnvelope, jsonResponse } from '../test/http';
import {
  ApiError,
  getHealth,
  getAuthToken,
  listMissions,
  mapErrorResponse,
  restoreBackup,
  setAuthToken,
} from './client';

describe('API client', () => {
  beforeEach(() => {
    setAuthToken(null);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    setAuthToken(null);
  });

  it('maps a backend error envelope to code, message, and details', async () => {
    const response = errorEnvelope('NOT_FOUND', 'Mission missing', { id: 'g3.fixture.orientation' }, 404);
    const mapped = await mapErrorResponse(response);
    expect(mapped).toBeInstanceOf(ApiError);
    expect(mapped).toMatchObject({
      code: 'NOT_FOUND',
      message: 'Mission missing',
      details: { id: 'g3.fixture.orientation' },
      status: 404,
    });
  });

  it('throws the mapped envelope for non-2xx responses', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => errorEnvelope('WORKER_UNAVAILABLE', 'Worker socket missing', { path: '/tmp/x' }, 503)),
    );

    await expect(getHealth()).rejects.toMatchObject({
      code: 'WORKER_UNAVAILABLE',
      message: 'Worker socket missing',
      details: { path: '/tmp/x' },
    });
  });

  it('does not send Authorization for health', async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({
        status: 'HEALTHY',
        version: '3.0.0',
        worker_alive: false,
        database_path: '/tmp/learningos.db',
      }),
    );
    vi.stubGlobal('fetch', fetchMock);
    setAuthToken('loopback-token');

    await getHealth();

    const init = fetchMock.mock.calls[0]?.[1] as RequestInit | undefined;
    const headers = new Headers(init?.headers);
    expect(headers.get('Authorization')).toBeNull();
    expect(getAuthToken()).toBe('loopback-token');
  });

  it('sends Bearer token on authenticated routes after bootstrap', async () => {
    const fetchMock = vi.fn(async () => jsonResponse({ missions: [] }));
    vi.stubGlobal('fetch', fetchMock);
    setAuthToken('loopback-token');

    await listMissions();

    const url = fetchMock.mock.calls[0]?.[0];
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit | undefined;
    expect(String(url)).toContain('/api/v1/missions');
    const headers = new Headers(init?.headers);
    expect(headers.get('Authorization')).toBe('Bearer loopback-token');
  });

  it('sends dest_home on restore so a clean home can be the target', async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({ status: 'RESTORED', path: '/tmp/backup.tar.gz', dest_home: '/tmp/clean-home' }),
    );
    vi.stubGlobal('fetch', fetchMock);
    setAuthToken('loopback-token');

    await restoreBackup({
      path: '/tmp/backup.tar.gz',
      dest_home: '/tmp/clean-home',
    });

    const url = String(fetchMock.mock.calls[0]?.[0]);
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit | undefined;
    expect(url).toContain('/api/v1/system/restore');
    expect(JSON.parse(String(init?.body))).toEqual({
      path: '/tmp/backup.tar.gz',
      dest_home: '/tmp/clean-home',
    });
  });
});
