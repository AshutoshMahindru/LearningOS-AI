import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { errorEnvelope, jsonResponse } from '../test/http';
import {
  ApiError,
  bootstrap,
  enterStage,
  evaluateGates,
  executeStage,
  getHealth,
  getAuthToken,
  getMission,
  listMissions,
  mapErrorResponse,
  predictStage,
  restoreBackup,
  setAuthToken,
  submitStage,
} from './client';
import { getDiagnostics } from './diagnostics';

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

  it('parses stages from a mission specification payload', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        jsonResponse({
          id: 'catalog.generic',
          title: 'Catalog fixture',
          stages: [
            {
              id: 'stage_orientation',
              title: 'Fixture orientation',
              type: 'orientation',
              assistance_policy: 'UNRESTRICTED',
              instructions: 'Read the invariant.',
            },
          ],
        }),
      ),
    );
    setAuthToken('loopback-token');

    const mission = await getMission('catalog.generic');
    expect(mission.stages).toEqual([
      {
        id: 'stage_orientation',
        title: 'Fixture orientation',
        type: 'orientation',
        assistance_policy: 'UNRESTRICTED',
        instructions: 'Read the invariant.',
        runner: undefined,
        validation_rubric: undefined,
      },
    ]);
  });

  it('posts typed G4 stage lifecycle endpoints', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/enter')) {
        return jsonResponse({ status: 'ENTERED', stage_id: 'stage_experiment' });
      }
      if (url.endsWith('/predict')) {
        return jsonResponse({ status: 'SEALED', prediction_hash: 'pred-1' });
      }
      if (url.endsWith('/execute')) {
        return jsonResponse({ status: 'SUCCESS', execution_id: 'exec-1', blocks: [] });
      }
      if (url.endsWith('/submit')) {
        return jsonResponse({ status: 'PASSED', next_stage_id: 'stage_gate' });
      }
      if (url.endsWith('/gates/evaluate')) {
        return jsonResponse({ status: 'PASSED', score: 1, passed: true });
      }
      return errorEnvelope('NOT_FOUND', 'missing', {}, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    setAuthToken('loopback-token');

    await enterStage('session-1', 'stage_experiment');
    await predictStage('session-1', 'stage_experiment', {
      hypothesis: 'latency drops',
      expected_values: { p95_ms: 20 },
    });
    await executeStage('session-1', 'stage_experiment', { parameters: { batch: 2 } });
    await submitStage('session-1', 'stage_experiment', { explanation: 'matched' });
    await evaluateGates('session-1');

    const urls = fetchMock.mock.calls.map(([input]) => String(input));
    expect(urls.some((url) => url.endsWith('/sessions/session-1/stages/stage_experiment/enter'))).toBe(true);
    expect(urls.some((url) => url.endsWith('/sessions/session-1/stages/stage_experiment/predict'))).toBe(true);
    expect(urls.some((url) => url.endsWith('/sessions/session-1/stages/stage_experiment/execute'))).toBe(true);
    expect(urls.some((url) => url.endsWith('/sessions/session-1/stages/stage_experiment/submit'))).toBe(true);
    expect(urls.some((url) => url.endsWith('/sessions/session-1/gates/evaluate'))).toBe(true);

    const predictInit = fetchMock.mock.calls.find(([input]) =>
      String(input).endsWith('/predict'),
    )?.[1] as RequestInit | undefined;
    expect(JSON.parse(String(predictInit?.body))).toEqual({
      hypothesis: 'latency drops',
      expected_values: { p95_ms: 20 },
    });
  });

  it('maps stage errors from {error:{code,message,details}} and redacts tokens in diagnostics', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/auth/bootstrap')) {
        return jsonResponse({ token: 'loopback-token', token_type: 'bearer' });
      }
      return errorEnvelope('VALIDATION_ERROR', 'Hypothesis required', { field: 'hypothesis' }, 422);
    });
    vi.stubGlobal('fetch', fetchMock);

    const boot = await bootstrap();
    expect(boot.token).toBe('loopback-token');
    setAuthToken(boot.token);

    await expect(predictStage('session-1', 'stage_experiment', { hypothesis: '', expected_values: {} })).rejects.toMatchObject({
      code: 'VALIDATION_ERROR',
      message: 'Hypothesis required',
      details: { field: 'hypothesis' },
    });

    const serialized = JSON.stringify(getDiagnostics());
    expect(serialized).toContain('/api/v1/auth/bootstrap');
    expect(serialized).toContain('[redacted]');
    expect(serialized).not.toContain('loopback-token');
    expect(serialized).not.toContain('Bearer');
  });
});
