import { API_PREFIX } from './types';
import type {
  ArtifactGetResponse,
  ArtifactPutRequest,
  ArtifactPutResponse,
  BackupResponse,
  BootstrapResponse,
  ConfigResponse,
  CurriculumPackage,
  CurriculumPackageListResponse,
  CurriculumPackageLoadRequest,
  HealthResponse,
  Learner,
  LearnerCreateRequest,
  MappedApiError,
  Mission,
  MissionListResponse,
  RestoreRequest,
  Session,
  SessionCreateRequest,
  VersionResponse,
} from './types';

export class ApiError extends Error implements MappedApiError {
  readonly code: string;
  readonly details: Record<string, unknown>;
  readonly status: number;

  constructor(payload: MappedApiError & { status: number }) {
    super(payload.message);
    this.name = 'ApiError';
    this.code = payload.code;
    this.details = payload.details;
    this.status = payload.status;
  }
}

export function isApiError(value: unknown): value is ApiError {
  return value instanceof ApiError;
}

type RequestOptions = RequestInit & {
  auth?: boolean;
};

let bearerToken: string | null = null;

export function setAuthToken(token: string | null): void {
  bearerToken = token;
}

export function getAuthToken(): string | null {
  return bearerToken;
}

function resolveUrl(path: string): string {
  const pathname = `${API_PREFIX}${path.startsWith('/') ? path : `/${path}`}`;
  const origin = globalThis.location?.origin;
  if (origin && origin !== 'null') {
    return new URL(pathname, origin).href;
  }
  return pathname;
}

function asRecord(value: unknown): Record<string, unknown> {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return {};
}

function asDetails(value: unknown): Record<string, unknown> {
  const record = asRecord(value);
  if (Object.keys(record).length > 0 || (value && typeof value === 'object' && !Array.isArray(value))) {
    return record;
  }
  if (value === undefined || value === null) {
    return {};
  }
  return { value };
}

export async function mapErrorResponse(response: Response): Promise<ApiError> {
  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    return new ApiError({
      code: 'INTERNAL',
      message: response.statusText || 'Request failed',
      details: {},
      status: response.status,
    });
  }

  const envelope = asRecord(payload);
  const nested = asRecord(envelope.error);
  if (typeof nested.code === 'string' || typeof nested.message === 'string') {
    return new ApiError({
      code: typeof nested.code === 'string' ? nested.code : 'INTERNAL',
      message: typeof nested.message === 'string' ? nested.message : 'Request failed',
      details: asDetails(nested.details),
      status: response.status,
    });
  }

  return new ApiError({
    code: 'INTERNAL',
    message: 'Request failed',
    details: asDetails(payload),
    status: response.status,
  });
}

async function request(path: string, options: RequestOptions = {}): Promise<Response> {
  const { auth = true, headers, ...rest } = options;
  const headerBag = new Headers(headers);
  if (auth) {
    if (!bearerToken) {
      throw new ApiError({
        code: 'UNAUTHORIZED',
        message: 'Not authenticated',
        details: {},
        status: 401,
      });
    }
    headerBag.set('Authorization', `Bearer ${bearerToken}`);
  }
  if (!headerBag.has('Accept')) {
    headerBag.set('Accept', 'application/json');
  }

  const response = await fetch(resolveUrl(path), {
    ...rest,
    headers: headerBag,
  });

  if (!response.ok) {
    throw await mapErrorResponse(response);
  }
  return response;
}

async function requestJson<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const response = await request(path, options);
  if (response.status === 204) {
    return undefined as T;
  }
  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError({
      code: 'INTERNAL',
      message: 'Invalid JSON response',
      details: { path },
      status: response.status,
    });
  }
}

function jsonBody(data: unknown): RequestOptions {
  return {
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  };
}

function asLearner(data: unknown): Learner {
  const record = asRecord(data);
  const id = String(record.id ?? record.learner_id ?? '');
  const username = String(record.username ?? '');
  if (!id) {
    throw new ApiError({
      code: 'INTERNAL',
      message: 'Learner response missing id',
      details: asDetails(data),
      status: 200,
    });
  }
  return {
    id,
    username,
    display_name: typeof record.display_name === 'string' ? record.display_name : null,
  };
}

function asSession(data: unknown): Session {
  const record = asRecord(data);
  const session_id = String(record.session_id ?? record.id ?? '');
  if (!session_id) {
    throw new ApiError({
      code: 'INTERNAL',
      message: 'Session response missing id',
      details: asDetails(data),
      status: 200,
    });
  }
  return {
    session_id,
    mission_id: String(record.mission_id ?? ''),
    learner_id: String(record.learner_id ?? ''),
    current_stage_id: typeof record.current_stage_id === 'string' ? record.current_stage_id : null,
    created_at: typeof record.created_at === 'string' ? record.created_at : null,
  };
}

function asMission(data: unknown): Mission {
  const record = asRecord(data);
  const id = String(record.id ?? record.mission_id ?? '');
  return {
    id,
    title: typeof record.title === 'string' ? record.title : undefined,
    description: typeof record.description === 'string' ? record.description : undefined,
    version: typeof record.version === 'string' ? record.version : undefined,
    phase_id: typeof record.phase_id === 'string' ? record.phase_id : undefined,
  };
}

export async function bootstrap(): Promise<BootstrapResponse> {
  const data = await requestJson<BootstrapResponse>('/auth/bootstrap', {
    method: 'POST',
    auth: false,
    ...jsonBody({}),
  });
  if (!data?.token) {
    throw new ApiError({
      code: 'INTERNAL',
      message: 'Bootstrap response missing token',
      details: asDetails(data),
      status: 200,
    });
  }
  return {
    token: data.token,
    token_type: data.token_type || 'bearer',
  };
}

export async function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>('/system/health', { auth: false });
}

export async function getVersion(): Promise<VersionResponse> {
  const data = await requestJson<Record<string, unknown>>('/system/version', { auth: false });
  const version = String(data.version ?? data.platform_version ?? '');
  return { version };
}

export async function getConfig(): Promise<ConfigResponse> {
  const data = await requestJson<Record<string, unknown>>('/system/config', { auth: false });
  return {
    data_home: String(data.data_home ?? ''),
    database_path: String(data.database_path ?? ''),
    worker_socket: String(data.worker_socket ?? ''),
    bind_host: String(data.bind_host ?? ''),
    api_prefix: String(data.api_prefix ?? API_PREFIX),
  };
}

export async function createLearner(body: LearnerCreateRequest): Promise<Learner> {
  const payload: LearnerCreateRequest = { username: body.username };
  if (body.display_name) {
    payload.display_name = body.display_name;
  }
  const data = await requestJson<unknown>('/learners', {
    method: 'POST',
    ...jsonBody(payload),
  });
  const learner = asLearner(data);
  if (!learner.username) {
    learner.username = body.username;
  }
  return learner;
}

export async function getLearner(learnerId: string): Promise<Learner> {
  return asLearner(await requestJson<unknown>(`/learners/${encodeURIComponent(learnerId)}`));
}

export async function createSession(body: SessionCreateRequest): Promise<Session> {
  return asSession(
    await requestJson<unknown>('/sessions', {
      method: 'POST',
      ...jsonBody(body),
    }),
  );
}

export async function getSession(sessionId: string): Promise<Session> {
  return asSession(await requestJson<unknown>(`/sessions/${encodeURIComponent(sessionId)}`));
}

export async function listMissions(): Promise<MissionListResponse> {
  const data = await requestJson<MissionListResponse | Mission[]>('/missions');
  if (Array.isArray(data)) {
    return { missions: data.map(asMission) };
  }
  const missions = Array.isArray(data.missions) ? data.missions.map(asMission) : [];
  return { missions };
}

export async function getMission(missionId: string): Promise<Mission> {
  return asMission(await requestJson<unknown>(`/missions/${encodeURIComponent(missionId)}`));
}

export async function putArtifact(input: ArtifactPutRequest): Promise<ArtifactPutResponse> {
  let response: Response;
  if (input.file) {
    const form = new FormData();
    form.append('file', input.file, input.filename ?? input.file.name);
    if (input.media_type) {
      form.append('media_type', input.media_type);
    }
    response = await request('/artifacts', { method: 'POST', body: form });
  } else {
    const body: Record<string, string> = {};
    if (input.bytes_b64) {
      body.bytes_b64 = input.bytes_b64;
    }
    if (input.media_type) {
      body.media_type = input.media_type;
    }
    if (input.filename) {
      body.filename = input.filename;
    }
    response = await request('/artifacts', {
      method: 'POST',
      ...jsonBody(body),
    });
  }
  const data = asRecord(await response.json());
  return {
    artifact_hash: String(data.artifact_hash ?? data.hash ?? ''),
    size: Number(data.size ?? 0),
  };
}

export async function getArtifact(artifactHash: string): Promise<ArtifactGetResponse> {
  const response = await request(`/artifacts/${encodeURIComponent(artifactHash)}`, {
    headers: { Accept: '*/*' },
  });
  const checksum =
    response.headers.get('x-checksum-sha256') ??
    response.headers.get('x-artifact-hash') ??
    response.headers.get('digest');
  const mediaType = response.headers.get('content-type');
  const bytes = new Uint8Array(await response.arrayBuffer());
  return {
    artifact_hash: artifactHash,
    bytes,
    checksum,
    media_type: mediaType,
  };
}

export async function loadCurriculumPackage(
  body: CurriculumPackageLoadRequest,
): Promise<CurriculumPackage | Record<string, unknown>> {
  return requestJson('/curriculum/packages/load', {
    method: 'POST',
    ...jsonBody({ package_dir: body.package_dir }),
  });
}

export async function listCurriculumPackages(): Promise<CurriculumPackageListResponse> {
  const data = await requestJson<CurriculumPackageListResponse | CurriculumPackage[]>(
    '/curriculum/packages',
  );
  if (Array.isArray(data)) {
    return { packages: data };
  }
  return { packages: Array.isArray(data.packages) ? data.packages : [] };
}

export async function createBackup(): Promise<BackupResponse> {
  const data = await requestJson<Record<string, unknown>>('/system/backup', {
    method: 'POST',
    ...jsonBody({}),
  });
  return {
    backup_id: String(data.backup_id ?? ''),
    path: String(data.path ?? ''),
  };
}

export async function restoreBackup(body: RestoreRequest): Promise<Record<string, unknown>> {
  const payload: RestoreRequest = {};
  if (body.backup_id) {
    payload.backup_id = body.backup_id;
  } else if (body.path) {
    payload.path = body.path;
  }
  const destHome = body.dest_home?.trim();
  if (destHome) {
    payload.dest_home = destHome;
  }
  return requestJson('/system/restore', {
    method: 'POST',
    ...jsonBody(payload),
  });
}

export const apiClient = {
  setAuthToken,
  getAuthToken,
  bootstrap,
  getHealth,
  getVersion,
  getConfig,
  createLearner,
  getLearner,
  createSession,
  getSession,
  listMissions,
  getMission,
  putArtifact,
  getArtifact,
  loadCurriculumPackage,
  listCurriculumPackages,
  createBackup,
  restoreBackup,
};
