import { recordDiagnostic } from './diagnostics';
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
  ExecuteStageRequest,
  ExecuteStageResponse,
  GateContract,
  GateEvaluateResponse,
  HealthResponse,
  Learner,
  LearnerCreateRequest,
  MappedApiError,
  Mission,
  MissionListResponse,
  MissionStage,
  PredictCommitRequest,
  PredictCommitResponse,
  RestoreRequest,
  Session,
  SessionCreateRequest,
  StageEnterResponse,
  StructuredResultBlock,
  SubmitStageRequest,
  SubmitStageResponse,
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

function requestPath(path: string): string {
  return `${API_PREFIX}${path.startsWith('/') ? path : `/${path}`}`;
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

function optionalString(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

function optionalNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
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

function errorFromUnknown(error: unknown, status = 0): ApiError {
  if (error instanceof ApiError) {
    return error;
  }
  return new ApiError({
    code: 'INTERNAL',
    message: error instanceof Error ? error.message : 'Request failed',
    details: {},
    status,
  });
}

async function readCloneBody(response: Response): Promise<unknown> {
  const clone = response.clone();
  const contentType = clone.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    try {
      return await clone.json();
    } catch {
      return null;
    }
  }
  try {
    const text = await clone.text();
    return text ? text : null;
  } catch {
    return null;
  }
}

function mappedFromPayload(payload: unknown, status: number): MappedApiError | null {
  const envelope = asRecord(payload);
  const nested = asRecord(envelope.error);
  if (typeof nested.code === 'string' || typeof nested.message === 'string') {
    return {
      code: typeof nested.code === 'string' ? nested.code : 'INTERNAL',
      message: typeof nested.message === 'string' ? nested.message : 'Request failed',
      details: asDetails(nested.details),
    };
  }
  if (status >= 400) {
    return {
      code: 'INTERNAL',
      message: 'Request failed',
      details: asDetails(payload),
    };
  }
  return null;
}

async function request(path: string, options: RequestOptions = {}): Promise<Response> {
  const { auth = true, headers, ...rest } = options;
  const headerBag = new Headers(headers);
  if (auth) {
    if (!bearerToken) {
      const error = new ApiError({
        code: 'UNAUTHORIZED',
        message: 'Not authenticated',
        details: {},
        status: 401,
      });
      recordDiagnostic({
        method: (rest.method ?? 'GET').toUpperCase(),
        path: requestPath(path),
        status: 401,
        request_body: rest.body ?? null,
        response_body: null,
        error: { code: error.code, message: error.message, details: error.details },
      });
      throw error;
    }
    headerBag.set('Authorization', `Bearer ${bearerToken}`);
  }
  if (!headerBag.has('Accept')) {
    headerBag.set('Accept', 'application/json');
  }

  const method = (rest.method ?? 'GET').toUpperCase();
  const diagnosticPath = requestPath(path);
  let response: Response;
  try {
    response = await fetch(resolveUrl(path), {
      ...rest,
      headers: headerBag,
    });
  } catch (error) {
    const mapped = errorFromUnknown(error);
    recordDiagnostic({
      method,
      path: diagnosticPath,
      status: null,
      request_body: rest.body ?? null,
      response_body: null,
      error: { code: mapped.code, message: mapped.message, details: mapped.details },
    });
    throw mapped;
  }

  const responseBody = await readCloneBody(response);
  const mappedError = response.ok ? null : mappedFromPayload(responseBody, response.status);
  recordDiagnostic({
    method,
    path: diagnosticPath,
    status: response.status,
    request_body: rest.body ?? null,
    response_body: responseBody,
    error: mappedError,
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
  const text = await response.text();
  if (!text) {
    return undefined as T;
  }
  try {
    return JSON.parse(text) as T;
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
    status: typeof record.status === 'string' ? record.status : null,
  };
}

function asRunner(value: unknown): MissionStage['runner'] | undefined {
  const record = asRecord(value);
  if (Object.keys(record).length === 0) {
    return undefined;
  }
  return {
    module: optionalString(record.module),
    entrypoint: optionalString(record.entrypoint),
    timeout_sec: optionalNumber(record.timeout_sec),
  };
}

function asRubric(value: unknown): MissionStage['validation_rubric'] | undefined {
  const record = asRecord(value);
  if (Object.keys(record).length === 0) {
    return undefined;
  }
  return {
    required_evidence_type: optionalString(record.required_evidence_type),
    pass_criteria: optionalString(record.pass_criteria),
  };
}

function asStage(data: unknown): MissionStage | null {
  const record = asRecord(data);
  const id = optionalString(record.id);
  if (!id) {
    return null;
  }
  return {
    id,
    title: optionalString(record.title) ?? id,
    type: optionalString(record.type) ?? 'orientation',
    assistance_policy: optionalString(record.assistance_policy),
    instructions: optionalString(record.instructions),
    runner: asRunner(record.runner),
    validation_rubric: asRubric(record.validation_rubric),
  };
}

function asGateContract(value: unknown): GateContract | undefined {
  const record = asRecord(value);
  if (Object.keys(record).length === 0) {
    return undefined;
  }
  const required = Array.isArray(record.required_evidence)
    ? record.required_evidence.map((item) => {
        const row = asRecord(item);
        return {
          competency_id: optionalString(row.competency_id),
          stage_id: optionalString(row.stage_id),
          artifact_type: optionalString(row.artifact_type),
        };
      })
    : undefined;
  const repair = asRecord(record.repair_policy);
  return {
    required_evidence: required,
    pass_threshold: optionalNumber(record.pass_threshold),
    repair_policy:
      Object.keys(repair).length > 0
        ? {
            allow_targeted_repair:
              typeof repair.allow_targeted_repair === 'boolean' ? repair.allow_targeted_repair : undefined,
            max_repair_attempts: optionalNumber(repair.max_repair_attempts),
          }
        : undefined,
  };
}

function asStringList(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }
  return value.filter((item): item is string => typeof item === 'string');
}

function mergeSpecRecord(data: unknown): Record<string, unknown> {
  let record = asRecord(data);
  const nestedSpec = record.spec;
  if (nestedSpec && typeof nestedSpec === 'object' && !Array.isArray(nestedSpec)) {
    record = { ...record, ...asRecord(nestedSpec) };
  }
  const specJson = record.spec_json;
  if (typeof specJson === 'string' && specJson) {
    try {
      const parsed = JSON.parse(specJson);
      record = { ...record, ...asRecord(parsed) };
    } catch {
      // Keep the row fields when spec_json is not valid JSON.
    }
  }
  return record;
}

function asMission(data: unknown): Mission {
  const record = mergeSpecRecord(data);
  const id = String(record.id ?? record.mission_id ?? '');
  const phase = asRecord(record.phase);
  const stages = Array.isArray(record.stages)
    ? record.stages.map(asStage).filter((stage): stage is MissionStage => stage !== null)
    : undefined;
  return {
    id,
    title: optionalString(record.title),
    description: optionalString(record.description),
    version: optionalString(record.version),
    phase_id: optionalString(record.phase_id) ?? optionalString(phase.id),
    phase:
      optionalString(phase.id) || optionalString(phase.title)
        ? { id: optionalString(phase.id), title: optionalString(phase.title) }
        : undefined,
    core_invariant: optionalString(record.core_invariant),
    competencies: asStringList(record.competencies),
    knowledge_nodes: asStringList(record.knowledge_nodes),
    prerequisites: asStringList(record.prerequisites),
    stages,
    gate_contract: asGateContract(record.gate_contract),
  };
}

function asEnterResponse(data: unknown): StageEnterResponse {
  const record = asRecord(data);
  return {
    session_id: optionalString(record.session_id),
    stage_id: optionalString(record.stage_id),
    status: optionalString(record.status) ?? 'ENTERED',
    assistance_policy: optionalString(record.assistance_policy),
    attempt_id: optionalString(record.attempt_id),
    current_stage_id: optionalString(record.current_stage_id),
    prediction_sealed: record.prediction_sealed === true,
  };
}

function asPredictResponse(data: unknown): PredictCommitResponse {
  const record = asRecord(data);
  return {
    status: optionalString(record.status) ?? 'SEALED',
    prediction_hash: optionalString(record.prediction_hash) ?? optionalString(record.hash),
    sealed_at: optionalString(record.sealed_at),
    hypothesis: optionalString(record.hypothesis),
  };
}

function asBlocks(value: unknown): StructuredResultBlock[] | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }
  return value.map((item) => {
    const record = asRecord(item);
    return {
      type: optionalString(record.type) ?? 'markdown',
      title: optionalString(record.title),
      payload: asRecord(record.payload),
    };
  });
}

function asExecuteResponse(data: unknown): ExecuteStageResponse {
  const record = asRecord(data);
  const diagnostics = asRecord(record.diagnostics);
  return {
    status: optionalString(record.status) ?? 'SUCCESS',
    execution_id: optionalString(record.execution_id),
    exit_code: optionalNumber(record.exit_code),
    duration_ms: optionalNumber(record.duration_ms),
    blocks: asBlocks(record.blocks),
    diagnostics: Object.keys(diagnostics).length > 0 ? diagnostics : undefined,
  };
}

function asSubmitResponse(data: unknown): SubmitStageResponse {
  const record = asRecord(data);
  return {
    status: optionalString(record.status) ?? 'SUBMITTED',
    passed: typeof record.passed === 'boolean' ? record.passed : undefined,
    next_stage_id: typeof record.next_stage_id === 'string' ? record.next_stage_id : null,
    current_stage_id: typeof record.current_stage_id === 'string' ? record.current_stage_id : null,
  };
}

function asGateResponse(data: unknown): GateEvaluateResponse {
  const record = asRecord(data);
  const repair = asRecord(record.repair_plan);
  return {
    status: optionalString(record.status) ?? 'EVALUATED',
    score: optionalNumber(record.score),
    passed: typeof record.passed === 'boolean' ? record.passed : record.status === 'PASSED',
    repair_plan: Object.keys(repair).length > 0 ? repair : null,
  };
}

function stagePath(sessionId: string, stageId: string, action: string): string {
  return `/sessions/${encodeURIComponent(sessionId)}/stages/${encodeURIComponent(stageId)}/${action}`;
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

export async function enterStage(sessionId: string, stageId: string): Promise<StageEnterResponse> {
  const data = await requestJson<unknown>(stagePath(sessionId, stageId, 'enter'), {
    method: 'POST',
    ...jsonBody({}),
  });
  return asEnterResponse(data);
}

export async function predictStage(
  sessionId: string,
  stageId: string,
  body: PredictCommitRequest,
): Promise<PredictCommitResponse> {
  const data = await requestJson<unknown>(stagePath(sessionId, stageId, 'predict'), {
    method: 'POST',
    ...jsonBody({
      hypothesis: body.hypothesis,
      expected_values: body.expected_values,
    }),
  });
  return asPredictResponse(data);
}

export async function executeStage(
  sessionId: string,
  stageId: string,
  body: ExecuteStageRequest = {},
): Promise<ExecuteStageResponse> {
  const payload: ExecuteStageRequest = {};
  if (body.code) {
    payload.code = body.code;
  }
  if (body.parameters) {
    payload.parameters = body.parameters;
  }
  const data = await requestJson<unknown>(stagePath(sessionId, stageId, 'execute'), {
    method: 'POST',
    ...jsonBody(payload),
  });
  return asExecuteResponse(data);
}

export async function submitStage(
  sessionId: string,
  stageId: string,
  body: SubmitStageRequest = {},
): Promise<SubmitStageResponse> {
  const payload: SubmitStageRequest = {};
  if (body.artifacts) {
    payload.artifacts = body.artifacts;
  }
  if (body.explanation) {
    payload.explanation = body.explanation;
  }
  const data = await requestJson<unknown>(stagePath(sessionId, stageId, 'submit'), {
    method: 'POST',
    ...jsonBody(payload),
  });
  return asSubmitResponse(data);
}

export async function evaluateGates(sessionId: string): Promise<GateEvaluateResponse> {
  const data = await requestJson<unknown>(`/sessions/${encodeURIComponent(sessionId)}/gates/evaluate`, {
    method: 'POST',
    ...jsonBody({}),
  });
  return asGateResponse(data);
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
  enterStage,
  predictStage,
  executeStage,
  submitStage,
  evaluateGates,
  putArtifact,
  getArtifact,
  loadCurriculumPackage,
  listCurriculumPackages,
  createBackup,
  restoreBackup,
};
