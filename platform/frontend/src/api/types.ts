export const API_PREFIX = '/api/v1';

export type HealthStatus = 'HEALTHY' | 'DEGRADED' | 'UNHEALTHY';

export type HealthResponse = {
  status: HealthStatus;
  version: string;
  worker_alive: boolean;
  database_path: string;
};

export type VersionResponse = {
  version: string;
};

export type ConfigResponse = {
  data_home: string;
  database_path: string;
  worker_socket: string;
  bind_host: string;
  api_prefix: string;
};

export type BootstrapResponse = {
  token: string;
  token_type: string;
};

export type LearnerCreateRequest = {
  username: string;
  display_name?: string;
};

export type Learner = {
  id: string;
  username: string;
  display_name?: string | null;
};

export type SessionCreateRequest = {
  mission_id: string;
  learner_id: string;
};

export type Session = {
  session_id: string;
  mission_id: string;
  learner_id: string;
  current_stage_id?: string | null;
  created_at?: string | null;
  status?: string | null;
};

export const CANONICAL_STAGE_TYPES = [
  'orientation',
  'trace_map',
  'interrogate',
  'experiment',
  'code_reading',
  'rebuild_debug',
  'controlled_failure',
  'transfer_assessment',
  'competency_gate',
  'reflection_adr',
  'flagship_integration',
] as const;

export type CanonicalStageType = (typeof CANONICAL_STAGE_TYPES)[number];

export type StageType = CanonicalStageType | (string & {});

export type AssistancePolicy =
  | 'UNRESTRICTED'
  | 'SOCRATIC_ONLY'
  | 'RESTRICTED_HINTS'
  | 'NO_AI_REQUIRED'
  | (string & {});

export type MissionStageRunner = {
  module?: string;
  entrypoint?: string;
  timeout_sec?: number;
};

export type MissionStageRubric = {
  required_evidence_type?: string;
  pass_criteria?: string;
};

export type MissionStage = {
  id: string;
  title: string;
  type: StageType;
  assistance_policy?: AssistancePolicy;
  instructions?: string;
  runner?: MissionStageRunner;
  validation_rubric?: MissionStageRubric;
};

export type GateEvidenceRequirement = {
  competency_id?: string;
  stage_id?: string;
  artifact_type?: string;
};

export type GateContract = {
  required_evidence?: GateEvidenceRequirement[];
  pass_threshold?: number;
  repair_policy?: {
    allow_targeted_repair?: boolean;
    max_repair_attempts?: number;
  };
};

export type Mission = {
  id: string;
  title?: string;
  description?: string;
  version?: string;
  phase_id?: string;
  phase?: { id?: string; title?: string };
  core_invariant?: string;
  competencies?: string[];
  knowledge_nodes?: string[];
  prerequisites?: string[];
  stages?: MissionStage[];
  gate_contract?: GateContract;
};

export type MissionListResponse = {
  missions: Mission[];
};

export type ArtifactPutRequest = {
  file?: File;
  bytes_b64?: string;
  media_type?: string;
  filename?: string;
};

export type ArtifactPutResponse = {
  artifact_hash: string;
  size: number;
};

export type ArtifactGetResponse = {
  artifact_hash: string;
  bytes: Uint8Array;
  checksum: string | null;
  media_type: string | null;
};

export type CurriculumPackageLoadRequest = {
  package_dir: string;
};

export type CurriculumPackage = {
  id: string;
  version?: string;
  path?: string;
  digest?: string;
};

export type CurriculumPackageListResponse = {
  packages: CurriculumPackage[];
};

export type BackupResponse = {
  backup_id: string;
  path: string;
};

export type RestoreRequest = {
  backup_id?: string;
  path?: string;
  dest_home?: string;
};

export type PredictCommitRequest = {
  hypothesis: string;
  expected_values: Record<string, unknown>;
};

export type ExecuteStageRequest = {
  code?: string;
  parameters?: Record<string, unknown>;
};

export type SubmitStageRequest = {
  artifacts?: Array<Record<string, unknown>>;
  explanation?: string;
};

export type StageEnterResponse = {
  session_id?: string;
  stage_id?: string;
  status: string;
  assistance_policy?: string;
  attempt_id?: string;
  current_stage_id?: string;
  prediction_sealed?: boolean;
};

export type PredictCommitResponse = {
  status: string;
  prediction_hash?: string;
  sealed_at?: string;
  hypothesis?: string;
};

export type StructuredResultBlock = {
  type: string;
  title?: string;
  payload?: Record<string, unknown>;
};

export type ExecuteStageResponse = {
  status: string;
  execution_id?: string;
  exit_code?: number;
  duration_ms?: number;
  blocks?: StructuredResultBlock[];
  diagnostics?: Record<string, unknown>;
};

export type SubmitStageResponse = {
  status: string;
  passed?: boolean;
  next_stage_id?: string | null;
  current_stage_id?: string | null;
};

export type GateEvaluateResponse = {
  status: string;
  score?: number;
  passed?: boolean;
  repair_plan?: Record<string, unknown> | null;
};

export type ApiErrorEnvelope = {
  error: {
    code: string;
    message: string;
    details?: Record<string, unknown> | null;
  };
};

export type MappedApiError = {
  code: string;
  message: string;
  details: Record<string, unknown>;
};
