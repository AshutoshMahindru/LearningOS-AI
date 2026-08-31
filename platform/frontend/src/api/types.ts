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
};

export type Mission = {
  id: string;
  title?: string;
  description?: string;
  version?: string;
  phase_id?: string;
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
