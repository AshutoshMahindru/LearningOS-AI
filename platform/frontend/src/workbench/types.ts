import type { StructuredResultBlock } from '../api/types';

export const STRUCTURED_RESULT_BLOCK_TYPES = [
  'table',
  'chart',
  'trace',
  'state_diff',
  'diagram',
  'markdown',
  'metric',
  'artifact',
] as const;

export type StructuredResultBlockType = (typeof STRUCTURED_RESULT_BLOCK_TYPES)[number];

export type WorkbenchBlock = StructuredResultBlock;

export type StructuredResult = {
  execution_id?: string;
  status?: string;
  exit_code?: number;
  duration_ms?: number;
  blocks?: WorkbenchBlock[];
  diagnostics?: Record<string, unknown>;
};

export type WorkspaceEntry = {
  path: string;
  kind?: 'file' | 'directory';
  artifact_hash?: string;
  media_type?: string;
  size?: number;
  content?: string;
};

export type WorkspaceTreeNode = {
  name: string;
  path: string;
  kind: 'file' | 'directory';
  children: WorkspaceTreeNode[];
  entry?: WorkspaceEntry;
};
