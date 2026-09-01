import type { StructuredResult, WorkbenchBlock, WorkspaceEntry, WorkspaceTreeNode } from './types';

export function asRecord(value: unknown): Record<string, unknown> {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return {};
}

export function firstString(record: Record<string, unknown>, keys: string[]): string | undefined {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === 'string' && value.length > 0) {
      return value;
    }
  }
  return undefined;
}

export function asNumberList(value: unknown): number[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((item) => Number(item)).filter((item) => Number.isFinite(item));
}

export function formatCell(value: unknown): string {
  if (value == null) {
    return '';
  }
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

export function asBlock(value: unknown): WorkbenchBlock {
  const record = asRecord(value);
  return {
    type: typeof record.type === 'string' && record.type.length > 0 ? record.type : 'markdown',
    title: typeof record.title === 'string' ? record.title : undefined,
    payload: asRecord(record.payload),
  };
}

export function parseStructuredResult(value: unknown): StructuredResult {
  if (Array.isArray(value)) {
    return { blocks: value.map(asBlock) };
  }
  if (!value || typeof value !== 'object') {
    throw new Error('Structured result must be a JSON object or a block array');
  }
  const record = value as Record<string, unknown>;
  if (Array.isArray(record.blocks)) {
    return {
      execution_id: typeof record.execution_id === 'string' ? record.execution_id : undefined,
      status: typeof record.status === 'string' ? record.status : undefined,
      exit_code: typeof record.exit_code === 'number' ? record.exit_code : undefined,
      duration_ms: typeof record.duration_ms === 'number' ? record.duration_ms : undefined,
      blocks: record.blocks.map(asBlock),
      diagnostics: Object.keys(asRecord(record.diagnostics)).length > 0 ? asRecord(record.diagnostics) : undefined,
    };
  }
  if (typeof record.type === 'string') {
    return { blocks: [asBlock(record)] };
  }
  throw new Error('JSON must include a blocks array');
}

export function parseTable(payload: Record<string, unknown>): { columns: string[]; rows: unknown[][] } {
  const columnField = payload.columns;
  const rowField = payload.rows ?? payload.data;
  if (Array.isArray(columnField) && Array.isArray(rowField)) {
    const columns = columnField.map((item) => String(item));
    const rows = rowField.map((row) => {
      if (Array.isArray(row)) {
        return row;
      }
      if (row && typeof row === 'object') {
        const record = row as Record<string, unknown>;
        return columns.map((column) => record[column]);
      }
      return [row];
    });
    return { columns, rows };
  }

  const records = Array.isArray(payload.records) ? payload.records : Array.isArray(payload.data) ? payload.data : null;
  if (records && records.every((item) => item && typeof item === 'object' && !Array.isArray(item))) {
    const columns: string[] = [];
    for (const item of records) {
      for (const key of Object.keys(item as object)) {
        if (!columns.includes(key)) {
          columns.push(key);
        }
      }
    }
    const rows = records.map((item) => columns.map((column) => (item as Record<string, unknown>)[column]));
    return { columns, rows };
  }

  return { columns: [], rows: [] };
}

export type ChartSeries = {
  name: string;
  x: number[];
  y: number[];
};

export function parseChart(payload: Record<string, unknown>): { chartType: string; series: ChartSeries[] } {
  const chartType = (firstString(payload, ['chart_type', 'type']) ?? 'line').toLowerCase();
  if (Array.isArray(payload.series)) {
    const series = payload.series
      .map((item, index) => seriesFromUnknown(item, `series ${index + 1}`))
      .filter((item): item is ChartSeries => item !== null);
    return { chartType, series };
  }
  const fromPoints = seriesFromPoints(payload.points, firstString(payload, ['name', 'label']) ?? 'series');
  if (fromPoints) {
    return { chartType, series: [fromPoints] };
  }
  const single = seriesFromUnknown(payload, firstString(payload, ['name', 'label']) ?? 'series');
  return { chartType, series: single ? [single] : [] };
}

function seriesFromUnknown(value: unknown, fallbackName: string): ChartSeries | null {
  const record = asRecord(value);
  const y = asNumberList(record.y ?? record.values ?? (Array.isArray(value) ? value : undefined));
  if (y.length === 0) {
    return null;
  }
  const xRaw = asNumberList(record.x);
  const x = xRaw.length === y.length ? xRaw : y.map((_, index) => index);
  return {
    name: firstString(record, ['name', 'label']) ?? fallbackName,
    x,
    y,
  };
}

function seriesFromPoints(value: unknown, name: string): ChartSeries | null {
  if (!Array.isArray(value) || value.length === 0) {
    return null;
  }
  const x: number[] = [];
  const y: number[] = [];
  for (const item of value) {
    if (Array.isArray(item) && item.length >= 2) {
      const px = Number(item[0]);
      const py = Number(item[1]);
      if (Number.isFinite(px) && Number.isFinite(py)) {
        x.push(px);
        y.push(py);
      }
      continue;
    }
    const record = asRecord(item);
    const px = Number(record.x);
    const py = Number(record.y);
    if (Number.isFinite(px) && Number.isFinite(py)) {
      x.push(px);
      y.push(py);
    }
  }
  if (y.length === 0) {
    return null;
  }
  return { name, x, y };
}

export type TraceStep = {
  id: string;
  label: string;
  status?: string;
  detail?: string;
};

export function parseTrace(payload: Record<string, unknown>): TraceStep[] {
  const raw = payload.steps ?? payload.events ?? payload.frames ?? payload.trace;
  if (!Array.isArray(raw)) {
    return [];
  }
  return raw.map((item, index) => {
    if (typeof item === 'string') {
      return { id: String(index), label: item };
    }
    const record = asRecord(item);
    return {
      id: String(record.id ?? record.name ?? index),
      label: String(record.label ?? record.name ?? record.id ?? `step ${index + 1}`),
      status: typeof record.status === 'string' ? record.status : undefined,
      detail: firstString(record, ['detail', 'message', 'data', 'output']) ?? undefined,
    };
  });
}

export function flattenRecord(value: unknown, prefix = ''): Record<string, string> {
  const out: Record<string, string> = {};
  if (value === null || value === undefined) {
    if (prefix) {
      out[prefix] = String(value);
    }
    return out;
  }
  if (typeof value !== 'object') {
    out[prefix || 'value'] = String(value);
    return out;
  }
  if (Array.isArray(value)) {
    if (value.length === 0 && prefix) {
      out[prefix] = '[]';
    }
    value.forEach((item, index) => {
      Object.assign(out, flattenRecord(item, prefix ? `${prefix}[${index}]` : `[${index}]`));
    });
    return out;
  }
  const keys = Object.keys(value);
  if (keys.length === 0 && prefix) {
    out[prefix] = '{}';
  }
  for (const key of keys) {
    const path = prefix ? `${prefix}.${key}` : key;
    Object.assign(out, flattenRecord((value as Record<string, unknown>)[key], path));
  }
  return out;
}

export type DiffRow = {
  path: string;
  before: string;
  after: string;
  change: 'added' | 'removed' | 'changed' | 'same';
};

export function parseStateDiff(payload: Record<string, unknown>): DiffRow[] {
  if (Array.isArray(payload.changes) || Array.isArray(payload.diff)) {
    const raw = (payload.changes ?? payload.diff) as unknown[];
    return raw.map((item, index) => {
      const record = asRecord(item);
      const before = formatCell(record.before ?? record.from ?? record.left);
      const after = formatCell(record.after ?? record.to ?? record.right);
      const change =
        record.op === 'add' || record.change === 'added'
          ? 'added'
          : record.op === 'remove' || record.change === 'removed'
            ? 'removed'
            : before === after
              ? 'same'
              : 'changed';
      return {
        path: String(record.path ?? record.key ?? index),
        before,
        after,
        change,
      };
    });
  }

  const before = flattenRecord(payload.before ?? payload.left ?? payload.expected ?? {});
  const after = flattenRecord(payload.after ?? payload.right ?? payload.actual ?? {});
  const keys = [...new Set([...Object.keys(before), ...Object.keys(after)])].sort();
  return keys.map((path) => {
    const hasBefore = Object.prototype.hasOwnProperty.call(before, path);
    const hasAfter = Object.prototype.hasOwnProperty.call(after, path);
    const beforeValue = hasBefore ? before[path] : '';
    const afterValue = hasAfter ? after[path] : '';
    const change: DiffRow['change'] = !hasBefore
      ? 'added'
      : !hasAfter
        ? 'removed'
        : beforeValue === afterValue
          ? 'same'
          : 'changed';
    return { path, before: beforeValue, after: afterValue, change };
  });
}

export type DiagramNode = {
  id: string;
  label: string;
  x?: number;
  y?: number;
};

export type DiagramEdge = {
  from: string;
  to: string;
  label?: string;
};

export function parseDiagram(payload: Record<string, unknown>): { nodes: DiagramNode[]; edges: DiagramEdge[] } {
  const rawNodes = payload.nodes ?? payload.vertices;
  const nodes: DiagramNode[] = Array.isArray(rawNodes)
    ? rawNodes.map((item, index) => {
        if (typeof item === 'string') {
          return { id: item, label: item };
        }
        const record = asRecord(item);
        const id = String(record.id ?? record.name ?? record.label ?? `n${index}`);
        return {
          id,
          label: String(record.label ?? record.name ?? record.id ?? id),
          x: typeof record.x === 'number' ? record.x : undefined,
          y: typeof record.y === 'number' ? record.y : undefined,
        };
      })
    : [];

  const rawEdges = payload.edges ?? payload.links;
  const edges: DiagramEdge[] = Array.isArray(rawEdges)
    ? rawEdges.map((item) => {
        const record = asRecord(item);
        return {
          from: String(record.from ?? record.source ?? record.from_id ?? ''),
          to: String(record.to ?? record.target ?? record.to_id ?? ''),
          label: firstString(record, ['label', 'name']),
        };
      })
    : [];

  return { nodes, edges };
}

export function markdownText(payload: Record<string, unknown>): string {
  return firstString(payload, ['markdown', 'text', 'body', 'content']) ?? '';
}

export function filesFromBlocks(blocks: WorkbenchBlock[]): WorkspaceEntry[] {
  const files: WorkspaceEntry[] = [];
  for (const block of blocks) {
    if (block.type !== 'artifact') {
      continue;
    }
    const payload = asRecord(block.payload);
    const path = firstString(payload, ['path', 'filename', 'name']) ?? block.title ?? 'artifact';
    files.push({
      path,
      kind: 'file',
      artifact_hash: firstString(payload, ['artifact_hash', 'hash']),
      media_type: firstString(payload, ['media_type', 'content_type']),
      size: typeof payload.size === 'number' ? payload.size : undefined,
      content: firstString(payload, ['preview', 'text', 'content']),
    });
  }
  return files;
}

export function nestWorkspace(entries: WorkspaceEntry[]): WorkspaceTreeNode[] {
  const root: WorkspaceTreeNode = {
    name: '',
    path: '',
    kind: 'directory',
    children: [],
  };

  for (const entry of entries) {
    const parts = entry.path.split('/').filter(Boolean);
    if (parts.length === 0) {
      continue;
    }
    let current = root;
    parts.forEach((part, index) => {
      const isLast = index === parts.length - 1;
      const path = parts.slice(0, index + 1).join('/');
      let child = current.children.find((node) => node.name === part);
      if (!child) {
        child = {
          name: part,
          path,
          kind: isLast ? (entry.kind ?? 'file') : 'directory',
          children: [],
          entry: isLast ? entry : undefined,
        };
        current.children.push(child);
      } else if (isLast) {
        child.entry = entry;
        child.kind = entry.kind ?? child.kind;
      }
      current = child;
    });
  }

  const sortNodes = (nodes: WorkspaceTreeNode[]): WorkspaceTreeNode[] => {
    nodes.sort((a, b) => {
      if (a.kind !== b.kind) {
        return a.kind === 'directory' ? -1 : 1;
      }
      return a.name.localeCompare(b.name);
    });
    nodes.forEach((node) => {
      node.children = sortNodes(node.children);
    });
    return nodes;
  };

  return sortNodes(root.children);
}
