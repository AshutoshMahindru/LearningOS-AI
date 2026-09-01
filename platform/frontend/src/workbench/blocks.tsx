import type { ReactNode } from 'react';
import {
  asRecord,
  firstString,
  formatCell,
  markdownText,
  parseChart,
  parseDiagram,
  parseStateDiff,
  parseTable,
  parseTrace,
  type ChartSeries,
} from './payload';

type BlockBodyProps = {
  payload: Record<string, unknown>;
};

function MissingPayload({ message }: { message: string }) {
  return <p className="text-sm text-textSecondary">{message}</p>;
}

export function TableBlock({ payload }: BlockBodyProps) {
  const { columns, rows } = parseTable(payload);
  if (columns.length === 0) {
    return <MissingPayload message="No tabular columns in this payload." />;
  }
  return (
    <div className="overflow-auto">
      <table className="min-w-full border-collapse text-sm" data-testid="table-block">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column} className="border-b border-border px-2 py-1 text-left font-semibold">
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {columns.map((column, columnIndex) => (
                <td key={`${column}-${columnIndex}`} className="border-b border-border/60 px-2 py-1 font-mono text-xs">
                  {formatCell(row[columnIndex])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const CHART_COLORS = ['#7dd3fc', '#6ee7b7', '#fcd34d', '#fca5a5'];

export function ChartBlock({ payload }: BlockBodyProps) {
  const { chartType, series } = parseChart(payload);
  if (series.length === 0) {
    return <MissingPayload message="No numeric series in this chart payload." />;
  }

  const width = 320;
  const height = 180;
  const pad = { top: 16, right: 16, bottom: 28, left: 36 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const allX = series.flatMap((item) => item.x);
  const allY = series.flatMap((item) => item.y);
  const minX = Math.min(...allX);
  const maxX = Math.max(...allX);
  const minY = Math.min(0, ...allY);
  const maxY = Math.max(...allY);
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const sx = (value: number) => pad.left + ((value - minX) / spanX) * innerW;
  const sy = (value: number) => pad.top + innerH - ((value - minY) / spanY) * innerH;
  const kind = chartType.includes('bar') ? 'bar' : chartType.includes('scatter') ? 'scatter' : 'line';

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="h-48 w-full max-w-xl text-primary"
      role="img"
      aria-label={`${kind} chart`}
      data-testid="chart-block"
    >
      <line x1={pad.left} y1={pad.top} x2={pad.left} y2={pad.top + innerH} stroke="currentColor" strokeOpacity="0.35" />
      <line
        x1={pad.left}
        y1={pad.top + innerH}
        x2={pad.left + innerW}
        y2={pad.top + innerH}
        stroke="currentColor"
        strokeOpacity="0.35"
      />
      <text x={pad.left} y={12} className="fill-current text-[10px]" fillOpacity="0.7">
        {maxY}
      </text>
      <text x={pad.left} y={height - 8} className="fill-current text-[10px]" fillOpacity="0.7">
        {minY}
      </text>
      {series.map((item, seriesIndex) => (
        <g key={item.name} stroke={CHART_COLORS[seriesIndex % CHART_COLORS.length]} fill={CHART_COLORS[seriesIndex % CHART_COLORS.length]}>
          {kind === 'bar' ? <BarSeries series={item} sx={sx} sy={sy} baseline={sy(minY)} /> : null}
          {kind === 'line' ? (
            <polyline
              fill="none"
              strokeWidth="2"
              points={item.x.map((x, index) => `${sx(x)},${sy(item.y[index] ?? 0)}`).join(' ')}
            />
          ) : null}
          {kind === 'scatter' || kind === 'line'
            ? item.x.map((x, index) => (
                <circle key={`${item.name}-${index}`} cx={sx(x)} cy={sy(item.y[index] ?? 0)} r={kind === 'scatter' ? 4 : 2.5} />
              ))
            : null}
        </g>
      ))}
    </svg>
  );
}

function BarSeries({
  series,
  sx,
  sy,
  baseline,
}: {
  series: ChartSeries;
  sx: (value: number) => number;
  sy: (value: number) => number;
  baseline: number;
}) {
  const firstX = series.x[0] ?? 0;
  const secondX = series.x[1];
  const width = secondX == null ? 16 : Math.abs(sx(secondX) - sx(firstX)) * 0.6;
  return (
    <>
      {series.x.map((x, index) => {
        const y = sy(series.y[index] ?? 0);
        const top = Math.min(y, baseline);
        const height = Math.max(1, Math.abs(baseline - y));
        return <rect key={index} x={sx(x) - width / 2} y={top} width={width} height={height} opacity="0.85" />;
      })}
    </>
  );
}

export function TraceBlock({ payload }: BlockBodyProps) {
  const steps = parseTrace(payload);
  if (steps.length === 0) {
    return <MissingPayload message="No trace steps in this payload." />;
  }
  return (
    <ol className="space-y-2" data-testid="trace-block">
      {steps.map((step, index) => (
        <li key={`${step.id}-${index}`} className="rounded-md border border-border bg-bg px-3 py-2">
          <p className="text-sm font-semibold">
            {step.label}
            {step.status ? <span className="ml-2 font-mono text-xs font-normal text-textSecondary">{step.status}</span> : null}
          </p>
          {step.detail ? <p className="mt-1 font-mono text-xs text-textSecondary">{step.detail}</p> : null}
        </li>
      ))}
    </ol>
  );
}

export function StateDiffBlock({ payload }: BlockBodyProps) {
  const rows = parseStateDiff(payload);
  if (rows.length === 0) {
    return <MissingPayload message="No before/after fields in this state diff." />;
  }
  return (
    <div className="overflow-auto" data-testid="state-diff-block">
      <table className="min-w-full border-collapse text-sm">
        <thead>
          <tr>
            <th className="border-b border-border px-2 py-1 text-left">Path</th>
            <th className="border-b border-border px-2 py-1 text-left">Before</th>
            <th className="border-b border-border px-2 py-1 text-left">After</th>
            <th className="border-b border-border px-2 py-1 text-left">Change</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.path} data-change={row.change}>
              <td className="border-b border-border/60 px-2 py-1 font-mono text-xs">{row.path}</td>
              <td className="border-b border-border/60 px-2 py-1 font-mono text-xs text-textSecondary">{row.before}</td>
              <td className="border-b border-border/60 px-2 py-1 font-mono text-xs">{row.after}</td>
              <td className="border-b border-border/60 px-2 py-1 text-xs">{row.change}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function DiagramBlock({ payload }: BlockBodyProps) {
  const { nodes, edges } = parseDiagram(payload);
  if (nodes.length === 0) {
    return <MissingPayload message="No diagram nodes in this payload." />;
  }
  const width = 360;
  const height = 240;
  const positioned = nodes.map((node, index) => {
    if (node.x != null && node.y != null) {
      return { ...node, x: node.x, y: node.y };
    }
    const angle = (2 * Math.PI * index) / nodes.length - Math.PI / 2;
    return {
      ...node,
      x: width / 2 + 120 * Math.cos(angle),
      y: height / 2 + 80 * Math.sin(angle),
    };
  });
  const byId = new Map(positioned.map((node) => [node.id, node]));

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="h-60 w-full max-w-xl text-primary"
      role="img"
      aria-label="Diagram"
      data-testid="diagram-block"
    >
      {edges.map((edge, index) => {
        const from = byId.get(edge.from);
        const to = byId.get(edge.to);
        if (!from || !to) {
          return null;
        }
        return (
          <g key={`${edge.from}-${edge.to}-${index}`}>
            <line x1={from.x} y1={from.y} x2={to.x} y2={to.y} stroke="currentColor" strokeOpacity="0.55" />
            {edge.label ? (
              <text x={(from.x + to.x) / 2} y={(from.y + to.y) / 2 - 6} className="fill-current text-[10px]" textAnchor="middle">
                {edge.label}
              </text>
            ) : null}
          </g>
        );
      })}
      {positioned.map((node) => (
        <g key={node.id}>
          <circle cx={node.x} cy={node.y} r="18" fill="var(--color-elevated)" stroke="currentColor" />
          <text x={node.x} y={node.y + 4} textAnchor="middle" className="fill-current text-[10px]">
            {node.label}
          </text>
        </g>
      ))}
    </svg>
  );
}

export function MarkdownBlock({ payload }: BlockBodyProps) {
  const source = markdownText(payload);
  if (!source) {
    return <MissingPayload message="No markdown text in this payload." />;
  }
  return (
    <div className="space-y-2 text-sm" data-testid="markdown-block">
      {splitFences(source).map((part, index) =>
        part.kind === 'code' ? (
          <pre key={index} className="overflow-auto rounded-md bg-bg p-3 font-mono text-xs">
            {part.value}
          </pre>
        ) : (
          <MarkdownFlow key={index} text={part.value} />
        ),
      )}
    </div>
  );
}

function splitFences(source: string): Array<{ kind: 'code' | 'text'; value: string }> {
  const parts: Array<{ kind: 'code' | 'text'; value: string }> = [];
  const fence = /```[^\n]*\n?([\s\S]*?)```/g;
  let last = 0;
  let match: RegExpExecArray | null = fence.exec(source);
  while (match) {
    if (match.index > last) {
      parts.push({ kind: 'text', value: source.slice(last, match.index) });
    }
    parts.push({ kind: 'code', value: (match[1] ?? '').replace(/\n$/, '') });
    last = match.index + match[0].length;
    match = fence.exec(source);
  }
  if (last < source.length) {
    parts.push({ kind: 'text', value: source.slice(last) });
  }
  return parts;
}

function MarkdownFlow({ text }: { text: string }) {
  const lines = text.replace(/\r\n/g, '\n').split('\n');
  const nodes: ReactNode[] = [];
  let index = 0;
  let key = 0;
  while (index < lines.length) {
    const line = lines[index] ?? '';
    if (!line.trim()) {
      index += 1;
      continue;
    }
    const heading = /^(#{1,4})\s+(.*)$/.exec(line);
    if (heading) {
      const level = heading[1]?.length ?? 1;
      const title = heading[2] ?? '';
      nodes.push(
        <MarkdownHeading key={key} level={level}>
          <InlineMarkdown text={title} />
        </MarkdownHeading>,
      );
      key += 1;
      index += 1;
      continue;
    }
    if (/^[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (index < lines.length && /^[-*]\s+/.test(lines[index] ?? '')) {
        items.push((lines[index] ?? '').replace(/^[-*]\s+/, ''));
        index += 1;
      }
      nodes.push(
        <ul key={key} className="list-disc space-y-1 pl-5">
          {items.map((item, itemIndex) => (
            <li key={itemIndex}>
              <InlineMarkdown text={item} />
            </li>
          ))}
        </ul>,
      );
      key += 1;
      continue;
    }
    const para: string[] = [];
    while (
      index < lines.length &&
      (lines[index] ?? '').trim() &&
      !/^(#{1,4})\s+/.test(lines[index] ?? '') &&
      !/^[-*]\s+/.test(lines[index] ?? '')
    ) {
      para.push(lines[index] ?? '');
      index += 1;
    }
    nodes.push(
      <p key={key}>
        <InlineMarkdown text={para.join(' ')} />
      </p>,
    );
    key += 1;
  }
  return <>{nodes}</>;
}

function MarkdownHeading({ level, children }: { level: number; children: ReactNode }) {
  const className =
    level <= 1 ? 'text-lg font-bold' : level === 2 ? 'text-base font-semibold' : 'text-sm font-semibold';
  if (level <= 1) {
    return <h4 className={className}>{children}</h4>;
  }
  if (level === 2) {
    return <h5 className={className}>{children}</h5>;
  }
  return <h6 className={className}>{children}</h6>;
}

function InlineMarkdown({ text }: { text: string }) {
  const nodes: ReactNode[] = [];
  const pattern = /\*\*([^*]+)\*\*|\*([^*]+)\*|`([^`]+)`|\[([^\]]+)\]\(([^)]+)\)/g;
  let last = 0;
  let match = pattern.exec(text);
  let key = 0;
  while (match) {
    if (match.index > last) {
      nodes.push(text.slice(last, match.index));
    }
    if (match[1]) {
      nodes.push(<strong key={key}>{match[1]}</strong>);
    } else if (match[2]) {
      nodes.push(<em key={key}>{match[2]}</em>);
    } else if (match[3]) {
      nodes.push(
        <code key={key} className="rounded bg-bg px-1 font-mono text-xs">
          {match[3]}
        </code>,
      );
    } else if (match[4] && match[5]) {
      const href = match[5];
      const safe = href.startsWith('https://') || href.startsWith('http://') || href.startsWith('#');
      nodes.push(
        safe ? (
          <a key={key} href={href} rel="noreferrer">
            {match[4]}
          </a>
        ) : (
          match[4]
        ),
      );
    }
    key += 1;
    last = match.index + match[0].length;
    match = pattern.exec(text);
  }
  if (last < text.length) {
    nodes.push(text.slice(last));
  }
  return <>{nodes}</>;
}

export function MetricBlock({ payload }: BlockBodyProps) {
  const items = Array.isArray(payload.items)
    ? payload.items.map((item, index) => {
        const record = asRecord(item);
        return {
          name: String(record.name ?? record.label ?? index),
          value: formatCell(record.value ?? record.metric),
        };
      })
    : Object.entries(payload).map(([name, value]) => ({
        name,
        value: formatCell(value),
      }));
  if (items.length === 0) {
    return <MissingPayload message="No metrics in this payload." />;
  }
  return (
    <dl className="grid gap-2 sm:grid-cols-2" data-testid="metric-block">
      {items.map((item) => (
        <div key={item.name} className="rounded-md border border-border bg-bg px-3 py-2">
          <dt className="text-xs uppercase tracking-wide text-textSecondary">{item.name}</dt>
          <dd className="font-mono text-sm">{item.value}</dd>
        </div>
      ))}
    </dl>
  );
}

export function ArtifactBlock({ payload }: BlockBodyProps) {
  const filename = firstString(payload, ['filename', 'path', 'name']) ?? 'artifact';
  const hash = firstString(payload, ['artifact_hash', 'hash']);
  const mediaType = firstString(payload, ['media_type', 'content_type']);
  const preview = firstString(payload, ['preview', 'text', 'content']);
  const bytesB64 = firstString(payload, ['bytes_b64', 'data']);
  const remote = firstString(payload, ['url', 'href']);
  const imageSrc =
    mediaType?.startsWith('image/') && bytesB64 ? `data:${mediaType};base64,${bytesB64}` : undefined;

  return (
    <div className="space-y-2 text-sm" data-testid="artifact-block">
      <p className="font-medium">{filename}</p>
      {hash ? <p className="break-all font-mono text-xs text-textSecondary">{hash}</p> : null}
      {mediaType ? <p className="text-xs text-textSecondary">{mediaType}</p> : null}
      {typeof payload.size === 'number' ? <p className="text-xs text-textSecondary">{payload.size} bytes</p> : null}
      {imageSrc ? <img src={imageSrc} alt={filename} className="max-h-48 rounded-md border border-border" /> : null}
      {preview ? (
        <pre className="overflow-auto rounded-md bg-bg p-3 font-mono text-xs">{preview}</pre>
      ) : null}
      {remote ? (
        <p className="break-all font-mono text-xs text-textSecondary">ref {remote}</p>
      ) : null}
    </div>
  );
}

export function FallbackBlock({ payload }: BlockBodyProps) {
  return (
    <pre className="overflow-auto rounded-md bg-bg p-3 font-mono text-xs" data-testid="fallback-block">
      {JSON.stringify(payload, null, 2)}
    </pre>
  );
}
