import type { ComponentType } from 'react';
import { EmptyState } from '../components/EmptyState';
import {
  ArtifactBlock,
  ChartBlock,
  DiagramBlock,
  FallbackBlock,
  MarkdownBlock,
  MetricBlock,
  StateDiffBlock,
  TableBlock,
  TraceBlock,
} from './blocks';
import { asRecord } from './payload';
import type { WorkbenchBlock } from './types';

const RENDERERS: Record<string, ComponentType<{ payload: Record<string, unknown> }>> = {
  table: TableBlock,
  chart: ChartBlock,
  trace: TraceBlock,
  state_diff: StateDiffBlock,
  diagram: DiagramBlock,
  markdown: MarkdownBlock,
  metric: MetricBlock,
  artifact: ArtifactBlock,
};

type BlockRendererProps = {
  block: WorkbenchBlock;
  index?: number;
};

export function BlockRenderer({ block, index = 0 }: BlockRendererProps) {
  const type = block.type || 'markdown';
  const Renderer = RENDERERS[type] ?? FallbackBlock;
  const payload = asRecord(block.payload);
  return (
    <article
      className="rounded-md border border-border bg-bg p-3"
      data-testid={`block-${type}`}
      data-block-type={type}
      data-block-index={index}
    >
      {block.title ? <h3 className="mb-2 text-sm font-semibold">{block.title}</h3> : null}
      <Renderer payload={payload} />
    </article>
  );
}

export function BlockList({ blocks }: { blocks: WorkbenchBlock[] }) {
  if (blocks.length === 0) {
    return (
      <EmptyState
        title="No result blocks"
        message="Load a structured result to render table, chart, trace, state diff, diagram, markdown, metric, and artifact blocks."
      />
    );
  }
  return (
    <div className="space-y-3" data-testid="workbench-blocks">
      {blocks.map((block, index) => (
        <BlockRenderer key={`${block.type}-${index}`} block={block} index={index} />
      ))}
    </div>
  );
}
