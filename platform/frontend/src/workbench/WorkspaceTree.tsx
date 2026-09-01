import { EmptyState } from '../components/EmptyState';
import { cn } from '../cn';
import { nestWorkspace } from './payload';
import type { WorkspaceEntry, WorkspaceTreeNode } from './types';

type WorkspaceTreeProps = {
  files: WorkspaceEntry[];
  selectedPath?: string | null;
  onSelect?: (entry: WorkspaceEntry) => void;
};

export function WorkspaceTree({ files, selectedPath, onSelect }: WorkspaceTreeProps) {
  const nodes = nestWorkspace(files);

  if (nodes.length === 0) {
    return (
      <EmptyState
        title="No workspace files"
        message="Provide artifacts or a file list. The tree is generic and is not bound to a mission catalog."
      />
    );
  }

  return (
    <nav aria-label="Workspace" data-testid="workspace-tree">
      <ul className="space-y-0.5 font-mono text-sm">{nodes.map((node) => (
        <TreeNodeView
          key={node.path}
          node={node}
          depth={0}
          selectedPath={selectedPath}
          onSelect={onSelect}
        />
      ))}</ul>
    </nav>
  );
}

function TreeNodeView({
  node,
  depth,
  selectedPath,
  onSelect,
}: {
  node: WorkspaceTreeNode;
  depth: number;
  selectedPath?: string | null;
  onSelect?: (entry: WorkspaceEntry) => void;
}) {
  const selected = selectedPath === node.path;
  if (node.kind === 'directory') {
    return (
      <li>
        <p className="px-2 py-1 text-textSecondary" style={{ paddingLeft: `${0.5 + depth}rem` }}>
          {node.name}/
        </p>
        {node.children.length > 0 ? (
          <ul>
            {node.children.map((child) => (
              <TreeNodeView
                key={child.path}
                node={child}
                depth={depth + 1}
                selectedPath={selectedPath}
                onSelect={onSelect}
              />
            ))}
          </ul>
        ) : null}
      </li>
    );
  }

  return (
    <li>
      <button
        type="button"
        className={cn(
          'w-full rounded-md px-2 py-1 text-left',
          selected ? 'bg-elevated text-primary' : 'hover:bg-elevated',
        )}
        style={{ paddingLeft: `${0.5 + depth}rem` }}
        aria-current={selected ? 'true' : undefined}
        onClick={() => node.entry && onSelect?.(node.entry)}
      >
        {node.name}
      </button>
    </li>
  );
}
