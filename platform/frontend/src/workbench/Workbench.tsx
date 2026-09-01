import { useMemo, useState } from 'react';
import { Panel } from '../components/Panel';
import { StatusBanner } from '../components/StatusBanner';
import { BlockList } from './BlockRenderer';
import { CodeEditor } from './CodeEditor';
import { filesFromBlocks } from './payload';
import type { StructuredResult, WorkbenchBlock, WorkspaceEntry } from './types';
import { WorkspaceTree } from './WorkspaceTree';

export type WorkbenchProps = {
  code?: string;
  onCodeChange?: (value: string) => void;
  files?: WorkspaceEntry[];
  result?: StructuredResult | null;
  blocks?: WorkbenchBlock[];
};

export function Workbench({ code = '', onCodeChange, files, result = null, blocks }: WorkbenchProps) {
  const resolvedBlocks = result?.blocks ?? blocks ?? [];
  const [localCode, setLocalCode] = useState(code);
  const treeFiles = useMemo(() => {
    const derived = filesFromBlocks(result?.blocks ?? blocks ?? []);
    if (!files || files.length === 0) {
      return derived;
    }
    const seen = new Set(files.map((entry) => entry.path));
    return [...files, ...derived.filter((entry) => !seen.has(entry.path))];
  }, [files, result?.blocks, blocks]);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const editorValue = onCodeChange ? code : localCode;
  const handleCodeChange = (value: string) => {
    if (!onCodeChange) {
      setLocalCode(value);
    }
    onCodeChange?.(value);
  };

  const statusTone =
    result?.status === 'FAILED' || result?.status === 'CRASHED' || result?.status === 'TIMEOUT'
      ? 'error'
      : result?.status === 'SUCCESS'
        ? 'success'
        : 'info';

  return (
    <div className="grid gap-4 lg:grid-cols-[16rem_minmax(0,1fr)]" data-testid="workbench-layout">
      <Panel title="Workspace" description="Generic file list from artifacts or caller-supplied paths.">
        <WorkspaceTree
          files={treeFiles}
          selectedPath={selectedPath}
          onSelect={(entry) => {
            setSelectedPath(entry.path);
            if (entry.content && !entry.media_type?.startsWith('image/')) {
              handleCodeChange(entry.content);
            }
          }}
        />
      </Panel>
      <div className="space-y-4">
        <Panel title="Editor">
          <CodeEditor value={editorValue} onChange={handleCodeChange} />
        </Panel>
        <Panel title="Results" description="WP-137 structured-result blocks.">
          {result?.status ? (
            <StatusBanner tone={statusTone} title={result.status} className="mb-4">
              {[
                result.execution_id ? `execution ${result.execution_id}` : null,
                result.exit_code != null ? `exit ${result.exit_code}` : null,
                result.duration_ms != null ? `${result.duration_ms} ms` : null,
              ]
                .filter(Boolean)
                .join(' · ')}
            </StatusBanner>
          ) : null}
          <BlockList blocks={resolvedBlocks} />
          {result?.diagnostics ? (
            <details className="mt-4">
              <summary className="cursor-pointer text-sm font-medium">Diagnostics</summary>
              <pre className="mt-2 overflow-auto rounded-md bg-bg p-3 font-mono text-xs">
                {JSON.stringify(result.diagnostics, null, 2)}
              </pre>
            </details>
          ) : null}
        </Panel>
      </div>
    </div>
  );
}
