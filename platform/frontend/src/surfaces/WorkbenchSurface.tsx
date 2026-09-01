import { useState, type FormEvent } from 'react';
import { Button } from '../components/Button';
import { Panel } from '../components/Panel';
import { StatusBanner } from '../components/StatusBanner';
import { parseStructuredResult } from '../workbench/payload';
import type { StructuredResult, WorkspaceEntry } from '../workbench/types';
import { Workbench } from '../workbench/Workbench';

async function entryFromFile(file: File): Promise<WorkspaceEntry> {
  if (file.type.startsWith('image/')) {
    const dataUrl = await readAsDataUrl(file);
    return {
      path: file.name,
      kind: 'file',
      media_type: file.type,
      size: file.size,
      content: dataUrl,
    };
  }
  const content = await file.text();
  return {
    path: file.name,
    kind: 'file',
    media_type: file.type || 'text/plain',
    size: file.size,
    content,
  };
}

function readAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result ?? ''));
    reader.onerror = () => reject(reader.error ?? new Error('Failed to read file'));
    reader.readAsDataURL(file);
  });
}

export function WorkbenchSurface() {
  const [code, setCode] = useState('');
  const [files, setFiles] = useState<WorkspaceEntry[]>([]);
  const [result, setResult] = useState<StructuredResult | null>(null);
  const [resultJson, setResultJson] = useState('');
  const [error, setError] = useState<string | null>(null);

  const loadResult = (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    try {
      const parsed: unknown = JSON.parse(resultJson);
      setResult(parseStructuredResult(parsed));
    } catch (err) {
      setResult(null);
      setError(err instanceof Error ? err.message : 'Invalid structured result JSON');
    }
  };

  const addFiles = (list: FileList | null) => {
    if (!list || list.length === 0) {
      return;
    }
    void Promise.all(Array.from(list).map(entryFromFile))
      .then((entries) => {
        setFiles((current) => [...current, ...entries]);
        setError(null);
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : 'Failed to read workspace files');
      });
  };

  return (
    <div className="mx-auto max-w-7xl space-y-6" data-testid="workbench">
      <div>
        <h1 className="text-3xl font-black tracking-tight">Workbench</h1>
        <p className="mt-2 text-textSecondary">
          Generic lab studio: editor, workspace tree, and renderers for structured result blocks.
        </p>
      </div>

      {error ? (
        <StatusBanner tone="error" title="Unable to load workbench input">
          {error}
        </StatusBanner>
      ) : null}

      <Workbench code={code} onCodeChange={setCode} files={files} result={result} />

      <Panel
        title="Load structured result"
        description="Paste WP-137 JSON from a local execution. Workspace files come from the file picker or artifact blocks."
      >
        <form className="space-y-4" onSubmit={loadResult}>
          <div>
            <label className="mb-1 block text-sm font-medium" htmlFor="structured-result-json">
              Structured result JSON
            </label>
            <textarea
              id="structured-result-json"
              className="min-h-32 w-full rounded-md border border-border bg-bg px-3 py-2 font-mono text-sm"
              value={resultJson}
              onChange={(event) => setResultJson(event.target.value)}
              spellCheck={false}
            />
          </div>
          <Button type="submit" disabled={!resultJson.trim()}>
            Render blocks
          </Button>
        </form>
        <div className="mt-4">
          <label className="mb-1 block text-sm font-medium" htmlFor="workspace-files">
            Workspace files
          </label>
          <input
            id="workspace-files"
            type="file"
            multiple
            onChange={(event) => {
              addFiles(event.target.files);
              event.target.value = '';
            }}
          />
        </div>
      </Panel>
    </div>
  );
}
