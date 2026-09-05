import { useState, type FormEvent } from 'react';
import { getArtifact, isApiError, putArtifact } from '../api/client';
import type { ArtifactGetResponse, ArtifactPutResponse } from '../api/types';
import { Button } from '../components/Button';
import { EmptyState } from '../components/EmptyState';
import { Panel } from '../components/Panel';
import { StatusBanner } from '../components/StatusBanner';

function toBase64(bytes: Uint8Array): string {
  let binary = '';
  bytes.forEach((value) => {
    binary += String.fromCharCode(value);
  });
  return btoa(binary);
}

export function ArtifactsSurface() {
  const [file, setFile] = useState<File | null>(null);
  const [putResult, setPutResult] = useState<ArtifactPutResponse | null>(null);
  const [lookupHash, setLookupHash] = useState('');
  const [retrieved, setRetrieved] = useState<ArtifactGetResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const handleUpload = async (event: FormEvent) => {
    event.preventDefault();
    if (!file) {
      setError('Choose a file to store.');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = await putArtifact({
        file,
        filename: file.name,
        media_type: file.type || undefined,
      });
      setPutResult(result);
      setLookupHash(result.artifact_hash);
    } catch (err) {
      setError(isApiError(err) ? err.message : 'Upload failed');
    } finally {
      setBusy(false);
    }
  };

  const handleRetrieve = async (event: FormEvent) => {
    event.preventDefault();
    const hash = lookupHash.trim();
    if (!hash) {
      setError('Enter an artifact hash.');
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = await getArtifact(hash);
      setRetrieved(result);
    } catch (err) {
      setRetrieved(null);
      setError(isApiError(err) ? err.message : 'Retrieve failed');
    } finally {
      setBusy(false);
    }
  };

  const downloadRetrieved = () => {
    if (!retrieved) {
      return;
    }
    const copy = new Uint8Array(retrieved.bytes.byteLength);
    copy.set(retrieved.bytes);
    const blob = new Blob([copy], { type: retrieved.media_type || 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = retrieved.artifact_hash;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div>
        <h1 className="text-3xl font-black tracking-tight">Artifacts</h1>
        <p className="mt-2 text-textSecondary">
          Store content-addressed bytes through the local API and retrieve them by hash.
        </p>
      </div>

      {error ? (
        <StatusBanner tone="error" title="Artifact request failed">
          {error}
        </StatusBanner>
      ) : null}

      <Panel title="Upload" description="POST /api/v1/artifacts">
        <form className="space-y-4" onSubmit={(event) => void handleUpload(event)}>
          <div>
            <label htmlFor="artifact-file" className="mb-1 block text-sm font-medium">
              File
            </label>
            <input
              id="artifact-file"
              name="file"
              type="file"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            />
          </div>
          <Button type="submit" disabled={busy || !file}>
            {busy ? 'Working…' : 'Store artifact'}
          </Button>
        </form>
        {putResult ? (
          <dl className="mt-4 grid gap-2 font-mono text-sm">
            <div>
              <dt className="text-textSecondary">Hash</dt>
              <dd className="break-all">{putResult.artifact_hash}</dd>
            </div>
            <div>
              <dt className="text-textSecondary">Size</dt>
              <dd>{putResult.size}</dd>
            </div>
          </dl>
        ) : (
          <EmptyState
            className="mt-4"
            title="No artifact stored yet"
            message="Choose a local file and store it to see the content hash returned by the API."
          />
        )}
      </Panel>

      <Panel title="Retrieve" description="GET /api/v1/artifacts/{hash}">
        <form className="space-y-4" onSubmit={(event) => void handleRetrieve(event)}>
          <div>
            <label htmlFor="artifact-hash" className="mb-1 block text-sm font-medium">
              Artifact hash
            </label>
            <input
              id="artifact-hash"
              name="artifactHash"
              className="w-full rounded-md border border-border bg-bg px-3 py-2 font-mono text-sm text-textPrimary"
              value={lookupHash}
              onChange={(event) => setLookupHash(event.target.value)}
            />
          </div>
          <Button type="submit" variant="secondary" disabled={busy || !lookupHash.trim()}>
            Retrieve
          </Button>
        </form>
        {retrieved ? (
          <div className="mt-4 space-y-2 text-sm">
            <p>
              <span className="text-textSecondary">Bytes: </span>
              {retrieved.bytes.byteLength}
            </p>
            <p>
              <span className="text-textSecondary">Checksum header: </span>
              <span className="font-mono break-all">{retrieved.checksum || 'none'}</span>
            </p>
            <p className="font-mono break-all text-xs text-textSecondary">
              preview {toBase64(retrieved.bytes.slice(0, 24))}
            </p>
            <Button variant="ghost" onClick={downloadRetrieved}>
              Download bytes
            </Button>
          </div>
        ) : null}
      </Panel>
    </div>
  );
}
