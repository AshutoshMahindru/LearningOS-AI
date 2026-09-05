import { useId, useState, type FormEvent } from 'react';
import { Button } from '../components/Button';
import { StatusBanner } from '../components/StatusBanner';
import { BlockList } from '../workbench/BlockRenderer';
import { StageFrame } from './StageFrame';
import type { StageRenderProps } from './types';

function parseObject(raw: string, label: string): Record<string, unknown> {
  const trimmed = raw.trim();
  if (!trimmed) {
    return {};
  }
  const parsed: unknown = JSON.parse(trimmed);
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(`${label} must be a JSON object`);
  }
  return parsed as Record<string, unknown>;
}

function ResultBlock({ title, value }: { title: string; value: unknown }) {
  if (value == null) {
    return null;
  }
  return (
    <section className="rounded-md border border-border bg-bg p-3">
      <h3 className="text-sm font-semibold">{title}</h3>
      <pre className="mt-2 overflow-auto text-xs">{JSON.stringify(value, null, 2)}</pre>
    </section>
  );
}

function NotesForm({
  id,
  label,
  submitLabel,
  busy,
  disabled,
  onSubmit,
}: {
  id?: string;
  label: string;
  submitLabel: string;
  busy: boolean;
  disabled?: boolean;
  onSubmit: (explanation: string) => void;
}) {
  const generatedId = useId();
  const fieldId = id ?? `stage-notes-${generatedId}`;
  const [notes, setNotes] = useState('');
  return (
    <form
      className="space-y-3"
      onSubmit={(event: FormEvent) => {
        event.preventDefault();
        onSubmit(notes.trim());
      }}
    >
      <label className="block text-sm font-medium" htmlFor={fieldId}>
        {label}
      </label>
      <textarea
        id={fieldId}
        className="min-h-32 w-full rounded-md border border-border bg-bg px-3 py-2 text-textPrimary"
        value={notes}
        onChange={(event) => setNotes(event.target.value)}
      />
      <Button type="submit" disabled={disabled || busy || !notes.trim()}>
        {busy ? 'Submitting…' : submitLabel}
      </Button>
    </form>
  );
}

function EvidenceStage(props: StageRenderProps & { notesLabel: string }) {
  const { stage, notesLabel, busy, onSubmit, submitResult } = props;
  return (
    <StageFrame stage={stage}>
      <NotesForm
        id={`${stage.id}-notes`}
        label={notesLabel}
        submitLabel="Submit stage"
        busy={busy === 'submit'}
        onSubmit={(explanation) => onSubmit({ explanation })}
      />
      <div className="mt-4">
        <ResultBlock title="Submit result" value={submitResult} />
      </div>
    </StageFrame>
  );
}

export function OrientationStage(props: StageRenderProps) {
  const invariant = props.mission.core_invariant;
  const competencies = props.mission.competencies ?? [];
  return (
    <StageFrame stage={props.stage}>
      {invariant ? (
        <p className="mb-4 rounded-md border border-border bg-elevated/50 p-3">
          <span className="block text-xs font-semibold uppercase tracking-widest text-textSecondary">
            Core invariant
          </span>
          {invariant}
        </p>
      ) : null}
      {competencies.length > 0 ? (
        <ul className="mb-4 list-disc space-y-1 pl-5 text-sm text-textSecondary">
          {competencies.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      ) : null}
      <NotesForm
        id={`${props.stage.id}-notes`}
        label="Orientation notes"
        submitLabel="Submit stage"
        busy={props.busy === 'submit'}
        onSubmit={(explanation) => props.onSubmit({ explanation })}
      />
      <div className="mt-4">
        <ResultBlock title="Submit result" value={props.submitResult} />
      </div>
    </StageFrame>
  );
}

export function TraceMapStage(props: StageRenderProps) {
  return <EvidenceStage {...props} notesLabel="Component and data-flow map" />;
}

export function InterrogateStage(props: StageRenderProps) {
  return <EvidenceStage {...props} notesLabel="Interrogation notes" />;
}

export function CodeReadingStage(props: StageRenderProps) {
  return <EvidenceStage {...props} notesLabel="Code-reading notes" />;
}

export function RebuildDebugStage(props: StageRenderProps) {
  return <EvidenceStage {...props} notesLabel="Rebuild and debug notes" />;
}

export function ControlledFailureStage(props: StageRenderProps) {
  return <EvidenceStage {...props} notesLabel="Failure isolation notes" />;
}

export function TransferAssessmentStage(props: StageRenderProps) {
  return (
    <StageFrame stage={props.stage}>
      <StatusBanner tone="warning" title="No-AI transfer" className="mb-4">
        Assistance is disabled for this stage. Submit unassisted evidence.
      </StatusBanner>
      <NotesForm
        id={`${props.stage.id}-notes`}
        label="Transfer solution"
        submitLabel="Submit transfer"
        busy={props.busy === 'submit'}
        onSubmit={(explanation) => props.onSubmit({ explanation })}
      />
      <div className="mt-4">
        <ResultBlock title="Submit result" value={props.submitResult} />
      </div>
    </StageFrame>
  );
}

export function FlagshipIntegrationStage(props: StageRenderProps) {
  return <EvidenceStage {...props} notesLabel="Flagship integration notes" />;
}

export function ReflectionAdrStage(props: StageRenderProps) {
  const [title, setTitle] = useState('');
  const [context, setContext] = useState('');
  const [decision, setDecision] = useState('');
  const [consequences, setConsequences] = useState('');
  const busy = props.busy === 'submit';

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault();
    props.onSubmit({
      explanation: [
        `Title: ${title.trim()}`,
        `Context: ${context.trim()}`,
        `Decision: ${decision.trim()}`,
        `Consequences: ${consequences.trim()}`,
      ].join('\n\n'),
    });
  };

  return (
    <StageFrame stage={props.stage}>
      <form className="grid gap-3" onSubmit={handleSubmit}>
        <label className="block text-sm font-medium" htmlFor="adr-title">
          Title
        </label>
        <input
          id="adr-title"
          className="w-full rounded-md border border-border bg-bg px-3 py-2 text-textPrimary"
          value={title}
          onChange={(event) => setTitle(event.target.value)}
        />
        <label className="block text-sm font-medium" htmlFor="adr-context">
          Context
        </label>
        <textarea
          id="adr-context"
          className="min-h-24 w-full rounded-md border border-border bg-bg px-3 py-2 text-textPrimary"
          value={context}
          onChange={(event) => setContext(event.target.value)}
        />
        <label className="block text-sm font-medium" htmlFor="adr-decision">
          Decision
        </label>
        <textarea
          id="adr-decision"
          className="min-h-24 w-full rounded-md border border-border bg-bg px-3 py-2 text-textPrimary"
          value={decision}
          onChange={(event) => setDecision(event.target.value)}
        />
        <label className="block text-sm font-medium" htmlFor="adr-consequences">
          Consequences
        </label>
        <textarea
          id="adr-consequences"
          className="min-h-24 w-full rounded-md border border-border bg-bg px-3 py-2 text-textPrimary"
          value={consequences}
          onChange={(event) => setConsequences(event.target.value)}
        />
        <Button type="submit" disabled={busy || !title.trim() || !decision.trim()}>
          {busy ? 'Submitting…' : 'Submit ADR'}
        </Button>
      </form>
      <div className="mt-4">
        <ResultBlock title="Submit result" value={props.submitResult} />
      </div>
    </StageFrame>
  );
}

export function ExperimentStage(props: StageRenderProps) {
  const [hypothesis, setHypothesis] = useState('');
  const [expectedRaw, setExpectedRaw] = useState('{}');
  const [code, setCode] = useState('');
  const [parametersRaw, setParametersRaw] = useState('{}');
  const [explanation, setExplanation] = useState('');
  const [localError, setLocalError] = useState<string | null>(null);
  const sealed = Boolean(props.predictResult) || props.enterResult?.prediction_sealed === true;

  const commitPrediction = (event: FormEvent) => {
    event.preventDefault();
    setLocalError(null);
    try {
      props.onPredict({
        hypothesis: hypothesis.trim(),
        expected_values: parseObject(expectedRaw, 'Expected values'),
      });
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : 'Invalid expected values');
    }
  };

  const runExecute = (event: FormEvent) => {
    event.preventDefault();
    setLocalError(null);
    try {
      props.onExecute({
        code: code.trim() || undefined,
        parameters: parseObject(parametersRaw, 'Parameters'),
      });
    } catch (error) {
      setLocalError(error instanceof Error ? error.message : 'Invalid parameters');
    }
  };

  const runSubmit = (event: FormEvent) => {
    event.preventDefault();
    props.onSubmit({ explanation: explanation.trim() });
  };

  return (
    <StageFrame stage={props.stage}>
      <ol className="mb-4 list-decimal space-y-1 pl-5 text-sm text-textSecondary">
        <li>Predict</li>
        <li>Commit</li>
        <li>Run</li>
        <li>Observe</li>
        <li>Explain</li>
      </ol>
      {localError ? (
        <StatusBanner tone="error" title="Check your input" className="mb-4">
          {localError}
        </StatusBanner>
      ) : null}

      <form className="space-y-3" onSubmit={commitPrediction}>
        <label className="block text-sm font-medium" htmlFor="hypothesis">
          Hypothesis
        </label>
        <textarea
          id="hypothesis"
          className="min-h-24 w-full rounded-md border border-border bg-bg px-3 py-2 text-textPrimary"
          value={hypothesis}
          onChange={(event) => setHypothesis(event.target.value)}
          disabled={sealed}
        />
        <label className="block text-sm font-medium" htmlFor="expected-values">
          Expected values (JSON object)
        </label>
        <textarea
          id="expected-values"
          className="min-h-20 w-full rounded-md border border-border bg-bg px-3 py-2 font-mono text-sm text-textPrimary"
          value={expectedRaw}
          onChange={(event) => setExpectedRaw(event.target.value)}
          disabled={sealed}
        />
        <Button type="submit" disabled={sealed || props.busy === 'predict' || !hypothesis.trim()}>
          {sealed ? 'Prediction sealed' : props.busy === 'predict' ? 'Sealing…' : 'Commit prediction'}
        </Button>
      </form>
      <div className="mt-3">
        <ResultBlock title="Prediction seal" value={props.predictResult} />
      </div>

      <form className="mt-6 space-y-3" onSubmit={runExecute}>
        <label className="block text-sm font-medium" htmlFor="experiment-code">
          Code (optional)
        </label>
        <textarea
          id="experiment-code"
          className="min-h-24 w-full rounded-md border border-border bg-bg px-3 py-2 font-mono text-sm text-textPrimary"
          value={code}
          onChange={(event) => setCode(event.target.value)}
          disabled={!sealed}
        />
        <label className="block text-sm font-medium" htmlFor="experiment-parameters">
          Parameters (JSON object)
        </label>
        <textarea
          id="experiment-parameters"
          className="min-h-20 w-full rounded-md border border-border bg-bg px-3 py-2 font-mono text-sm text-textPrimary"
          value={parametersRaw}
          onChange={(event) => setParametersRaw(event.target.value)}
          disabled={!sealed}
        />
        <Button type="submit" disabled={!sealed || props.busy === 'execute'}>
          {props.busy === 'execute' ? 'Running…' : 'Run experiment'}
        </Button>
      </form>
      <div className="mt-3 space-y-3">
        {props.executeResult ? (
          <ResultBlock
            title="Observation"
            value={{
              status: props.executeResult.status,
              execution_id: props.executeResult.execution_id,
              duration_ms: props.executeResult.duration_ms,
            }}
          />
        ) : null}
        {props.executeResult ? <BlockList blocks={props.executeResult.blocks ?? []} /> : null}
      </div>

      <form className="mt-6 space-y-3" onSubmit={runSubmit}>
        <label className="block text-sm font-medium" htmlFor="experiment-explain">
          Explain the delta
        </label>
        <textarea
          id="experiment-explain"
          className="min-h-24 w-full rounded-md border border-border bg-bg px-3 py-2 text-textPrimary"
          value={explanation}
          onChange={(event) => setExplanation(event.target.value)}
        />
        <Button type="submit" disabled={props.busy === 'submit' || !explanation.trim() || !props.executeResult}>
          {props.busy === 'submit' ? 'Submitting…' : 'Submit explanation'}
        </Button>
      </form>
      <div className="mt-3">
        <ResultBlock title="Submit result" value={props.submitResult} />
      </div>
    </StageFrame>
  );
}

export function CompetencyGateStage(props: StageRenderProps) {
  const contract = props.mission.gate_contract;
  return (
    <StageFrame stage={props.stage}>
      {contract?.pass_threshold != null ? (
        <p className="mb-3 text-sm text-textSecondary">Pass threshold: {contract.pass_threshold}</p>
      ) : null}
      {contract?.required_evidence && contract.required_evidence.length > 0 ? (
        <ul className="mb-4 list-disc space-y-1 pl-5 text-sm text-textSecondary">
          {contract.required_evidence.map((item, index) => (
            <li key={`${item.stage_id ?? 'evidence'}-${index}`}>
              {item.competency_id || 'competency'} · {item.stage_id || 'stage'} · {item.artifact_type || 'artifact'}
            </li>
          ))}
        </ul>
      ) : null}
      <Button onClick={() => props.onEvaluateGate()} disabled={props.busy === 'gate'}>
        {props.busy === 'gate' ? 'Evaluating…' : 'Evaluate gate'}
      </Button>
      <div className="mt-4">
        <ResultBlock title="Gate result" value={props.gateResult} />
      </div>
    </StageFrame>
  );
}

export function FallbackStage(props: StageRenderProps) {
  return <EvidenceStage {...props} notesLabel="Stage notes" />;
}
