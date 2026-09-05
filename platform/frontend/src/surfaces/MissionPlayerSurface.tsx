import { type KeyboardEvent } from 'react';
import { Link } from 'react-router-dom';
import { Button } from '../components/Button';
import { LearnerErrorBanner } from '../components/LearnerErrorBanner';
import { PageStatus } from '../components/PageStatus';
import { cn } from '../cn';
import { StageView } from '../player/StageView';
import { useMissionPlayer } from '../player/useMissionPlayer';

export function MissionPlayerSurface() {
  const player = useMissionPlayer();
  const {
    sessionId,
    missionQuery,
    session,
    mission,
    currentStage,
    currentStageId,
    loading,
    loadError,
    actionError,
    busy,
    enterResult,
    predictResult,
    executeResult,
    submitResult,
    gateResult,
    selectStage,
    onPredict,
    onExecute,
    onSubmit,
    onEvaluateGate,
  } = player;

  const stages = mission?.stages ?? [];

  const onStageNavKey = (event: KeyboardEvent<HTMLButtonElement>, stageId: string) => {
    if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp' && event.key !== 'Home' && event.key !== 'End') {
      return;
    }
    event.preventDefault();
    const index = stages.findIndex((stage) => stage.id === stageId);
    if (index < 0) {
      return;
    }
    let nextIndex = index;
    if (event.key === 'ArrowDown') {
      nextIndex = Math.min(stages.length - 1, index + 1);
    } else if (event.key === 'ArrowUp') {
      nextIndex = Math.max(0, index - 1);
    } else if (event.key === 'Home') {
      nextIndex = 0;
    } else {
      nextIndex = stages.length - 1;
    }
    const next = stages[nextIndex];
    if (!next) {
      return;
    }
    selectStage(next.id);
    const nav = event.currentTarget.closest('nav');
    const buttons = nav?.querySelectorAll<HTMLButtonElement>('button[data-stage-id]');
    buttons?.[nextIndex]?.focus();
  };

  const empty =
    !loading && !loadError && !session
      ? {
          title: 'No session selected',
          message: 'Start a mission from the dashboard or resume with a session id.',
          action: (
            <Link className="text-primary underline" to="/">
              Back to dashboard
            </Link>
          ),
        }
      : !loading && !loadError && mission && stages.length === 0
        ? {
            title: 'No stages in this mission',
            message: 'The loaded specification does not define any stages.',
          }
        : null;

  return (
    <div className="mx-auto max-w-6xl space-y-6" data-testid="mission-player">
      <div>
        <h1 id="player-heading" className="text-3xl font-black tracking-tight">
          {mission?.title || 'Mission player'}
        </h1>
        <p className="mt-2 text-textSecondary">
          Generic stage runtime driven by the loaded mission specification.
        </p>
        {session ? (
          <p className="mt-1 font-mono text-xs text-textSecondary">
            session {session.session_id} · mission {session.mission_id}
            {currentStageId ? ` · stage ${currentStageId}` : ''}
          </p>
        ) : null}
      </div>

      <PageStatus
        loading={loading}
        loadingLabel={missionQuery && !sessionId ? 'Starting session' : 'Loading mission'}
        error={loadError}
        empty={empty}
      >
        {mission && session ? (
          <div className="grid gap-6 lg:grid-cols-[16rem_1fr]">
            <nav aria-label="Stages">
              <ol className="space-y-1">
                {stages.map((stage) => {
                  const active = stage.id === currentStageId;
                  return (
                    <li key={stage.id}>
                      <button
                        type="button"
                        data-stage-id={stage.id}
                        className={cn(
                          'w-full rounded-md px-3 py-2 text-left text-sm font-semibold',
                          active ? 'bg-elevated text-primary' : 'hover:bg-elevated',
                        )}
                        aria-current={active ? 'step' : undefined}
                        aria-label={`${stage.title} (${stage.type})`}
                        onClick={() => selectStage(stage.id)}
                        onKeyDown={(event) => onStageNavKey(event, stage.id)}
                      >
                        <span className="block">{stage.title}</span>
                        <span className="block font-mono text-xs font-normal text-textSecondary">{stage.type}</span>
                      </button>
                    </li>
                  );
                })}
              </ol>
            </nav>
            <div className="space-y-4" role="region" aria-label="Current stage">
              {actionError ? <LearnerErrorBanner error={actionError} /> : null}
              {currentStage ? (
                <StageView
                  mission={mission}
                  session={session}
                  stage={currentStage}
                  enterResult={enterResult}
                  predictResult={predictResult}
                  executeResult={executeResult}
                  submitResult={submitResult}
                  gateResult={gateResult}
                  busy={busy}
                  onPredict={onPredict}
                  onExecute={onExecute}
                  onSubmit={onSubmit}
                  onEvaluateGate={onEvaluateGate}
                />
              ) : (
                <PageStatus
                  empty={{
                    title: 'Stage unavailable',
                    message: 'The current stage id is not present in the specification.',
                    action: (
                      <Button variant="secondary" onClick={() => stages[0] && selectStage(stages[0].id)}>
                        Open first stage
                      </Button>
                    ),
                  }}
                />
              )}
            </div>
          </div>
        ) : null}
      </PageStatus>
    </div>
  );
}
