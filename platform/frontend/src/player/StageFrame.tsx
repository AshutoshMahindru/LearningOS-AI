import type { ReactNode } from 'react';
import { Panel } from '../components/Panel';
import type { StageRenderProps } from './types';

type StageFrameProps = Pick<StageRenderProps, 'stage'> & {
  children: ReactNode;
};

export function StageFrame({ stage, children }: StageFrameProps) {
  return (
    <Panel
      title={stage.title}
      description={`${stage.type}${stage.assistance_policy ? ` · ${stage.assistance_policy}` : ''}`}
    >
      <div data-testid="stage-frame" data-stage-type={stage.type} data-stage-id={stage.id}>
        {stage.instructions ? <p className="mb-4 text-textSecondary">{stage.instructions}</p> : null}
        {stage.validation_rubric?.pass_criteria ? (
          <p className="mb-4 text-sm text-textSecondary">Pass criteria: {stage.validation_rubric.pass_criteria}</p>
        ) : null}
        {children}
      </div>
    </Panel>
  );
}
