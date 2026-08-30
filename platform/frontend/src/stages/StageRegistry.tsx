import React from 'react';
import {
  OrientationStage,
  TraceMapStage,
  InterrogateStage,
  ExperimentStage,
  CodeReadingStage,
  RebuildDebugStage,
  ControlledFailureStage,
  TransferAssessmentStage,
  CompetencyGateStage,
  ReflectionADRStage,
  FlagshipIntegrationStage
} from './components';

interface StageRegistryProps {
  type: string;
  sessionId: string;
  stageId: string;
}

export const StageRegistry: React.FC<StageRegistryProps> = ({ type, sessionId, stageId }) => {
  switch (type) {
    case 'orientation': return <OrientationStage />;
    case 'trace_map': return <TraceMapStage />;
    case 'interrogate': return <InterrogateStage sessionId={sessionId} stageId={stageId} />;
    case 'experiment': return <ExperimentStage sessionId={sessionId} stageId={stageId} />;
    case 'code_reading': return <CodeReadingStage sessionId={sessionId} stageId={stageId} />;
    case 'rebuild_debug': return <RebuildDebugStage />;
    case 'controlled_failure': return <ControlledFailureStage />;
    case 'transfer_assessment': return <TransferAssessmentStage />;
    case 'competency_gate': return <CompetencyGateStage sessionId={sessionId} stageId={stageId} />;
    case 'reflection_adr': return <ReflectionADRStage />;
    case 'flagship_integration': return <FlagshipIntegrationStage />;
    default:
      return <div className="text-red-500">Unknown stage type: {type}</div>;
  }
};
