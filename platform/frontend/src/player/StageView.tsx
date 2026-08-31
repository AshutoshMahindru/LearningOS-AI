import {
  CodeReadingStage,
  CompetencyGateStage,
  ControlledFailureStage,
  ExperimentStage,
  FallbackStage,
  FlagshipIntegrationStage,
  InterrogateStage,
  OrientationStage,
  RebuildDebugStage,
  ReflectionAdrStage,
  TraceMapStage,
  TransferAssessmentStage,
} from './stages';
import type { StageRenderProps } from './types';

export function StageView(props: StageRenderProps) {
  switch (props.stage.type) {
    case 'orientation':
      return <OrientationStage {...props} />;
    case 'trace_map':
      return <TraceMapStage {...props} />;
    case 'interrogate':
      return <InterrogateStage {...props} />;
    case 'experiment':
      return <ExperimentStage {...props} />;
    case 'code_reading':
      return <CodeReadingStage {...props} />;
    case 'rebuild_debug':
      return <RebuildDebugStage {...props} />;
    case 'controlled_failure':
      return <ControlledFailureStage {...props} />;
    case 'transfer_assessment':
      return <TransferAssessmentStage {...props} />;
    case 'competency_gate':
      return <CompetencyGateStage {...props} />;
    case 'reflection_adr':
      return <ReflectionAdrStage {...props} />;
    case 'flagship_integration':
      return <FlagshipIntegrationStage {...props} />;
    default:
      return <FallbackStage {...props} />;
  }
}
