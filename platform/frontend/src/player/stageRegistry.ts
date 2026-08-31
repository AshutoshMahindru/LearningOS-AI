import type { ComponentType } from 'react';
import type { CanonicalStageType, StageType } from '../api/types';
import { CANONICAL_STAGE_TYPES } from '../api/types';
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

export type StageRenderer = ComponentType<StageRenderProps>;

export const STAGE_REGISTRY: Record<CanonicalStageType, StageRenderer> = {
  orientation: OrientationStage,
  trace_map: TraceMapStage,
  interrogate: InterrogateStage,
  experiment: ExperimentStage,
  code_reading: CodeReadingStage,
  rebuild_debug: RebuildDebugStage,
  controlled_failure: ControlledFailureStage,
  transfer_assessment: TransferAssessmentStage,
  competency_gate: CompetencyGateStage,
  reflection_adr: ReflectionAdrStage,
  flagship_integration: FlagshipIntegrationStage,
};

export function resolveStageRenderer(type: StageType): StageRenderer {
  if (type in STAGE_REGISTRY) {
    return STAGE_REGISTRY[type as CanonicalStageType];
  }
  return FallbackStage;
}

export function registeredStageTypes(): CanonicalStageType[] {
  return [...CANONICAL_STAGE_TYPES];
}
