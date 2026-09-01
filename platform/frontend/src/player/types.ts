import type {
  ExecuteStageRequest,
  ExecuteStageResponse,
  GateEvaluateResponse,
  Mission,
  MissionStage,
  PredictCommitRequest,
  PredictCommitResponse,
  Session,
  StageEnterResponse,
  SubmitStageRequest,
  SubmitStageResponse,
} from '../api/types';

export type StageRenderProps = {
  mission: Mission;
  session: Session;
  stage: MissionStage;
  enterResult: StageEnterResponse | null;
  predictResult: PredictCommitResponse | null;
  executeResult: ExecuteStageResponse | null;
  submitResult: SubmitStageResponse | null;
  gateResult: GateEvaluateResponse | null;
  busy: string | null;
  onPredict: (body: PredictCommitRequest) => void;
  onExecute: (body: ExecuteStageRequest) => void;
  onSubmit: (body: SubmitStageRequest) => void;
  onEvaluateGate: () => void;
};
