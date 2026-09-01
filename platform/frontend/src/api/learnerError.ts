import { isApiError } from './client';

export type LearnerFacingError = {
  title: string;
  message: string;
  code: string;
};

const TITLE_BY_CODE: Record<string, string> = {
  UNAUTHORIZED: 'Sign-in required',
  NOT_FOUND: 'Not found',
  VALIDATION_ERROR: 'Check your input',
  WORKER_UNAVAILABLE: 'Execution worker unavailable',
  G3_QUARANTINED: 'This action is not available yet',
  QUARANTINED: 'This action is not available yet',
  TUTOR_UNAVAILABLE: 'Tutor is not available',
  CURRICULUM_UNAVAILABLE: 'Curriculum is not available',
  STORAGE_UNAVAILABLE: 'Storage is not available',
  CONFLICT: 'Could not save that change',
  INTERNAL: 'Something went wrong',
};

export function learnerFacingError(error: unknown): LearnerFacingError {
  if (isApiError(error)) {
    return {
      title: TITLE_BY_CODE[error.code] ?? 'Request failed',
      message: error.message,
      code: error.code,
    };
  }
  if (error instanceof Error && error.message) {
    return {
      title: 'Request failed',
      message: error.message,
      code: 'INTERNAL',
    };
  }
  return {
    title: 'Request failed',
    message: 'Unexpected error',
    code: 'INTERNAL',
  };
}
