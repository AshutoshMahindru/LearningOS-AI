import { getAuthToken, mapErrorResponse } from '../api/client';
import { API_PREFIX } from '../api/types';
import type { TutorRoleId } from './roles';

export type TutorChatRequest = {
  session_id: string;
  stage_id: string;
  role: TutorRoleId | string;
  prompt: string;
};

export type TutorChatResponse = {
  role: string;
  reply: string;
  provider: string;
  assistance_policy?: string;
  learner?: {
    session_id?: string;
    stage_id?: string;
    stage_type?: string;
    assistance_policy?: string;
    guidance_mode?: string;
  };
};

export async function postTutorChat(body: TutorChatRequest): Promise<TutorChatResponse> {
  const token = getAuthToken();
  const headers = new Headers({
    Accept: 'application/json',
    'Content-Type': 'application/json',
  });
  if (token) {
    headers.set('Authorization', `Bearer ${token}`);
  }
  const response = await fetch(`${API_PREFIX}/tutor/chat`, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      session_id: body.session_id,
      stage_id: body.stage_id,
      role: body.role,
      prompt: body.prompt,
    }),
  });
  if (!response.ok) {
    throw await mapErrorResponse(response);
  }
  return (await response.json()) as TutorChatResponse;
}
