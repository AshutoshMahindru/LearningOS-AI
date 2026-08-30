const API_BASE = 'http://127.0.0.1:8765/api/v1';

export const apiClient = {
  getHealth: async () => {
    const response = await fetch(`${API_BASE}/system/health`);
    if (!response.ok) throw new Error('Network response was not ok');
    return response.json();
  },
  
  getMissions: async () => {
    const response = await fetch(`${API_BASE}/missions`);
    if (!response.ok) throw new Error('Network response was not ok');
    return response.json();
  },
  
  getMission: async (id: string) => {
    const response = await fetch(`${API_BASE}/missions/${id}`);
    if (!response.ok) throw new Error('Network response was not ok');
    return response.json();
  },

  createSession: async (missionId: string) => {
    const response = await fetch(`${API_BASE}/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mission_id: missionId }),
    });
    if (!response.ok) throw new Error('Failed to create session');
    return response.json();
  },

  predictStage: async (sessionId: string, stageId: string, payload: { hypothesis: string; expected_values: Record<string, any> }) => {
    const response = await fetch(`${API_BASE}/sessions/${sessionId}/stages/${stageId}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error('Failed to commit prediction');
    return response.json();
  },

  executeStage: async (sessionId: string, stageId: string, payload: { code: string; parameters: Record<string, any> }) => {
    const response = await fetch(`${API_BASE}/sessions/${sessionId}/stages/${stageId}/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error('Failed to execute stage');
    return response.json();
  },

  submitStage: async (sessionId: string, stageId: string, payload: { artifacts: any[]; explanation: string }) => {
    const response = await fetch(`${API_BASE}/sessions/${sessionId}/stages/${stageId}/submit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error('Failed to submit stage');
    return response.json();
  },

  getEvidence: async (learnerId: string) => {
    const response = await fetch(`${API_BASE}/learners/${learnerId}/evidence`);
    if (!response.ok) throw new Error('Failed to fetch evidence');
    return response.json();
  },

  getCompetencies: async (learnerId: string) => {
    const response = await fetch(`${API_BASE}/learners/${learnerId}/competencies`);
    if (!response.ok) throw new Error('Failed to fetch competencies');
    return response.json();
  },

  tutorChat: async (sessionId: string, stageId: string, payload: { role: string; prompt: string }) => {
    const response = await fetch(`${API_BASE}/tutor/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        stage_id: stageId,
        role: payload.role,
        prompt: payload.prompt
      }),
    });
    if (!response.ok) throw new Error('Failed to send message to tutor');
    return response.json();
  },
};
