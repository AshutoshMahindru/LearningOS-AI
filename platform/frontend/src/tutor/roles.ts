export const TUTOR_ROLES = [
  { id: 'NAVIGATOR', label: 'Navigator' },
  { id: 'SOCRATIC', label: 'Socratic' },
  { id: 'DEBUGGER', label: 'Debugger' },
  { id: 'FEYNMAN', label: 'Feynman' },
] as const;

export type TutorRoleId = (typeof TUTOR_ROLES)[number]['id'];
