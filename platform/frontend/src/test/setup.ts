import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach, vi } from 'vitest';
import { clearDiagnostics } from '../api/diagnostics';

afterEach(() => {
  cleanup();
  sessionStorage.clear();
  localStorage.clear();
  clearDiagnostics();
  vi.unstubAllGlobals();
});
