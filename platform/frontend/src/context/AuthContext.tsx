import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';
import {
  bootstrap,
  createLearner as createLearnerRequest,
  getLearner,
  isApiError,
  setAuthToken,
} from '../api/client';
import type { Learner } from '../api/types';

const TOKEN_STORAGE_KEY = 'learningos.loopbackToken';
const LEARNER_STORAGE_KEY = 'learningos.learner';

export type AuthStatus = 'bootstrapping' | 'ready' | 'error';

type AuthContextValue = {
  status: AuthStatus;
  token: string | null;
  learner: Learner | null;
  errorMessage: string | null;
  createLearner: (username: string, displayName?: string) => Promise<Learner>;
  logout: () => void;
  retry: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);

function readStoredLearner(): Learner | null {
  const raw = sessionStorage.getItem(LEARNER_STORAGE_KEY);
  if (!raw) {
    return null;
  }
  try {
    const parsed = JSON.parse(raw) as Learner;
    if (parsed && typeof parsed.id === 'string') {
      return parsed;
    }
  } catch {
    sessionStorage.removeItem(LEARNER_STORAGE_KEY);
  }
  return null;
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>('bootstrapping');
  const [token, setToken] = useState<string | null>(null);
  const [learner, setLearner] = useState<Learner | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const connect = useCallback(async () => {
    try {
      const result = await bootstrap();
      setAuthToken(result.token);
      sessionStorage.setItem(TOKEN_STORAGE_KEY, result.token);
      setToken(result.token);

      const stored = readStoredLearner();
      if (stored?.id) {
        try {
          const fresh = await getLearner(stored.id);
          setLearner(fresh);
          sessionStorage.setItem(LEARNER_STORAGE_KEY, JSON.stringify(fresh));
        } catch {
          sessionStorage.removeItem(LEARNER_STORAGE_KEY);
          setLearner(null);
        }
      } else {
        setLearner(null);
      }
      setStatus('ready');
    } catch (err) {
      setAuthToken(null);
      setToken(null);
      setLearner(null);
      setErrorMessage(isApiError(err) ? err.message : 'Failed to bootstrap local session');
      setStatus('error');
    }
  }, []);

  useEffect(() => {
    void connect();
  }, [connect]);

  const createLearner = useCallback(async (username: string, displayName?: string) => {
    const created = await createLearnerRequest({
      username,
      display_name: displayName,
    });
    setLearner(created);
    sessionStorage.setItem(LEARNER_STORAGE_KEY, JSON.stringify(created));
    return created;
  }, []);

  const logout = useCallback(() => {
    setLearner(null);
    sessionStorage.removeItem(LEARNER_STORAGE_KEY);
  }, []);

  const retry = useCallback(async () => {
    setStatus('bootstrapping');
    setErrorMessage(null);
    await connect();
  }, [connect]);

  const value = useMemo<AuthContextValue>(
    () => ({
      status,
      token,
      learner,
      errorMessage,
      createLearner,
      logout,
      retry,
    }),
    [status, token, learner, errorMessage, createLearner, logout, retry],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
}
