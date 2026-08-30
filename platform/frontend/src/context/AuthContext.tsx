import React, { createContext, useContext, useState, useEffect } from 'react';

interface AuthContextType {
  learnerId: string | null;
  login: (id: string) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType>({
  learnerId: null,
  login: () => {},
  logout: () => {},
});

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [learnerId, setLearnerId] = useState<string | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    const storedId = localStorage.getItem('learningos_learner_id');
    if (storedId) {
      setLearnerId(storedId);
    }
    setIsLoaded(true);
  }, []);

  const login = (id: string) => {
    localStorage.setItem('learningos_learner_id', id);
    setLearnerId(id);
  };

  const logout = () => {
    localStorage.removeItem('learningos_learner_id');
    setLearnerId(null);
  };

  if (!isLoaded) return null; // Avoid flicker

  return (
    <AuthContext.Provider value={{ learnerId, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
