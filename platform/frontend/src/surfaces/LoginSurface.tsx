import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';

export const LoginSurface: React.FC = () => {
  const [username, setUsername] = useState('');
  const { login } = useAuth();

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    if (username.trim()) {
      login(username.trim().toLowerCase());
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background text-textPrimary p-4">
      <div className="max-w-md w-full bg-slate-800 border border-slate-700 rounded-lg p-8 shadow-2xl">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-primary mb-2">LearningOS V3</h1>
          <p className="text-slate-400">Identify yourself to access the platform.</p>
        </div>

        <form onSubmit={handleLogin} className="space-y-6">
          <div>
            <label htmlFor="username" className="block text-sm font-medium text-slate-300 mb-2">
              Learner ID
            </label>
            <input
              id="username"
              type="text"
              className="w-full bg-slate-900 border border-slate-600 rounded px-4 py-3 text-white focus:outline-none focus:border-primary transition-colors"
              placeholder="e.g. learner_default"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoFocus
            />
            <p className="text-xs text-slate-500 mt-2">
              Use "learner_default" to access the pre-seeded demo environment.
            </p>
          </div>

          <button
            type="submit"
            disabled={!username.trim()}
            className="w-full bg-primary hover:bg-blue-600 text-white font-bold py-3 px-4 rounded transition-colors disabled:opacity-50"
          >
            Enter Mission Control
          </button>
        </form>
      </div>
    </div>
  );
};
