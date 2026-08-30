import React, { useEffect, useState } from 'react';
import { apiClient } from '../api/client';

export const SettingsSurface: React.FC = () => {
  const [health, setHealth] = useState<any>(null);

  useEffect(() => {
    apiClient.getHealth()
      .then(setHealth)
      .catch(console.error);
  }, []);

  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold mb-4">Settings & Diagnostics</h1>
      <p className="text-textSecondary mb-4">Local runtime health and system configuration.</p>
      
      <div className="p-4 bg-slate-800 rounded border border-slate-700">
        <h2 className="text-xl font-bold mb-2">Backend Connection</h2>
        {health ? (
          <pre className="text-sm text-green-400">{JSON.stringify(health, null, 2)}</pre>
        ) : (
          <p className="text-red-400">Offline or connecting...</p>
        )}
      </div>
    </div>
  );
};
