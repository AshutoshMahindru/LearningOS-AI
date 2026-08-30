import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { apiClient } from '../api/client';

export const DashboardSurface: React.FC = () => {
  const [missions, setMissions] = useState<any[]>([]);

  useEffect(() => {
    apiClient.getMissions()
      .then(data => setMissions(data.missions || []))
      .catch(console.error);
  }, []);

  return (
    <div className="p-10 max-w-6xl mx-auto">
      <div className="mb-12">
        <h1 className="text-4xl font-black mb-3 tracking-tight">Dashboard</h1>
        <p className="text-slate-400 text-lg">Current mission status and active flagship version progress.</p>
      </div>
      
      <h2 className="text-2xl font-bold mb-6 flex items-center">
        <span className="w-8 h-1 bg-indigo-500 mr-4 rounded-full"></span>
        Available Missions
      </h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {missions.map(m => (
          <Link to={`/missions/${m.id}`} key={m.id} className="block p-6 glass-card rounded-xl group relative overflow-hidden">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-indigo-500 to-cyan-400 transform scale-x-0 group-hover:scale-x-100 transition-transform origin-left duration-300"></div>
            <div className="text-xs text-slate-400 mb-2 font-mono">{m.id} • {m.phase_id}</div>
            <h3 className="text-xl font-bold text-slate-100 group-hover:text-white transition-colors">{m.title}</h3>
            <div className="mt-6 flex justify-end">
              <span className="text-sm font-medium text-indigo-400 group-hover:text-indigo-300 transition-colors flex items-center">
                Launch <span className="ml-1 group-hover:translate-x-1 transition-transform">→</span>
              </span>
            </div>
          </Link>
        ))}
        {missions.length === 0 && <p className="text-textSecondary">No missions loaded.</p>}
      </div>
    </div>
  );
};
