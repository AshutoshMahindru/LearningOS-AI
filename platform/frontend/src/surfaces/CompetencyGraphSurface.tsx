import React, { useEffect, useState } from 'react';
import { apiClient } from '../api/client';
import { useAuth } from '../context/AuthContext';

export const CompetencyGraphSurface: React.FC = () => {
  const [competencies, setCompetencies] = useState<any[]>([]);
  const { learnerId } = useAuth();

  useEffect(() => {
    if (!learnerId) return;
    apiClient.getCompetencies(learnerId)
      .then(data => setCompetencies(data.competencies || []))
      .catch(console.error);
  }, [learnerId]);

  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold mb-4">Competency Graph</h1>
      <p className="text-textSecondary mb-8">Visual mapping of learner capability derived entirely from cryptographic evidence.</p>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {competencies.map(comp => (
          <div key={comp.competency_id} className="bg-slate-800 p-6 rounded border border-slate-700 relative overflow-hidden">
            <div className="absolute top-0 right-0 p-2 text-xs font-bold bg-primary text-slate-900 rounded-bl">
              Level {comp.level}
            </div>
            <h3 className="text-lg font-mono text-slate-200 mb-2 mt-4">{comp.competency_id}</h3>
            <div className="w-full bg-slate-900 rounded-full h-2.5 mb-2 border border-slate-700">
              <div className="bg-green-500 h-2.5 rounded-full" style={{ width: `${(comp.level / 5) * 100}%` }}></div>
            </div>
            <p className="text-xs text-slate-500 mt-4">Last evaluated: {new Date(comp.last_evaluated_at).toLocaleDateString()}</p>
          </div>
        ))}
        {competencies.length === 0 && (
          <div className="col-span-full p-8 text-center text-slate-500 border border-dashed border-slate-600 rounded">
            No competencies awarded yet. Execute stages to generate evidence.
          </div>
        )}
      </div>
    </div>
  );
};
