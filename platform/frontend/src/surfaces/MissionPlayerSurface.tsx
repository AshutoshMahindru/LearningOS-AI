import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { apiClient } from '../api/client';
import { StageRegistry } from '../stages/StageRegistry';
import { SocraticDrawer } from '../components/SocraticDrawer';

export const MissionPlayerSurface: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [mission, setMission] = useState<any>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [isTutorOpen, setIsTutorOpen] = useState(false);

  useEffect(() => {
    if (id) {
      // Fetch schema and create session concurrently
      Promise.all([
        apiClient.getMission(id),
        apiClient.createSession(id)
      ])
      .then(([missionData, sessionData]) => {
        setMission(missionData);
        setSessionId(sessionData.session_id);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
    }
  }, [id]);

  if (loading) return <div className="p-8">Loading mission {id}...</div>;
  if (!mission || !sessionId) return <div className="p-8 text-red-500">Mission {id} failed to initialize.</div>;

  return (
    <div className="relative min-h-screen">
      <div className="p-8 max-w-4xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold mb-2">{mission.title}</h1>
            <p className="text-textSecondary">{mission.description}</p>
          </div>
          <button 
            onClick={() => setIsTutorOpen(true)}
            className="flex items-center px-4 py-2 bg-slate-800 hover:bg-slate-700 border border-slate-600 rounded text-primary transition-colors shadow-lg"
          >
            <span className="mr-2">🦉</span> Ask Tutor
          </button>
        </div>
        
        <div className="space-y-12">
        {mission.stages?.map((stage: any, index: number) => (
          <div key={stage.id} className="relative pl-8 border-l-2 border-slate-700">
            <div className="absolute w-4 h-4 bg-slate-800 border-2 border-primary rounded-full -left-[9px] top-4"></div>
            <h3 className="text-xl font-semibold mb-1 text-slate-200">
              {index + 1}. {stage.title}
            </h3>
            <p className="text-sm text-textSecondary mb-4 italic">{stage.instructions}</p>
            <StageRegistry type={stage.type} sessionId={sessionId} stageId={stage.id} />
          </div>
        ))}
      </div>
      </div>
      
      <SocraticDrawer 
        isOpen={isTutorOpen} 
        onClose={() => setIsTutorOpen(false)} 
        sessionId={sessionId}
        stageId={mission.stages[0]?.id || "unknown"} 
      />
    </div>
  );
};
