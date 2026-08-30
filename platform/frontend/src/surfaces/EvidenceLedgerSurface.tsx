import React, { useEffect, useState } from 'react';
import { apiClient } from '../api/client';
import { useAuth } from '../context/AuthContext';

export const EvidenceLedgerSurface: React.FC = () => {
  const [evidence, setEvidence] = useState<any[]>([]);
  const { learnerId } = useAuth();

  useEffect(() => {
    if (!learnerId) return;
    apiClient.getEvidence(learnerId)
      .then(data => setEvidence(data.evidence || []))
      .catch(console.error);
  }, [learnerId]);

  return (
    <div className="p-8">
      <h1 className="text-3xl font-bold mb-4">Evidence Ledger</h1>
      <p className="text-textSecondary mb-8">Immutable cryptographic log of executed artifacts and code traces.</p>
      
      <div className="bg-slate-800 rounded border border-slate-700 overflow-hidden">
        <table className="w-full text-left">
          <thead className="bg-slate-900 text-slate-300 text-sm font-semibold">
            <tr>
              <th className="px-6 py-3 border-b border-slate-700">Timestamp</th>
              <th className="px-6 py-3 border-b border-slate-700">Mission</th>
              <th className="px-6 py-3 border-b border-slate-700">Competency</th>
              <th className="px-6 py-3 border-b border-slate-700">Type</th>
              <th className="px-6 py-3 border-b border-slate-700">Hash</th>
            </tr>
          </thead>
          <tbody>
            {evidence.map(item => (
              <tr key={item.id} className="border-b border-slate-700 hover:bg-slate-700/50">
                <td className="px-6 py-4 text-sm text-slate-400">{new Date(item.created_at).toLocaleString()}</td>
                <td className="px-6 py-4 text-sm font-medium">{item.mission_id}</td>
                <td className="px-6 py-4 text-sm font-mono text-primary">{item.competency_id}</td>
                <td className="px-6 py-4 text-sm">{item.artifact_type}</td>
                <td className="px-6 py-4 text-xs font-mono text-slate-500">{item.artifact_hash.substring(0, 16)}...</td>
              </tr>
            ))}
            {evidence.length === 0 && (
              <tr>
                <td colSpan={5} className="px-6 py-8 text-center text-slate-500">No evidence recorded yet.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
