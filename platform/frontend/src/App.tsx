import React from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import { LoginSurface } from './surfaces/LoginSurface';
import { DashboardSurface } from './surfaces/DashboardSurface';
import { MissionPlayerSurface } from './surfaces/MissionPlayerSurface';
import { WorkbenchSurface } from './surfaces/WorkbenchSurface';
import { TutorSurface } from './surfaces/TutorSurface';
import { EvidenceLedgerSurface } from './surfaces/EvidenceLedgerSurface';
import { CompetencyGraphSurface } from './surfaces/CompetencyGraphSurface';
import { ReviewsSurface } from './surfaces/ReviewsSurface';
import { SettingsSurface } from './surfaces/SettingsSurface';

const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { learnerId, logout } = useAuth();
  
  if (!learnerId) {
    return <LoginSurface />;
  }

  return (
    <div className="min-h-screen flex text-slate-200">
      <nav className="w-64 glass-panel p-6 flex flex-col z-10 relative">
        <div className="flex-1">
          <h2 className="text-2xl font-black mb-8 bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent tracking-tight">LearningOS</h2>
          <ul className="space-y-3">
            <li><Link to="/" className="block px-3 py-2 rounded-md hover:bg-slate-800/50 hover:text-indigo-400 transition-all text-sm font-medium">Dashboard</Link></li>
            <li><Link to="/missions/demo" className="block px-3 py-2 rounded-md hover:bg-slate-800/50 hover:text-indigo-400 transition-all text-sm font-medium">Mission Player</Link></li>
            <li><Link to="/workbench" className="block px-3 py-2 rounded-md hover:bg-slate-800/50 hover:text-indigo-400 transition-all text-sm font-medium">Workbench</Link></li>
            <li><Link to="/tutor" className="block px-3 py-2 rounded-md hover:bg-slate-800/50 hover:text-indigo-400 transition-all text-sm font-medium">Socratic Tutor</Link></li>
            <li><Link to="/evidence" className="block px-3 py-2 rounded-md hover:bg-slate-800/50 hover:text-indigo-400 transition-all text-sm font-medium">Evidence Ledger</Link></li>
            <li><Link to="/competencies" className="block px-3 py-2 rounded-md hover:bg-slate-800/50 hover:text-indigo-400 transition-all text-sm font-medium">Competencies</Link></li>
            <li><Link to="/reviews" className="block px-3 py-2 rounded-md hover:bg-slate-800/50 hover:text-indigo-400 transition-all text-sm font-medium">Reviews</Link></li>
            <li><Link to="/settings" className="block px-3 py-2 rounded-md hover:bg-slate-800/50 hover:text-indigo-400 transition-all text-sm font-medium">Settings</Link></li>
          </ul>
        </div>
        
        <div className="pt-6 mt-auto">
          <div className="text-xs text-slate-500 mb-1 uppercase tracking-wider font-semibold">Active Profile</div>
          <div className="font-mono text-sm text-indigo-300 mb-4 truncate" title={learnerId}>{learnerId}</div>
          <button 
            onClick={logout}
            className="w-full py-2.5 bg-slate-800/50 hover:bg-red-900/40 hover:text-red-400 text-slate-300 text-sm font-medium rounded-lg transition-all border border-slate-700/50"
          >
            Logout
          </button>
        </div>
      </nav>
      <main className="flex-1 overflow-auto">
        {children}
      </main>
    </div>
  );
};

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<DashboardSurface />} />
            <Route path="/missions/:id" element={<MissionPlayerSurface />} />
            <Route path="/workbench" element={<WorkbenchSurface />} />
            <Route path="/tutor" element={<TutorSurface />} />
            <Route path="/evidence" element={<EvidenceLedgerSurface />} />
            <Route path="/competencies" element={<CompetencyGraphSurface />} />
            <Route path="/reviews" element={<ReviewsSurface />} />
            <Route path="/settings" element={<SettingsSurface />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;
