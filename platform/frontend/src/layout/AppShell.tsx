import { useState, type ReactNode } from 'react';
import { NavLink } from 'react-router-dom';
import { Button } from '../components/Button';
import { DiagnosticsDrawer } from '../components/DiagnosticsDrawer';
import { useAuth } from '../context/AuthContext';
import { cn } from '../cn';

const PRIMARY_NAV = [
  { to: '/', label: 'Dashboard' },
  { to: '/player', label: 'Player' },
  { to: '/artifacts', label: 'Artifacts' },
  { to: '/settings', label: 'Settings' },
] as const;

const LATER_NAV = [
  { to: '/workbench', label: 'Workbench' },
  { to: '/tutor', label: 'Tutor' },
  { to: '/reviews', label: 'Reviews' },
  { to: '/competencies', label: 'Competency' },
] as const;

function navClassName({ isActive }: { isActive: boolean }): string {
  return cn('nav-link focus-visible:shadow-focus', isActive && 'bg-elevated text-primary');
}

export function AppShell({ children }: { children: ReactNode }) {
  const { learner, logout } = useAuth();
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false);

  return (
    <div className="app-shell">
      <header className="app-sidebar">
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-textSecondary">
            Local platform
          </p>
          <p className="text-2xl font-black tracking-tight">LearningOS</p>
        </div>
        <nav aria-label="Primary">
          <ul className="space-y-1">
            {PRIMARY_NAV.map((item) => (
              <li key={item.to}>
                <NavLink to={item.to} end={item.to === '/'} className={navClassName}>
                  {item.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
        <nav aria-label="Later surfaces">
          <p className="mb-2 text-xs font-semibold uppercase tracking-widest text-textSecondary">
            Later
          </p>
          <ul className="space-y-1">
            {LATER_NAV.map((item) => (
              <li key={item.to}>
                <NavLink to={item.to} className={navClassName}>
                  {item.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
        <div className="mt-auto space-y-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-widest text-textSecondary">
              Learner
            </p>
            <p className="truncate font-mono text-sm text-primary" title={learner?.username}>
              {learner?.display_name || learner?.username}
            </p>
          </div>
          <Button variant="secondary" onClick={logout} className="w-full">
            Switch learner
          </Button>
          <Button variant="ghost" className="w-full" onClick={() => setDiagnosticsOpen(true)}>
            Diagnostics
          </Button>
        </div>
      </header>
      <main id="main-content" className="min-h-screen overflow-auto p-8" tabIndex={-1}>
        {children}
      </main>
      <DiagnosticsDrawer open={diagnosticsOpen} onClose={() => setDiagnosticsOpen(false)} />
    </div>
  );
}
