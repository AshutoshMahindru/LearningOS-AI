import { useEffect } from 'react';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { Spinner } from './components/Spinner';
import { AuthProvider, useAuth } from './context/AuthContext';
import { AppShell } from './layout/AppShell';
import { ArtifactsSurface } from './surfaces/ArtifactsSurface';
import { CompetencyGraphSurface } from './surfaces/CompetencyGraphSurface';
import { DashboardSurface } from './surfaces/DashboardSurface';
import { LoginSurface } from './surfaces/LoginSurface';
import { MissionPlayerSurface } from './surfaces/MissionPlayerSurface';
import { ReviewsSurface } from './surfaces/ReviewsSurface';
import { SessionSurface } from './surfaces/SessionSurface';
import { SettingsSurface } from './surfaces/SettingsSurface';
import { TutorSurface } from './surfaces/TutorSurface';
import { WorkbenchSurface } from './surfaces/WorkbenchSurface';

function SkipLink() {
  return (
    <a className="skip-link" href="#main-content">
      Skip to main content
    </a>
  );
}

function DocumentTitle({ title }: { title: string }) {
  useEffect(() => {
    document.title = title;
  }, [title]);
  return null;
}

export function AppRoutes() {
  const { status, learner } = useAuth();

  if (status === 'bootstrapping') {
    return (
      <>
        <SkipLink />
        <DocumentTitle title="LearningOS" />
        <main id="main-content" className="flex min-h-screen items-center justify-center" tabIndex={-1}>
          <Spinner label="Connecting to local API" />
        </main>
      </>
    );
  }

  if (!learner) {
    return (
      <>
        <SkipLink />
        <DocumentTitle title="Sign in · LearningOS" />
        <LoginSurface />
      </>
    );
  }

  return (
    <>
      <SkipLink />
      <AppShell>
        <Routes>
          <Route path="/" element={<DashboardSurface />} />
          <Route path="/artifacts" element={<ArtifactsSurface />} />
          <Route path="/settings" element={<SettingsSurface />} />
          <Route path="/sessions/:id" element={<SessionSurface />} />
          <Route path="/player" element={<MissionPlayerSurface />} />
          <Route path="/workbench" element={<WorkbenchSurface />} />
          <Route path="/tutor" element={<TutorSurface />} />
          <Route path="/reviews" element={<ReviewsSurface />} />
          <Route path="/competencies" element={<CompetencyGraphSurface />} />
        </Routes>
      </AppShell>
    </>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </AuthProvider>
  );
}
