import { useState } from 'react';
import { useDynaHMRC } from './hooks/useDynaHMRC';
import { LandingPage } from './pages/LandingPage';
import { MissionPage } from './pages/MissionPage';
import { SplashPage } from './pages/SplashPage';

type Page = 'splash' | 'landing' | 'mission';

// 全局动画 CSS
const styleTag = document.createElement('style');
styleTag.textContent = `
  @keyframes floatDown {
    0% { transform: translateY(-10px) scale(1); opacity: 0; }
    10% { opacity: 1; }
    90% { opacity: 0.5; }
    100% { transform: translateY(100vh) scale(0.3); opacity: 0; }
  }
  @keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
  }
  @keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
  }
`;
document.head.appendChild(styleTag);

export default function App() {
  const hmrc = useDynaHMRC();
  const [page, setPage] = useState<Page>('splash');
  const [runId, setRunId] = useState<string | null>(null);
  const [landingKey, setLandingKey] = useState(0);

  const handleStartMission = (newRunId: string) => {
    setRunId(newRunId);
    setPage('mission');
  };

  const handleBack = () => {
    setRunId(null);
    setPage('landing');
    setLandingKey(k => k + 1);
  };

  return (
    <div style={{
      width: '100%',
      maxWidth: '100%',
      background: '#0f172a',
      minHeight: '100vh',
    }}>
      {page === 'splash' && (
        <SplashPage onEnter={() => setPage('landing')} />
      )}
      {page === 'landing' && (
        <LandingPage key={landingKey} hmrc={hmrc} onStartMission={handleStartMission} onBack={() => setPage('splash')} />
      )}
      {page === 'mission' && (
        <MissionPage key={runId} hmrc={hmrc} onBack={handleBack} />
      )}
    </div>
  );
}
