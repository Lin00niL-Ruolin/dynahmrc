import { useState, useEffect } from 'react';
import { useDynaHMRC } from './hooks/useDynaHMRC';
import { ConfigPanel } from './components/ConfigPanel';
import { DialoguePanel } from './components/DialoguePanel';
import { SimulationView } from './components/SimulationView';
import { ControlBar } from './components/ControlBar';
import type { AppConfig } from './types';

export default function App() {
  const hmrc = useDynaHMRC();
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [activeView, setActiveView] = useState<'both' | 'dialogue' | 'simulation'>('both');

  useEffect(() => {
    hmrc.loadConfig().then(setConfig);
  }, []);

  return (
    <div style={{
      minHeight: '100vh',
      background: '#0f172a',
      color: '#e2e8f0',
      fontFamily: "'Inter', system-ui, sans-serif",
    }}>
      {/* Header */}
      <header style={{
        background: 'linear-gradient(135deg, #1e293b, #0f172a)',
        borderBottom: '1px solid #334155',
        padding: '16px 24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 28 }}>🤖</span>
          <div>
            <h1 style={{ margin: 0, fontSize: 20, fontWeight: 600, color: '#f1f5f9' }}>
              DynaHMRC Demo
            </h1>
            <p style={{ margin: 0, fontSize: 12, color: '#94a3b8' }}>
              Decentralized Heterogeneous Multi-Robot Collaboration
            </p>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <span style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 6,
            fontSize: 13,
            color: hmrc.connected ? '#4ade80' : '#f87171',
          }}>
            <span style={{
              width: 8, height: 8, borderRadius: '50%',
              background: hmrc.connected ? '#4ade80' : '#f87171',
              display: 'inline-block',
            }} />
            {hmrc.connected ? `Connected${hmrc.runId ? ` (${hmrc.runId})` : ''}` : 'Disconnected'}
          </span>
          <button onClick={() => setActiveView(
            activeView === 'both' ? 'dialogue' : activeView === 'dialogue' ? 'simulation' : 'both'
          )} style={{
            background: '#334155', color: '#e2e8f0', border: 'none',
            padding: '6px 12px', borderRadius: 6, cursor: 'pointer',
            fontSize: 12,
          }}>
            {activeView === 'both' ? 'Split' : activeView === 'dialogue' ? 'Show Sim' : 'Show Dialogue'}
          </button>
        </div>
      </header>

      <div style={{ display: 'flex', gap: 0, height: 'calc(100vh - 72px)' }}>
        {/* Left Sidebar - Config & Controls */}
        <aside style={{
          width: 280, minWidth: 280,
          borderRight: '1px solid #334155',
          overflowY: 'auto',
          padding: 16,
          display: 'flex', flexDirection: 'column', gap: 16,
        }}>
          <ConfigPanel config={config} onRun={hmrc.createRun} running={!!hmrc.state?.step && !hmrc.state?.taskCompleted} />
          <ControlBar
            state={hmrc.state}
            onStart={hmrc.start}
            onPause={hmrc.pause}
            onResume={hmrc.resume}
            onStop={hmrc.stop}
            running={hmrc.state?.step !== undefined && !hmrc.state?.taskCompleted}
          />
          <StatusPanel state={hmrc.state} />
        </aside>

        {/* Main Content */}
        <main style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
          {(activeView === 'both' || activeView === 'dialogue') && (
            <DialoguePanel
              dialogues={hmrc.dialogues}
              style={{ flex: activeView === 'both' ? 1 : undefined,
                       width: activeView === 'dialogue' ? '100%' : undefined }}
            />
          )}
          {(activeView === 'both' || activeView === 'simulation') && (
            <SimulationView
              state={hmrc.state}
              style={{ flex: activeView === 'both' ? 1 : undefined,
                       width: activeView === 'simulation' ? '100%' : undefined }}
            />
          )}
        </main>
      </div>
    </div>
  );
}

function StatusPanel({ state }: { state: any }) {
  if (!state) return null;
  return (
    <div style={{
      background: '#1e293b', borderRadius: 8, padding: 12,
      border: '1px solid #334155', fontSize: 13,
    }}>
      <h3 style={{ margin: '0 0 8px', fontSize: 13, color: '#94a3b8', fontWeight: 500 }}>Status</h3>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <Row label="Step" value={`${state.step}`} />
        <Row label="Stage" value={state.stage?.replace(/_/g, ' ') || '-'} />
        <Row label="Leader" value={state.leader || '-'} />
        <Row label="Progress" value={state.taskProgress || '-'} />
        <Row label="Completed" value={state.taskCompleted ? '✅' : '⏳'} />
        <Row label="Actions" value={`${state.actions?.length || 0}`} />
      </div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
      <span style={{ color: '#64748b' }}>{label}</span>
      <span style={{ color: '#e2e8f0', fontWeight: 500 }}>{value}</span>
    </div>
  );
}
