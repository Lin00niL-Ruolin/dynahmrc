import { useState } from 'react';
import { useDynaHMRC } from '../hooks/useDynaHMRC';
import { DialoguePanel } from '../components/DialoguePanel';
import { SimulationView } from '../components/SimulationView';
import { ControlBar } from '../components/ControlBar';
import type { SimulationState } from '../types';

type ViewMode = 'dialogue' | 'simulation' | 'split';

interface Props {
  hmrc: ReturnType<typeof useDynaHMRC>;
  onBack: () => void;
}

const TASK_NAMES: Record<string, string> = {
  pack_objects: 'Pack Objects',
  sort_solids: 'Sort Solids',
  make_sandwich: 'Make Sandwich',
};

const TASK_ICONS: Record<string, string> = {
  pack_objects: '📦',
  sort_solids: '🎨',
  make_sandwich: '🥪',
};

export function MissionPage({ hmrc, onBack }: Props) {
  const [viewMode, setViewMode] = useState<ViewMode>('dialogue');

  // Try to guess task type from state (or we can infer from context)
  const taskType = hmrc.runId?.includes('make_sandwich') ? 'make_sandwich'
    : hmrc.runId?.includes('sort_solids') ? 'sort_solids'
    : 'pack_objects';

  const viewOptions: { mode: ViewMode; label: string; icon: string }[] = [
    { mode: 'dialogue', label: 'Dialogue', icon: '💬' },
    { mode: 'simulation', label: 'Simulation', icon: '🎮' },
    { mode: 'split', label: 'Split', icon: '🖖' },
  ];

  const handleStop = () => {
    hmrc.stop();
    onBack();
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: '#0f172a',
      color: '#e2e8f0',
      fontFamily: "'Inter', system-ui, sans-serif",
      display: 'flex',
      flexDirection: 'column',
    }}>
      {/* Mission Header */}
      <header style={{
        background: 'linear-gradient(135deg, #1e293b, #0f172a)',
        borderBottom: '1px solid #334155',
        padding: '10px 20px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {/* Back button */}
          <button
            onClick={handleStop}
            style={{
              background: '#1e293b',
              border: '1px solid #334155',
              borderRadius: 8,
              padding: '6px 12px',
              color: '#cbd5e1',
              cursor: 'pointer',
              fontSize: 13,
              display: 'flex',
              alignItems: 'center',
              gap: 6,
              transition: 'all 0.2s',
            }}
            onMouseEnter={e => {
              (e.target as HTMLButtonElement).style.background = '#334155';
              (e.target as HTMLButtonElement).style.color = '#e2e8f0';
            }}
            onMouseLeave={e => {
              (e.target as HTMLButtonElement).style.background = '#1e293b';
              (e.target as HTMLButtonElement).style.color = '#94a3b8';
            }}
          >
            ← Back
          </button>

          {/* Mission Info */}
          <div style={{
            borderLeft: '1px solid #334155',
            paddingLeft: 12,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 18 }}>
                {TASK_ICONS[taskType] || '🤖'}
              </span>
              <h1 style={{
                margin: 0, fontSize: 16, fontWeight: 600, color: '#f1f5f9',
              }}>
                {TASK_NAMES[taskType] || 'Mission'} — {hmrc.runId || '...'}
              </h1>
            </div>
            <div style={{
              fontSize: 14, color: '#94a3b8', marginTop: 1,
              display: 'flex', gap: 12,
            }}>
              <span>Step: {hmrc.state?.step ?? 0}</span>
              <span>Stage: {hmrc.state?.stage?.replace(/_/g, ' ') || 'init'}</span>
              <span>Dialogues: {hmrc.dialogues.length}</span>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          {/* Connection status */}
          <span style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 6,
            fontSize: 15,
            padding: '4px 10px',
            borderRadius: 6,
            background: hmrc.connected ? '#064e3b' : '#450a0a',
            color: hmrc.connected ? '#4ade80' : '#f87171',
          }}>
            <span style={{
              width: 6, height: 6, borderRadius: '50%',
              background: hmrc.connected ? '#4ade80' : '#f87171',
              display: 'inline-block',
            }} />
            {hmrc.connected ? 'Live' : 'Offline'}
          </span>

          {/* View toggle */}
          <div style={{
            display: 'flex',
            background: '#0f172a',
            borderRadius: 8,
            border: '1px solid #334155',
            overflow: 'hidden',
          }}>
            {viewOptions.map(opt => (
              <button
                key={opt.mode}
                onClick={() => setViewMode(opt.mode)}
                style={{
                  padding: '5px 10px',
                  border: 'none',
                  background: viewMode === opt.mode ? '#334155' : 'transparent',
                  color: viewMode === opt.mode ? '#f1f5f9' : '#64748b',
                  cursor: 'pointer',
                  fontSize: 15,
                  fontWeight: viewMode === opt.mode ? 600 : 400,
                  transition: 'all 0.15s',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 4,
                }}
              >
                {opt.icon} {opt.label}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div style={{
        flex: 1,
        display: 'flex',
        overflow: 'hidden',
        height: 'calc(100vh - 60px)',
      }}>
        {/* Center Content */}
        <main style={{
          flex: 1,
          display: 'flex',
          overflow: 'hidden',
        }}>
          {viewMode === 'dialogue' && (
            <DialoguePanel
              dialogues={hmrc.dialogues}
              style={{ width: '100%', height: '100%', border: 'none' }}
            />
          )}
          {viewMode === 'simulation' && (
            <SimulationView
              state={hmrc.state}
              style={{ width: '100%', height: '100%' }}
            />
          )}
          {viewMode === 'split' && (
            <>
              <DialoguePanel
                dialogues={hmrc.dialogues}
                style={{ flex: 1, borderRight: '1px solid #334155' }}
              />
              <SimulationView
                state={hmrc.state}
                style={{ flex: 1 }}
              />
            </>
          )}
        </main>

        {/* Right Sidebar */}
        <aside style={{
          width: 260,
          minWidth: 260,
          borderLeft: '1px solid #334155',
          background: '#0f172a',
          overflowY: 'auto',
          padding: 16,
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
        }}>
          {/* Controls */}
          <ControlBar
            state={hmrc.state}
            onStart={hmrc.start}
            onPause={hmrc.pause}
            onResume={hmrc.resume}
            onStop={hmrc.stop}
            running={hmrc.state?.step !== undefined && !hmrc.state?.taskCompleted}
          />

          {/* Status Panel */}
          <StatusPanel state={hmrc.state} />
        </aside>
      </div>
    </div>
  );
}

const robotEmojis: Record<string, string> = {
  Alice: '🦾', Bob: '🦿', David: '🚗', Lucy: '🚁',
};

function StatusPanel({ state }: { state: SimulationState | null }) {
  if (!state) return null;

  return (
    <div style={{
      background: '#1e293b', borderRadius: 8, padding: 12,
      border: '1px solid #334155', fontSize: 13,
    }}>
      <h3 style={{ margin: '0 0 8px', fontSize: 13, color: '#cbd5e1', fontWeight: 500 }}>
        📊 Mission Status
      </h3>

      {/* Progress */}
      {state.taskProgress && (
        <div style={{ marginBottom: 10 }}>
          <div style={{
            display: 'flex', justifyContent: 'space-between',
            fontSize: 14, color: '#94a3b8', marginBottom: 3,
          }}>
            <span>{state.taskProgress}</span>
            <span>{state.taskCompleted ? '100%' : `${Math.min(state.step * 2, 99)}%`}</span>
          </div>
          <div style={{
            height: 4, background: '#0f172a', borderRadius: 2, overflow: 'hidden',
          }}>
            <div style={{
              height: '100%', background: 'linear-gradient(90deg, #22c55e, #16a34a)', borderRadius: 2,
              transition: 'width 0.3s',
              width: state.taskCompleted ? '100%' : `${Math.min(state.step * 2, 95)}%`,
            }} />
          </div>
        </div>
      )}

      <div style={{
        display: 'flex', flexDirection: 'column', gap: 5,
      }}>
        <StatRow label="Step" value={`${state.step}`} />
        <StatRow
          label="Stage"
          value={state.stage?.replace(/_/g, ' → ').replace(/(^| )/g, (m: string) => m.toUpperCase()) || '-'}
        />
        <StatRow label="Leader" value={state.leader ? `👑 ${state.leader}` : '❌ None'} />
        <StatRow label="Actions" value={`${state.actions?.length || 0}`} />
        <StatRow label="Status" value={state.taskCompleted ? '✅ Completed' : '⏳ Running'} />
      </div>

      {/* Robot Status */}
      {state.robots && Object.keys(state.robots).length > 0 && (
        <div style={{
          marginTop: 10, borderTop: '1px solid #334155', paddingTop: 10,
        }}>
          <div style={{ fontSize: 14, color: '#94a3b8', marginBottom: 6 }}>
            🤖 Active Robots
          </div>
          {Object.values(state.robots as any[]).map((r: any) => (
            <div key={r.name} style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '4px 6px', fontSize: 14, borderRadius: 4,
              background: r.name === state.leader ? `${COLORS.leader}10` : 'transparent',
              marginBottom: 2,
            }}>
              <span>{robotEmojis[r.robotType] || '🤖'}</span>
              <span style={{
                color: r.name === state.leader ? COLORS.leader : '#e2e8f0',
                fontWeight: r.name === state.leader ? 600 : 400,
              }}>
                {r.name}
              </span>
              {r.name === state.leader && (
                <span style={{ fontSize: 13, color: COLORS.leader }}>👑</span>
              )}
              <span style={{ marginLeft: 'auto', color: '#94a3b8', fontSize: 10 }}>
                {r.graspingObject ? `📦${r.graspingObject}` : '🟢'}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Error display */}
      {state.taskCompleted && (
        <div style={{
          marginTop: 10, padding: 8, borderRadius: 6,
          background: '#064e3b', border: '1px solid #22c55e40',
          textAlign: 'center',
        }}>
          <span style={{ fontSize: 15, color: '#4ade80', fontWeight: 500 }}>
            ✅ Mission Complete
          </span>
        </div>
      )}
    </div>
  );
}

const COLORS = {
  bg: '#0f172a', grid: '#1e293b', furniture: '#334155',
  container_closed: '#475569', container_open: '#1e3a5f',
  item: '#f59e0b', item_placed: '#22c55e', label: '#94a3b8',
  text: '#e2e8f0', accent: '#22d3ee', leader: '#fbbf24',
  stand_pose: '#22d3ee', zone: 'rgba(34, 211, 238, 0.08)',
};

function StatRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <span style={{ color: '#94a3b8', fontSize: 12 }}>{label}</span>
      <span style={{ color: '#e2e8f0', fontWeight: 500, fontSize: 12 }}>{value}</span>
    </div>
  );
}
