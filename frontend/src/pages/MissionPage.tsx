import { useState } from 'react';
import { useDynaHMRC } from '../hooks/useDynaHMRC';
import { DialoguePanel } from '../components/DialoguePanel';
import { SimulationView } from '../components/SimulationView';
import { ControlBar } from '../components/ControlBar';
import type { SimulationState } from '../types';

type ViewMode = 'split' | 'dialogue' | 'simulation';

interface Props {
  hmrc: ReturnType<typeof useDynaHMRC>;
  onBack: () => void;
}

const TASK_NAMES: Record<string, string> = {
  pack_objects: '📦 Pack Objects',
  sort_solids: '🎨 Sort Solids',
  make_sandwich: '🥪 Make Sandwich',
};

const robotEmojis: Record<string, string> = {
  Alice: '🦾', Bob: '🦿', David: '🚗', Lucy: '🚁',
};

export function MissionPage({ hmrc, onBack }: Props) {
  const [viewMode, setViewMode] = useState<ViewMode>('split');

  const taskType = hmrc.state?.taskType || 'pack_objects';

  const isRunning = hmrc.state !== null
    && hmrc.state?.stage !== 'completed'
    && hmrc.state?.stage !== 'stopped';

  return (
    <div style={{
      height: '100vh',
      background: '#0f172a',
      color: '#e2e8f0',
      fontFamily: "'Inter', system-ui, sans-serif",
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
    }}>
      {/* ===== Header ===== */}
      <header style={{
        background: '#1e293b',
        borderBottom: '1px solid #334155',
        padding: '8px 16px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexShrink: 0,
      }}>
        {/* Left: Back + Task info */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <button onClick={onBack} style={{
            background: 'none', border: '1px solid #334155',
            borderRadius: 6, padding: '4px 10px',
            color: '#94a3b8', cursor: 'pointer', fontSize: 13,
          }}>← Back</button>
          <span style={{ fontSize: 18 }}>{TASK_NAMES[taskType] || '🤖 Mission'}</span>
          <span style={{ fontSize: 12, color: '#64748b' }}>#{hmrc.runId?.slice(-6) || '...'}</span>
        </div>

        {/* Center: Stage indicator */}
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {['self_description', 'task_allocation_bidding', 'leader_election', 'execution_reflection'].map((s, i) => {
            const stage = hmrc.state?.stage || '';
            const stageOrder = ['self_description', 'task_allocation_bidding', 'leader_election', 'execution_reflection', 'completed', 'stopped'];
            const currentIdx = stageOrder.indexOf(stage);
            const done = i <= currentIdx;
            const labels = ['📝 介绍', '📋 分工', '🗳️ 投票', '⚡ 执行'];
            return (
              <div key={s} style={{
                display: 'flex', alignItems: 'center', gap: 4,
                fontSize: 12, color: done ? '#22d3ee' : '#334155',
                fontWeight: done ? 500 : 400,
              }}>
                {i > 0 && <span style={{ color: done ? '#22d3ee' : '#334155' }}>→</span>}
                <span>{labels[i]}</span>
              </div>
            );
          })}
        </div>

        {/* Right: Status + View toggle */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{
            fontSize: 12, padding: '2px 8px', borderRadius: 4,
            background: hmrc.connected ? '#064e3b' : '#450a0a',
            color: hmrc.connected ? '#4ade80' : '#f87171',
          }}>
            {hmrc.connected ? '● Live' : '● Offline'}
          </span>

          {/* View toggle */}
          <div style={{ display: 'flex', background: '#0f172a', borderRadius: 6, border: '1px solid #334155' }}>
            {[
              { mode: 'split' as ViewMode, label: '分屏' },
              { mode: 'dialogue' as ViewMode, label: '对话' },
              { mode: 'simulation' as ViewMode, label: '仿真' },
            ].map(opt => (
              <button key={opt.mode} onClick={() => setViewMode(opt.mode)} style={{
                padding: '4px 8px', border: 'none', fontSize: 12,
                background: viewMode === opt.mode ? '#334155' : 'transparent',
                color: viewMode === opt.mode ? '#f1f5f9' : '#64748b',
                cursor: 'pointer', fontWeight: viewMode === opt.mode ? 600 : 400,
              }}>{opt.label}</button>
            ))}
          </div>
        </div>
      </header>

      {/* ===== Main Content ===== */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

        {/* ---- Left: Dialogue Panel ---- */}
        {(viewMode === 'split' || viewMode === 'dialogue') && (
          <div style={{
            flex: viewMode === 'split' ? 1 : 1,
            display: 'flex', flexDirection: 'column', overflow: 'hidden',
            borderRight: viewMode === 'split' ? '1px solid #334155' : 'none',
          }}>
            <DialoguePanel
              dialogues={hmrc.dialogues}
              style={{ flex: 1, border: 'none' }}
            />
          </div>
        )}

        {/* ---- Right: Simulation + Sidebar ---- */}
        {(viewMode === 'split' || viewMode === 'simulation') && (
          <div style={{
            flex: viewMode === 'split' ? 1 : 1,
            display: 'flex', flexDirection: 'column', overflow: 'hidden',
          }}>
            {/* Simulation Canvas */}
            <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
              <SimulationView
                state={hmrc.state}
                style={{ width: '100%', height: '100%' }}
              />
            </div>

            {/* Bottom bar: Controls + Status + Robots */}
            <div style={{
              height: 90, flexShrink: 0,
              background: '#1e293b', borderTop: '1px solid #334155',
              display: 'flex', alignItems: 'stretch',
            }}>
              {/* Controls */}
              <div style={{
                width: 180, padding: '8px 12px',
                borderRight: '1px solid #334155',
                display: 'flex', flexDirection: 'column', justifyContent: 'center',
              }}>
                <ControlBar
                  state={hmrc.state}
                  onStart={hmrc.start}
                  onPause={hmrc.pause}
                  onResume={hmrc.resume}
                  onStop={hmrc.stop}
                  running={isRunning}
                />
              </div>

              {/* Progress */}
              <div style={{
                width: 200, padding: '8px 12px',
                borderRight: '1px solid #334155',
                display: 'flex', flexDirection: 'column', justifyContent: 'center',
                gap: 4,
              }}>
                <div style={{ fontSize: 11, color: '#94a3b8' }}>
                  Step {hmrc.state?.step ?? 0}
                </div>
                {hmrc.state?.taskProgress && (
                  <div style={{ fontSize: 11, color: '#64748b' }}>
                    {hmrc.state.taskProgress}
                  </div>
                )}
                {/* Progress bar */}
                <div style={{ height: 4, background: '#0f172a', borderRadius: 2, overflow: 'hidden' }}>
                  <div style={{
                    height: '100%',
                    background: hmrc.state?.taskCompleted ? '#22c55e' : '#22d3ee',
                    width: hmrc.state?.taskCompleted ? '100%'
                      : `${Math.min(((hmrc.state?.step || 0) / 50) * 100, 95)}%`,
                    transition: 'width 0.3s',
                  }} />
                </div>
              </div>

              {/* Robot Status */}
              <div style={{
                flex: 1, padding: '8px 12px', overflow: 'hidden',
                display: 'flex', flexDirection: 'column', justifyContent: 'center',
              }}>
                <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>
                  {hmrc.state?.robots ? `${Object.keys(hmrc.state.robots).length} robots` : 'Waiting...'}
                </div>
                <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  {hmrc.state?.robots && Object.values(hmrc.state.robots as any[]).map((r: any) => (
                    <div key={r.name} style={{
                      display: 'flex', alignItems: 'center', gap: 4,
                      padding: '2px 6px', borderRadius: 4, fontSize: 11,
                      background: r.name === hmrc.state?.leader ? '#78350f' : '#0f172a',
                      border: r.name === hmrc.state?.leader ? '1px solid #fbbf2440' : '1px solid #334155',
                    }}>
                      <span>{robotEmojis[r.robotType] || '🤖'}</span>
                      <span style={{
                        color: r.name === hmrc.state?.leader ? '#fbbf24' : '#cbd5e1',
                        fontWeight: r.name === hmrc.state?.leader ? 600 : 400,
                      }}>{r.name}</span>
                      {r.graspingObject && <span style={{ fontSize: 10, color: '#22d3ee' }}>📦{r.graspingObject}</span>}
                      {r.name === hmrc.state?.leader && <span style={{ fontSize: 10 }}>👑</span>}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
