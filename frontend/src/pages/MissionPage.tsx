import { useState, useEffect, useCallback } from 'react';
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

// ==================== 任务配置文件 ====================

const TASK_THEMES: Record<string, {
  taskName: string;
  accent: string;
  gradient: string;
  icon: string;
  goal: string;
  robots: { name: string; role: string; emoji: string }[];
}> = {
  make_sandwich: {
    taskName: '🥪 Make Sandwich',
    accent: '#f59e0b',
    gradient: 'linear-gradient(135deg, #f59e0b, #d97706)',
    icon: '🥪',
    goal: 'Stack bread, lettuce, tomato, cheese, ham on cutting board in order',
    robots: [
      { name: 'Alice', role: 'Mobile Manipulation Robot', emoji: '🦾' },
      { name: 'Bob', role: 'Fixed Manipulation Robot', emoji: '🦿' },
      { name: 'David', role: 'Mobile Scout Robot', emoji: '🚗' },
      { name: 'Lucy', role: 'Aerial Drone Robot', emoji: '🚁' },
    ],
    steps: [
      'Collect ingredients (bread, lettuce, tomato, cheese, ham)',
      'Deliver to Bob\'s cutting board',
      'Bob assembles the sandwich in order',
      'Place top bread to complete',
    ],
  },
  sort_solids: {
    taskName: '🎨 Sort Solids',
    accent: '#8b5cf6',
    gradient: 'linear-gradient(135deg, #8b5cf6, #6d28d9)',
    icon: '🎨',
    goal: 'Sort red_cube, blue_sphere, green_cylinder onto matching colored panels',
    robots: [
      { name: 'Alice', role: 'Mobile Manipulation Robot', emoji: '🦾' },
      { name: 'Bob', role: 'Fixed Manipulation Robot', emoji: '🦿' },
      { name: 'David', role: 'Mobile Scout Robot', emoji: '🚗' },
      { name: 'Lucy', role: 'Aerial Drone Robot', emoji: '🚁' },
    ],
    steps: [
      'Identify colored solids on table_2',
      'Match to correct colored panels',
      'Transport each to its panel',
      'Verify all sorted correctly',
    ],
  },
  pack_objects: {
    taskName: '📦 Pack Objects',
    accent: '#06b6d4',
    gradient: 'linear-gradient(135deg, #06b6d4, #0891b2)',
    icon: '📦',
    goal: 'Find bowl, fork, soap, apple around the house and place them all into the tray',
    robots: [
      { name: 'Alice', role: 'Mobile Manipulation Robot', emoji: '🦾' },
      { name: 'Bob', role: 'Fixed Manipulation Robot', emoji: '🦿' },
      { name: 'David', role: 'Mobile Scout Robot', emoji: '🚗' },
      { name: 'Lucy', role: 'Aerial Drone Robot', emoji: '🚁' },
    ],
    steps: [
      'Locate items: bowl, fork, soap, apple',
      'Navigate to each item and pick it up',
      'Bring items to the packing table',
      'Place all items into the tray',
    ],
  },
};

// ==================== 组件 ====================

export function MissionPage({ hmrc, onBack }: Props) {
  const [viewMode, setViewMode] = useState<ViewMode>('split');

  const taskType = hmrc.state?.taskType || 'pack_objects';
  const theme = TASK_THEMES[taskType] || TASK_THEMES.pack_objects;

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
          {/* Task badge */}
          <span style={{
            fontSize: 13, fontWeight: 600,
            background: `${theme.accent}20`,
            color: theme.accent,
            padding: '2px 10px', borderRadius: 6,
            border: `1px solid ${theme.accent}40`,
          }}>
            {theme.taskName}
          </span>
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
                fontSize: 12, color: done ? theme.accent : '#334155',
                fontWeight: done ? 500 : 400,
              }}>
                {i > 0 && <span style={{ color: done ? theme.accent : '#334155' }}>→</span>}
                <span>{labels[i]}</span>
              </div>
            );
          })}
        </div>

        {/* Right: BestMan + Status + View toggle */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {/* BestMan 3D Connection */}
          <BestManControl taskType={taskType} />

          <span style={{
            fontSize: 12, padding: '2px 8px', borderRadius: 4,
            background: hmrc.connected ? '#064e3b' : '#450a0a',
            color: hmrc.connected ? '#4ade80' : '#f87171',
          }}>
            {hmrc.connected ? '● Live' : '● Offline'}
          </span>

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
            {/* Task briefing (shown before execution phase) */}
            {hmrc.state?.stage !== 'execution_reflection' && hmrc.state?.stage !== 'completed' && hmrc.state?.stage !== 'stopped' && (
              <TaskBriefing theme={theme} taskType={taskType} />
            )}
            <DialoguePanel
              dialogues={hmrc.dialogues}
              style={{ flex: 1, border: 'none' }}
            />
          </div>
        )}

        {/* ---- Right: Simulation + Bottom Bar ---- */}
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

            {/* Bottom bar — redesigned: no overlap, compact */}
            <div style={{
              height: 72, flexShrink: 0,
              background: '#1e293b', borderTop: '1px solid #334155',
              display: 'flex', alignItems: 'center', padding: '0 8px',
              gap: 0,
            }}>
              {/* Controls */}
              <div style={{
                flex: '0 0 120px', padding: '4px 8px',
                borderRight: '1px solid #334155',
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
                flex: '0 0 130px', padding: '4px 8px',
                borderRight: '1px solid #334155',
              }}>
                <div style={{ fontSize: 11, color: '#94a3b8', lineHeight: 1.4 }}>
                  Step {hmrc.state?.step ?? 0}
                </div>
                {hmrc.state?.taskProgress && (
                  <div style={{ fontSize: 10, color: '#64748b', lineHeight: 1.3 }}>
                    {hmrc.state.taskProgress}
                  </div>
                )}
                <div style={{ height: 3, background: '#0f172a', borderRadius: 2, marginTop: 2, overflow: 'hidden' }}>
                  <div style={{
                    height: '100%',
                    background: hmrc.state?.taskCompleted ? '#22c55e' : theme.accent,
                    width: hmrc.state?.taskCompleted ? '100%'
                      : `${Math.min(((hmrc.state?.step || 0) / 50) * 100, 95)}%`,
                    transition: 'width 0.3s',
                  }} />
                </div>
              </div>

              {/* Robot Status */}
              <RobotStatusBar state={hmrc.state} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// ==================== 任务简介面板 ====================

function TaskBriefing({ theme, taskType }: { theme: typeof TASK_THEMES[string]; taskType: string }) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div style={{
      background: '#1e293b',
      borderBottom: `1px solid ${theme.accent}30`,
      padding: collapsed ? '6px 12px' : '10px 14px',
      flexShrink: 0,
    }}>
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        cursor: 'pointer',
      }} onClick={() => setCollapsed(!collapsed)}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: 16 }}>{theme.icon}</span>
          <span style={{ fontSize: 13, fontWeight: 600, color: theme.accent }}>
            Mission Briefing
          </span>
        </div>
        <span style={{ fontSize: 11, color: '#64748b' }}>
          {collapsed ? '▸ Expand' : '▾ Hide'}
        </span>
      </div>

      {!collapsed && (
        <div style={{ marginTop: 8, fontSize: 12, color: '#94a3b8', lineHeight: 1.5 }}>
          <p style={{ margin: '0 0 6px' }}>
            <span style={{ color: '#cbd5e1', fontWeight: 500 }}>Goal:</span> {theme.goal}
          </p>

          {/* Steps removed - robots decide during collaboration */}

          {/* Robots */}
          <div>
            <span style={{ color: '#cbd5e1', fontWeight: 500 }}>Robots:</span>
            <div style={{ display: 'flex', gap: 6, marginTop: 4, flexWrap: 'wrap' }}>
              {theme.robots.map(r => (
                <span key={r.name} style={{
                  padding: '2px 6px', borderRadius: 4,
                  background: '#0f172a', border: '1px solid #334155',
                  fontSize: 11,
                }}>
                  {r.emoji} {r.name} <span style={{ color: '#64748b' }}>({r.role})</span>
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ==================== BestMan 3D 连接控制 ====================

function BestManControl({ taskType }: { taskType: string }) {
  const [connecting, setConnecting] = useState(false);
  const [connected, setConnected] = useState(false);
  const [bmStatus, setBmStatus] = useState<string>('');

  const checkStatus = useCallback(async () => {
    try {
      const resp = await fetch('/api/bestman/status');
      const data = await resp.json();
      setConnected(data.running || false);
      setBmStatus(data.env || '');
    } catch {}
  }, []);

  useEffect(() => {
    checkStatus();
    const interval = setInterval(checkStatus, 5000);
    return () => clearInterval(interval);
  }, [checkStatus]);

  const handleToggle = async () => {
    if (connected) {
      await fetch('/api/bestman/stop', { method: 'POST' });
      setConnected(false);
    } else {
      setConnecting(true);
      try {
        const sceneMap: Record<string, string> = {
          make_sandwich: 'scene1',
          sort_solids: 'scene2',
          pack_objects: 'scene3',
        };
        const resp = await fetch('/api/bestman/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ scene: sceneMap[taskType] || 'scene1', gui: true }),
        });
        const data = await resp.json();
        setConnected(data.ok || false);
      } catch {}
      setConnecting(false);
    }
  };

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
      <button
        onClick={handleToggle}
        disabled={connecting || bmStatus.includes('not')}
        title={bmStatus}
        style={{
          fontSize: 11, padding: '3px 8px', borderRadius: 4, border: 'none',
          cursor: connecting ? 'wait' : 'pointer',
          background: connected ? '#065f46' : connecting ? '#1e293b' : '#1e293b',
          border: `1px solid ${connected ? '#10b981' : '#334155'}`,
          color: connected ? '#6ee7b7' : '#94a3b8',
          fontWeight: 500,
        }}
      >
        {connecting ? '⟳' : connected ? '🧊 3D' : '🧊 3D Off'}
      </button>
    </div>
  );
}

// ==================== 机器人状态栏 ====================

const robotEmojis: Record<string, string> = {
  Alice: '🦾', Bob: '🦿', David: '🚗', Lucy: '🚁',
};

function RobotStatusBar({ state }: { state: SimulationState | null }) {
  if (!state?.robots) return null;

  const robotList = Object.values(state.robots as any[]);

  return (
    <div style={{
      flex: 1, padding: '4px 8px', overflow: 'hidden',
      display: 'flex', alignItems: 'center', gap: 6,
    }}>
      {robotList.map((r: any) => (
        <div key={r.name} style={{
          display: 'flex', alignItems: 'center', gap: 3,
          padding: '3px 8px', borderRadius: 4, fontSize: 11,
          background: r.name === state?.leader ? '#78350f' : '#0f172a',
          border: r.name === state?.leader ? '1px solid #fbbf2440' : '1px solid #334155',
          whiteSpace: 'nowrap',
        }}>
          <span>{robotEmojis[r.robotType] || '🤖'}</span>
          <span style={{
            color: r.name === state?.leader ? '#fbbf24' : '#cbd5e1',
            fontWeight: r.name === state?.leader ? 600 : 400,
          }}>{r.name}</span>
          {r.graspingObject && <span style={{ fontSize: 9, color: '#22d3ee' }}>📦{r.graspingObject}</span>}
          {r.name === state?.leader && <span style={{ fontSize: 10 }}>👑</span>}
        </div>
      ))}
    </div>
  );
}
