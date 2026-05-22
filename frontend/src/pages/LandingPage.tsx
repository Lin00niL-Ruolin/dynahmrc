import { useState, useEffect } from 'react';
import { useDynaHMRC } from '../hooks/useDynaHMRC';
import type { AppConfig } from '../types';

interface Props {
  hmrc: ReturnType<typeof useDynaHMRC>;
  onStartMission: (runId: string) => void;
  onBack: () => void;
}

const TASKS = [
  {
    id: 'make_sandwich',
    name: 'Make Sandwich',
    icon: '🥪',
    desc: 'Stack ingredients on the cutting board',
    color: '#f59e0b',
    scene: 'scene1',
    sceneLabel: '场景一 (10m×8m)',
  },
  {
    id: 'sort_solids',
    name: 'Sort Solids',
    icon: '🎨',
    desc: 'Sort colored solids onto matching panels',
    color: '#8b5cf6',
    scene: 'kitchen',
    sceneLabel: '场景二 (Kitchen)',
  },
  {
    id: 'pack_objects',
    name: 'Pack Objects',
    icon: '📦',
    desc: 'Pack bowl, fork, soap, apple into tray',
    color: '#06b6d4',
    scene: 'living_room',
    sceneLabel: '场景三 (Living Room)',
  },
];

// Task → Scene 映射
const TASK_SCENE_MAP: Record<string, string> = {
  make_sandwich: 'scene1',
  sort_solids: 'kitchen',
  pack_objects: 'living_room',
};

const SCENES = [
  { id: 'scene1', name: '场景一 (10m×8m)', desc: '对应 Make Sandwich 任务 — 有 Bob Lab 和厨房区' },
  { id: 'kitchen', name: '场景二 (Kitchen)', desc: '对应 Sort Solids 任务 — 标准厨房布局' },
  { id: 'living_room', name: '场景三 (Living Room)', desc: '对应 Pack Objects 任务 — 客厅家具布局' },
];

const ROBOTS = [
  { id: 'Alice', icon: '🦾', type: 'Mobile Manipulation', selected: true },
  { id: 'Bob', icon: '🦿', type: 'Fixed Arm', selected: true },
  { id: 'David', icon: '🚗', type: 'Mobile Robot', selected: true },
  { id: 'Lucy', icon: '🚁', type: 'Drone', selected: true },
];

export function LandingPage({ hmrc, onStartMission, onBack }: Props) {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [taskType, setTaskType] = useState('make_sandwich');
  const [layout, setLayout] = useState(TASK_SCENE_MAP['make_sandwich']);
  const [selectedRobots, setSelectedRobots] = useState(ROBOTS.map(r => r.id));

  // Layout-specific robot defaults
  const layoutRobotDefaults: Record<string, string[]> = {
    scene1: ['Alice', 'Bob', 'David', 'Lucy'],
    kitchen: ['Alice', 'Bob', 'David', 'Lucy'],
    living_room: ['Alice', 'Bob', 'David', 'Lucy'],
  };

  const handleTaskChange = (id: string) => {
    setTaskType(id);
    // 自动选择对应的场景
    const sceneId = TASK_SCENE_MAP[id];
    if (sceneId) {
      setLayout(sceneId);
      if (layoutRobotDefaults[sceneId]) {
        setSelectedRobots(layoutRobotDefaults[sceneId]);
      }
    }
  };
  const [maxSteps, setMaxSteps] = useState(50);
  const [starting, setStarting] = useState(false);

  useEffect(() => {
    hmrc.loadConfig().then(setConfig);
  }, []);

  const toggleRobot = (id: string) => {
    setSelectedRobots(prev =>
      prev.includes(id) ? prev.filter(r => r !== id) : [...prev, id]
    );
  };

  const handleStartMission = async () => {
    if (selectedRobots.length === 0) return;
    setStarting(true);
    const runId = await hmrc.createRun({
      taskType,
      layout,
      robots: selectedRobots,
      maxSteps,
    });
    setStarting(false);
    if (runId) {
      onStartMission(runId);
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: '#0f172a',
      color: '#e2e8f0',
      fontFamily: "'Inter', system-ui, sans-serif",
      display: 'flex',
      flexDirection: 'column',
      overflow: 'auto',
    }}>
      {/* Connection Bar */}
      <div style={{
        background: '#1e293b',
        borderBottom: '1px solid #334155',
        padding: '4px 24px',
        fontSize: 14,
        color: '#94a3b8',
        display: 'flex',
        gap: 16,
        fontFamily: 'monospace',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        {/* Back Button */}
        <button onClick={onBack} style={{
          background: 'none',
          border: 'none',
          color: '#94a3b8',
          cursor: 'pointer',
          fontSize: 14,
          fontFamily: 'monospace',
          padding: '4px 8px',
          borderRadius: 4,
          display: 'flex',
          alignItems: 'center',
          gap: 4,
        }}
          onMouseEnter={e => (e.target as HTMLButtonElement).style.color = '#e2e8f0'}
          onMouseLeave={e => (e.target as HTMLButtonElement).style.color = '#94a3b8'}
        >
          ← Back
        </button>
        <span style={{
          display: 'inline-flex', alignItems: 'center', gap: 6,
          color: config ? '#4ade80' : '#f87171',
        }}>
          <span style={{
            width: 6, height: 6, borderRadius: '50%',
            background: config ? '#4ade80' : '#f87171',
            display: 'inline-block',
          }} />
          {config ? 'Server Online' : 'Connecting...'}
        </span>
        {/* WebSocket status: only relevant after mission starts */}
        {hmrc.connected && (
          <span style={{
            display: 'inline-flex', alignItems: 'center', gap: 6,
            color: '#22d3ee',
          }}>
            <span style={{
              width: 6, height: 6, borderRadius: '50%',
              background: '#22d3ee',
              display: 'inline-block',
            }} />
            WebSocket Connected
          </span>
        )}
        {hmrc.error && (
          <span style={{ color: '#f87171' }}>⚠ {hmrc.error}</span>
        )}
      </div>

      {/* Hero Section */}
      <div style={{
        textAlign: 'center',
        padding: '48px 24px 32px',
        background: 'linear-gradient(180deg, #0f172a 0%, #1e293b 100%)',
        borderBottom: '1px solid #334155',
      }}>
        <div style={{ fontSize: 56, marginBottom: 8 }}>🤖</div>
        <h1 style={{
          margin: 0,
          fontSize: 36,
          fontWeight: 700,
          background: 'linear-gradient(135deg, #22d3ee, #3b82f6, #8b5cf6)',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          letterSpacing: '-0.5px',
        }}>
          DynaHMRC
        </h1>
        <p style={{
          margin: '8px 0 0',
          fontSize: 15,
          color: '#cbd5e1',
          fontWeight: 400,
        }}>
          Decentralized Heterogeneous Multi-Robot Collaboration
        </p>

      </div>

      <div style={{
        flex: 1,
        maxWidth: 1200,
        width: '100%',
        margin: '0 auto',
        padding: '32px 24px 48px',
        display: 'flex',
        flexDirection: 'column',
        gap: 32,
      }}>
        {/* Task Selection */}
        <section>
          <SectionTitle icon="🎯" text="Task Selection" />
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: 12,
          }}>
            {TASKS.map(task => (
              <TaskCard
                key={task.id}
                task={task}
                selected={taskType === task.id}
                onClick={() => handleTaskChange(task.id)}
              />
            ))}
          </div>
        </section>

        {/* Scene Selection — 自动根据任务匹配 */}
        <section>
          <SectionTitle icon="🗺️" text="Task → Scene Mapping" />
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: 10,
          }}>
            {TASKS.map(task => {
              const scene = SCENES.find(s => s.id === TASK_SCENE_MAP[task.id]);
              if (!scene) return null;
              return (
                <SceneCard
                  key={scene.id}
                  scene={scene}
                  selected={layout === scene.id}
                  onClick={() => {}}
                />
              );
            })}
          </div>
        </section>

        {/* Robot Selection */}
        <section>
          <SectionTitle icon="🤖" text="Robot Selection" />
          <p style={{ fontSize: 15, color: '#94a3b8', marginBottom: 12 }}>
            Select which robots to deploy. All selected by default.
          </p>
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(4, 1fr)',
            gap: 10,
          }}>
            {ROBOTS.map(robot => (
              <RobotCard
                key={robot.id}
                robot={robot}
                selected={selectedRobots.includes(robot.id)}
                onToggle={() => toggleRobot(robot.id)}
              />
            ))}
          </div>
        </section>

        {/* Max Steps */}
        <section>
          <SectionTitle icon="⚙️" text="Simulation Settings" />
          <div style={{
            background: '#1e293b',
            borderRadius: 10,
            border: '1px solid #334155',
            padding: '14px 18px',
          }}>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}>
              <span style={{ fontSize: 13, color: '#cbd5e1' }}>Max Execution Steps</span>
              <span style={{
                fontSize: 14, color: '#22d3ee', fontWeight: 600,
                fontFamily: 'monospace',
              }}>
                {maxSteps}
              </span>
            </div>
            <input
              type="range" min={10} max={100} value={maxSteps}
              onChange={e => setMaxSteps(Number(e.target.value))}
              style={{
                width: '100%', marginTop: 8,
                accentColor: '#3b82f6', height: 4,
              }}
            />
            <div style={{
              display: 'flex', justifyContent: 'space-between',
              fontSize: 13, color: '#64748b', marginTop: 2,
            }}>
              <span>10</span>
              <span>100</span>
            </div>
          </div>
        </section>

        {/* Start Mission Button */}
        <div style={{ textAlign: 'center', paddingTop: 8 }}>
          <button
            onClick={handleStartMission}
            disabled={selectedRobots.length === 0 || starting}
            style={{
              padding: '16px 48px',
              borderRadius: 12,
              border: 'none',
              background: starting
                ? '#334155'
                : 'linear-gradient(135deg, #3b82f6, #8b5cf6)',
              color: starting ? '#64748b' : '#fff',
              cursor: starting || selectedRobots.length === 0 ? 'not-allowed' : 'pointer',
              fontWeight: 700,
              fontSize: 18,
              letterSpacing: '0.5px',
              boxShadow: starting
                ? 'none'
                : '0 0 30px rgba(59,130,246,0.3), 0 4px 12px rgba(0,0,0,0.3)',
              transition: 'all 0.3s',
              position: 'relative',
              overflow: 'hidden',
            }}
            onMouseEnter={e => {
              if (!starting && selectedRobots.length > 0) {
                (e.target as HTMLButtonElement).style.transform = 'translateY(-2px)';
                (e.target as HTMLButtonElement).style.boxShadow = '0 0 40px rgba(59,130,246,0.4), 0 6px 20px rgba(0,0,0,0.4)';
              }
            }}
            onMouseLeave={e => {
              (e.target as HTMLButtonElement).style.transform = '';
              (e.target as HTMLButtonElement).style.boxShadow = starting
                ? 'none'
                : '0 0 30px rgba(59,130,246,0.3), 0 4px 12px rgba(0,0,0,0.3)';
            }}
          >
            {starting ? (
              <>
                <span style={{ display: 'inline-block', animation: 'pulse 1s infinite' }}>⟳</span> Starting Mission...
              </>
            ) : (
              '🚀 Start Mission'
            )}
          </button>
          {selectedRobots.length === 0 && (
            <p style={{ fontSize: 15, color: '#f87171', marginTop: 8 }}>
              Please select at least one robot
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function SectionTitle({ icon, text }: { icon: string; text: string }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 8,
      marginBottom: 14,
    }}>
      <span style={{ fontSize: 16 }}>{icon}</span>
      <h2 style={{
        margin: 0, fontSize: 16, fontWeight: 600, color: '#f1f5f9',
      }}>
        {text}
      </h2>
      <div style={{
        flex: 1, height: 1,
        background: 'linear-gradient(90deg, #334155, transparent)',
        marginLeft: 4,
      }} />
    </div>
  );
}

function TaskCard({ task, selected, onClick }: {
  task: typeof TASKS[0];
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <div
      onClick={onClick}
      style={{
        background: selected ? '#1e293b' : '#151d2d',
        border: `1px solid ${selected ? task.color : '#334155'}`,
        borderRadius: 10,
        padding: 16,
        cursor: 'pointer',
        transition: 'all 0.2s',
        ...(selected ? {
          boxShadow: `0 0 20px ${task.color}20`,
        } : {}),
      }}
      onMouseEnter={e => {
        if (!selected) {
          (e.currentTarget as HTMLDivElement).style.borderColor = '#475569';
          (e.currentTarget as HTMLDivElement).style.background = '#1e293b';
        }
      }}
      onMouseLeave={e => {
        if (!selected) {
          (e.currentTarget as HTMLDivElement).style.borderColor = '#334155';
          (e.currentTarget as HTMLDivElement).style.background = '#151d2d';
        }
      }}
    >
      <div style={{
        fontSize: 32, textAlign: 'center', marginBottom: 8,
      }}>
        {task.icon}
      </div>
      <h3 style={{
        margin: 0, fontSize: 14, fontWeight: 600,
        color: selected ? task.color : '#e2e8f0',
        textAlign: 'center',
      }}>
        {task.name}
      </h3>
      <p style={{
        margin: '6px 0 0', fontSize: 15, color: '#94a3b8',
        textAlign: 'center', lineHeight: 1.4,
      }}>
        {task.desc}
      </p>
      <div style={{
        marginTop: 8, fontSize: 14, color: selected ? task.color : '#64748b',
        textAlign: 'center', fontWeight: 500,
        padding: '3px 8px', borderRadius: 4,
        background: selected ? `${task.color}15` : 'transparent',
      }}>
        🗺️ {task.sceneLabel}
      </div>
      {selected && (
        <div style={{
          textAlign: 'center', marginTop: 8,
        }}>
          <span style={{
            fontSize: 14, color: task.color, fontWeight: 500,
            background: `${task.color}20`,
            padding: '2px 8px', borderRadius: 4,
          }}>
            ✓ SELECTED
          </span>
        </div>
      )}
    </div>
  );
}

function SceneCard({ scene, selected, onClick }: {
  scene: typeof SCENES[0];
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <div
      onClick={onClick}
      style={{
        background: selected ? '#1e293b' : '#151d2d',
        border: `1px solid ${selected ? '#22d3ee' : '#334155'}`,
        borderRadius: 10,
        padding: '12px 14px',
        cursor: 'pointer',
        transition: 'all 0.2s',
        ...(selected ? { boxShadow: '0 0 15px rgba(34,211,238,0.15)' } : {}),
      }}
      onMouseEnter={e => {
        if (!selected) {
          (e.currentTarget as HTMLDivElement).style.borderColor = '#475569';
          (e.currentTarget as HTMLDivElement).style.background = '#1e293b';
        }
      }}
      onMouseLeave={e => {
        if (!selected) {
          (e.currentTarget as HTMLDivElement).style.borderColor = '#334155';
          (e.currentTarget as HTMLDivElement).style.background = '#151d2d';
        }
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <div style={{
          width: 10, height: 10, borderRadius: '50%', flexShrink: 0,
          background: selected ? '#22d3ee' : '#334155',
        }} />
        <div>
          <div style={{
            fontSize: 13, fontWeight: 500,
            color: selected ? '#22d3ee' : '#e2e8f0',
          }}>
            {scene.name}
          </div>
          <div style={{ fontSize: 14, color: '#94a3b8' }}>
            {scene.desc}
          </div>
        </div>
      </div>
    </div>
  );
}

function RobotCard({ robot, selected, onToggle }: {
  robot: typeof ROBOTS[0];
  selected: boolean;
  onToggle: () => void;
}) {
  return (
    <div
      onClick={onToggle}
      style={{
        background: selected ? '#1e293b' : '#151d2d',
        border: `1px solid ${selected ? '#3b82f6' : '#334155'}`,
        borderRadius: 10,
        padding: 14,
        cursor: 'pointer',
        transition: 'all 0.2s',
        textAlign: 'center',
        ...(selected ? { boxShadow: '0 0 15px rgba(59,130,246,0.2)' } : {}),
      }}
      onMouseEnter={e => {
        if (!selected) {
          (e.currentTarget as HTMLDivElement).style.borderColor = '#475569';
          (e.currentTarget as HTMLDivElement).style.background = '#1e293b';
        }
      }}
      onMouseLeave={e => {
        if (!selected) {
          (e.currentTarget as HTMLDivElement).style.borderColor = '#334155';
          (e.currentTarget as HTMLDivElement).style.background = '#151d2d';
        }
      }}
    >
      <div style={{ fontSize: 28, marginBottom: 6 }}>{robot.icon}</div>
      <div style={{
        fontSize: 14, fontWeight: 600,
        color: selected ? '#3b82f6' : '#e2e8f0',
      }}>
        {robot.id}
      </div>
      <div style={{ fontSize: 13, color: '#94a3b8', marginTop: 2 }}>
        {robot.type}
      </div>
      <div style={{
        marginTop: 8,
        width: 36,
        height: 18,
        borderRadius: 10,
        background: selected ? '#3b82f6' : '#334155',
        position: 'relative',
        transition: 'all 0.2s',
        display: 'inline-block',
      }}>
        <div style={{
          width: 14, height: 14, borderRadius: '50%',
          background: '#fff',
          position: 'absolute',
          top: 2,
          left: selected ? 20 : 2,
          transition: 'left 0.2s',
        }} />
      </div>
    </div>
  );
}
