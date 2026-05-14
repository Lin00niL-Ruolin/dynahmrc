import { useState } from 'react';
import type { AppConfig } from '../types';

interface Props {
  config: AppConfig | null;
  onRun: (cfg: any) => Promise<string | null>;
  running: boolean;
}

export function ConfigPanel({ config, onRun, running }: Props) {
  const [taskType, setTaskType] = useState('pack_objects');
  const [layout, setLayout] = useState('kitchen');
  const [selectedRobots, setSelectedRobots] = useState(['Alice', 'Bob', 'David', 'Lucy']);
  const [maxSteps, setMaxSteps] = useState(50);

  const allRobots = ['Alice', 'Bob', 'David', 'Lucy'];

  const toggleRobot = (name: string) => {
    setSelectedRobots(prev =>
      prev.includes(name) ? prev.filter(r => r !== name) : [...prev, name]
    );
  };

  const handleRun = async () => {
    if (selectedRobots.length === 0) return;
    await onRun({
      taskType,
      layout,
      robots: selectedRobots,
      maxSteps,
    });
  };

  return (
    <div style={{
      background: '#1e293b', borderRadius: 8, padding: 14,
      border: '1px solid #334155', fontSize: 13,
    }}>
      <h3 style={{ margin: '0 0 12px', fontSize: 14, color: '#f1f5f9', fontWeight: 600 }}>⚙️ Configuration</h3>

      {/* Task Type */}
      <div style={{ marginBottom: 12 }}>
        <label style={{ color: '#94a3b8', display: 'block', marginBottom: 4 }}>Task</label>
        <select value={taskType} onChange={e => setTaskType(e.target.value)}
          style={selectStyle}>
          <option value="pack_objects">Pack Objects</option>
          <option value="sort_solids">Sort Solids</option>
          <option value="make_sandwich">Make Sandwich</option>
        </select>
      </div>

      {/* Layout */}
      <div style={{ marginBottom: 12 }}>
        <label style={{ color: '#94a3b8', display: 'block', marginBottom: 4 }}>Layout</label>
        <select value={layout} onChange={e => setLayout(e.target.value)}
          style={selectStyle}>
          <option value="kitchen">Kitchen</option>
          <option value="living_room">Living Room</option>
        </select>
      </div>

      {/* Robots */}
      <div style={{ marginBottom: 12 }}>
        <label style={{ color: '#94a3b8', display: 'block', marginBottom: 4 }}>Robots</label>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {allRobots.map(name => (
            <button key={name} onClick={() => toggleRobot(name)}
              style={{
                padding: '4px 10px', borderRadius: 6, border: '1px solid #475569',
                background: selectedRobots.includes(name) ? '#3b82f6' : '#1e293b',
                color: selectedRobots.includes(name) ? '#fff' : '#94a3b8',
                cursor: 'pointer', fontSize: 12,
              }}>
              {name}
            </button>
          ))}
        </div>
      </div>

      {/* Max Steps */}
      <div style={{ marginBottom: 16 }}>
        <label style={{ color: '#94a3b8', display: 'block', marginBottom: 4 }}>
          Max Steps: {maxSteps}
        </label>
        <input type="range" min={10} max={100} value={maxSteps}
          onChange={e => setMaxSteps(Number(e.target.value))}
          style={{ width: '100%', accentColor: '#3b82f6' }} />
      </div>

      {/* Run Button */}
      <button onClick={handleRun} disabled={running || selectedRobots.length === 0}
        style={{
          width: '100%', padding: '10px', borderRadius: 8, border: 'none',
          background: running ? '#334155' : '#3b82f6',
          color: running ? '#64748b' : '#fff',
          cursor: running ? 'not-allowed' : 'pointer',
          fontWeight: 600, fontSize: 14,
        }}>
        {running ? 'Running...' : '🚀 Start Run'}
      </button>
    </div>
  );
}

const selectStyle: React.CSSProperties = {
  width: '100%', padding: '6px 8px', borderRadius: 6,
  background: '#0f172a', color: '#e2e8f0', border: '1px solid #475569',
  fontSize: 13, outline: 'none',
};
