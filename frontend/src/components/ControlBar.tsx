import type { SimulationState } from '../types';

interface Props {
  state: SimulationState | null;
  onStart: () => void;
  onPause: () => void;
  onResume: () => void;
  onStop: () => void;
  running: boolean;
}

export function ControlBar({ state, onStart, onPause, onResume, onStop, running }: Props) {
  const isPaused = state?.stage !== 'completed' && running && state && state.step > 0;
  const canStart = state && state.step === 0;

  return (
    <div style={{
      background: '#1e293b', borderRadius: 8, padding: 12,
      border: '1px solid #334155',
    }}>
      <h3 style={{ margin: '0 0 8px', fontSize: 13, color: '#94a3b8', fontWeight: 500 }}>
        🎮 Controls
      </h3>
      <div style={{ display: 'flex', gap: 6 }}>
        <ControlButton
          label="▶ Start"
          onClick={onStart}
          disabled={!canStart}
          color="#22c55e"
        />
        <ControlButton
          label="⏸ Pause"
          onClick={onPause}
          disabled={!isPaused}
          color="#f59e0b"
        />
        <ControlButton
          label="▶ Resume"
          onClick={onResume}
          disabled={!isPaused}
          color="#3b82f6"
        />
        <ControlButton
          label="⏹ Stop"
          onClick={onStop}
          disabled={!running}
          color="#ef4444"
        />
      </div>
    </div>
  );
}

function ControlButton({ label, onClick, disabled, color }: {
  label: string;
  onClick: () => void;
  disabled: boolean;
  color: string;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        flex: 1, padding: '8px 4px', borderRadius: 6,
        background: disabled ? '#1e293b' : `${color}20`,
        color: disabled ? '#475569' : color,
        cursor: disabled ? 'not-allowed' : 'pointer',
        fontWeight: 600, fontSize: 12,
        border: disabled ? '1px solid #1e293b' : `1px solid ${color}40`,
        transition: 'all 0.2s',
      }}
    >
      {label}
    </button>
  );
}
