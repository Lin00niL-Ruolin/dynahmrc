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
  const isPaused = state?.paused === true;
  const canPause = running && state && !isPaused && (state.step ?? 0) > 0 && state.stage === 'execution_reflection';
  const canResume = running && state && isPaused;
  const canStop = running && state && state.step > 0;
  const canStart = state && state.step === 0;

  return (
    <div style={{
      background: '#1e293b', borderRadius: 8, padding: 8,
      border: '1px solid #334155',
    }}>
      <div style={{ display: 'flex', gap: 4 }}>
        <ControlButton
          label="▶ Start"
          onClick={onStart}
          disabled={!canStart}
          color="#22c55e"
        />
        <ControlButton
          label="⏸"
          onClick={onPause}
          disabled={!canPause}
          color="#f59e0b"
          title="Pause"
        />
        <ControlButton
          label="▶"
          onClick={onResume}
          disabled={!canResume}
          color="#3b82f6"
          title="Resume"
        />
        <ControlButton
          label="⏹"
          onClick={onStop}
          disabled={!canStop}
          color="#ef4444"
          title="Stop"
        />
      </div>
    </div>
  );
}

function ControlButton({ label, onClick, disabled, color, title }: {
  label: string;
  onClick: () => void;
  disabled: boolean;
  color: string;
  title?: string;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title || label}
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
