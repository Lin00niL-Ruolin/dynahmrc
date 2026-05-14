import { useEffect, useRef } from 'react';
import type { RobotDialogue } from '../types';

interface Props {
  dialogues: RobotDialogue[];
  style?: React.CSSProperties;
}

const STAGE_COLORS: Record<string, string> = {
  self_description: '#f59e0b',
  task_allocation_bidding: '#8b5cf6',
  leader_election: '#ec4899',
  execution_reflection: '#06b6d4',
  completed: '#22c55e',
};

const ROBOT_EMOJIS: Record<string, string> = {
  Alice: '🦾',
  Bob: '🦿',
  David: '🚗',
  Lucy: '🚁',
  '[SYSTEM]': '💻',
};

const STAGE_NAMES: Record<string, string> = {
  self_description: 'Self-Description',
  task_allocation_bidding: 'Planning & Bidding',
  leader_election: 'Leader Election',
  execution_reflection: 'Execution & Reflection',
  completed: 'Completed',
};

export function DialoguePanel({ dialogues, style }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [dialogues]);

  if (dialogues.length === 0) {
    return (
      <div style={{
        flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: '#475569', fontSize: 14, ...style,
      }}>
        <p>Start a run to see robot dialogues here</p>
      </div>
    );
  }

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', overflow: 'hidden',
      borderRight: '1px solid #334155', ...style,
    }}>
      <div style={{
        padding: '8px 12px', background: '#1e293b',
        borderBottom: '1px solid #334155', fontSize: 13,
        color: '#94a3b8', fontWeight: 500,
      }}>
        💬 Robot Dialogues ({dialogues.length})
      </div>

      <div ref={scrollRef} style={{
        flex: 1, overflowY: 'auto', padding: 12,
        display: 'flex', flexDirection: 'column', gap: 8,
      }}>
        {dialogues.map((d, i) => (
          <DialogueCard key={i} dialogue={d} index={i} />
        ))}
      </div>
    </div>
  );
}

function DialogueCard({ dialogue, index }: { dialogue: RobotDialogue; index: number }) {
  const stageColor = STAGE_COLORS[dialogue.stage] || '#64748b';
  const emoji = ROBOT_EMOJIS[dialogue.robotName] || '🤖';

  return (
    <div style={{
      background: '#1e293b',
      border: '1px solid #334155',
      borderLeft: `3px solid ${stageColor}`,
      borderRadius: 6,
      padding: 10,
      fontSize: 13,
    }}>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6,
      }}>
        <span>{emoji}</span>
        <strong style={{ color: '#f1f5f9', fontSize: 13 }}>
          {dialogue.robotName}
        </strong>
        <span style={{
          fontSize: 11, color: stageColor,
          background: `${stageColor}20`,
          padding: '1px 6px', borderRadius: 4,
        }}>
          {STAGE_NAMES[dialogue.stage] || dialogue.stage}
        </span>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: '#64748b' }}>
          #{index + 1}
        </span>
      </div>

      {/* Thoughts */}
      {dialogue.thoughts && (
        <div style={{
          background: '#0f172a', borderRadius: 4, padding: '6px 8px',
          marginBottom: 6, fontSize: 12, color: '#94a3b8',
          borderLeft: '2px solid #f59e0b',
        }}>
          <span style={{ color: '#f59e0b', fontSize: 11, fontWeight: 500 }}>💭 Thoughts</span>
          <p style={{ margin: '2px 0 0', whiteSpace: 'pre-wrap', lineHeight: 1.4 }}>
            {dialogue.thoughts}
          </p>
        </div>
      )}

      {/* Content */}
      <div style={{
        color: '#cbd5e1',
        whiteSpace: 'pre-wrap',
        lineHeight: 1.5,
        fontSize: 12,
      }}>
        {dialogue.content}
      </div>
    </div>
  );
}
