import { useEffect, useRef } from 'react';
import type { RobotDialogue } from '../types';

interface Props {
  dialogues: RobotDialogue[];
  style?: React.CSSProperties;
}

const STAGE_CONFIG: Record<string, { label: string; color: string; icon: string }> = {
  self_description: { label: 'Self-Description', color: '#f59e0b', icon: '📝' },
  task_allocation_bidding: { label: 'Planning & Bidding', color: '#8b5cf6', icon: '📋' },
  leader_election: { label: 'Leader Election', color: '#ec4899', icon: '🗳️' },
  execution_reflection: { label: 'Execution', color: '#06b6d4', icon: '⚡' },
  completed: { label: 'Completed', color: '#22c55e', icon: '✅' },
};

const ROBOT_EMOJIS: Record<string, string> = {
  Alice: '🦾',
  Bob: '🦿',
  David: '🚗',
  Lucy: '🚁',
  '[SYSTEM]': '💻',
};

export function DialoguePanel({ dialogues, style }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [dialogues]);

  // Group dialogues by stage
  const grouped = dialogues.reduce<Record<string, RobotDialogue[]>>((acc, d) => {
    if (!acc[d.stage]) acc[d.stage] = [];
    acc[d.stage].push(d);
    return acc;
  }, {});

  const stageOrder = ['self_description', 'task_allocation_bidding', 'leader_election', 'execution_reflection', 'completed'];

  if (dialogues.length === 0) {
    return (
      <div style={{
        flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: '#475569', fontSize: 14, background: '#0f172a', ...style,
      }}>
        <div style={{ textAlign: 'center' }}>
          <span style={{ fontSize: 40 }}>🤖</span>
          <p style={{ marginTop: 8 }}>Running a task to see robot dialogues</p>
          <p style={{ fontSize: 12, color: '#334155', marginTop: 4 }}>
            Each robot's thinking process and actions will appear here
          </p>
        </div>
      </div>
    );
  }

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', overflow: 'hidden',
      borderRight: '1px solid #334155', background: '#0f172a', ...style,
    }}>
      {/* Header */}
      <div style={{
        padding: '10px 14px', background: '#1e293b',
        borderBottom: '1px solid #334155',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <span style={{ fontSize: 13, color: '#94a3b8', fontWeight: 500 }}>
          💬 Robot Dialogues
        </span>
        <span style={{ fontSize: 11, color: '#64748b' }}>
          {dialogues.length} messages
        </span>
      </div>

      {/* Dialogues */}
      <div ref={scrollRef} style={{
        flex: 1, overflowY: 'auto', padding: 12,
        display: 'flex', flexDirection: 'column', gap: 4,
      }}>
        {/* Stage progress bar */}
        <StageProgressBar currentStage={dialogues[dialogues.length - 1]?.stage || 'self_description'} />

        {stageOrder.map(stage => {
          const msgs = grouped[stage];
          if (!msgs || msgs.length === 0) return null;
          const cfg = STAGE_CONFIG[stage] || { label: stage, color: '#64748b', icon: '•' };

          return (
            <div key={stage}>
              <div style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '6px 0', marginTop: 8,
                borderBottom: `1px solid ${cfg.color}30`,
              }}>
                <span style={{ fontSize: 14 }}>{cfg.icon}</span>
                <span style={{ fontSize: 12, color: cfg.color, fontWeight: 600 }}>
                  {cfg.label}
                </span>
                <span style={{ fontSize: 11, color: '#64748b' }}>
                  ({msgs.length} messages)
                </span>
              </div>

              {msgs.map((d, i) => (
                <DialogueCard key={`${stage}-${i}`} dialogue={d} />
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function StageProgressBar({ currentStage }: { currentStage: string }) {
  const stages = ['self_description', 'task_allocation_bidding', 'leader_election', 'execution_reflection'];
  const currentIdx = stages.indexOf(currentStage);
  const displayIdx = currentIdx >= 0 ? currentIdx : 3;

  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 4,
      padding: '8px 0', marginBottom: 4,
    }}>
      {stages.map((stage, i) => {
        const done = i < displayIdx;
        const active = i === displayIdx;
        const cfg = STAGE_CONFIG[stage];

        return (
          <div key={stage} style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 4 }}>
            <div style={{
              width: 20, height: 20, borderRadius: '50%',
              background: done ? cfg.color : active ? cfg.color + '60' : '#1e293b',
              border: `2px solid ${active ? cfg.color : '#334155'}`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 9, color: '#fff', flexShrink: 0,
              transition: 'all 0.3s',
            }}>
              {done ? '✓' : active ? i + 1 : i + 1}
            </div>
            <span style={{
              fontSize: 10, color: active ? cfg.color : '#475569',
              fontWeight: active ? 600 : 400,
              whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
            }}>
              {cfg.label}
            </span>
            {i < stages.length - 1 && (
              <div style={{
                flex: 1, height: 2,
                background: done ? cfg.color : '#1e293b',
                minWidth: 8,
              }} />
            )}
          </div>
        );
      })}
    </div>
  );
}

function DialogueCard({ dialogue }: { dialogue: RobotDialogue }) {
  const cfg = STAGE_CONFIG[dialogue.stage] || { color: '#64748b', icon: '•', label: '' };
  const emoji = ROBOT_EMOJIS[dialogue.robotName] || '🤖';
  const isSystem = dialogue.robotName === '[SYSTEM]';
  const isLeader = dialogue.robotName.includes('[LEADER]');

  return (
    <div style={{
      background: isLeader ? '#1e293b' : '#151d2d',
      border: `1px solid ${isLeader ? cfg.color + '40' : '#1e293b'}`,
      borderLeft: `3px solid ${cfg.color}`,
      borderRadius: 6,
      padding: 10,
      fontSize: 13,
    }}>
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6,
        marginBottom: isSystem ? 0 : 6,
      }}>
        <span>{emoji}</span>
        <strong style={{
          color: isSystem ? '#64748b' : '#f1f5f9',
          fontSize: 13,
        }}>
          {dialogue.robotName.replace('[LEADER]', ' 👑')}
        </strong>
        {isLeader && (
          <span style={{ fontSize: 11, color: '#fbbf24' }}>LEADER</span>
        )}
      </div>

      {/* Thoughts */}
      {dialogue.thoughts && !isSystem && (
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
        color: isSystem ? '#64748b' : '#cbd5e1',
        whiteSpace: 'pre-wrap',
        lineHeight: 1.5,
        fontSize: isSystem ? 12 : 12,
        fontStyle: isSystem ? 'italic' : 'normal',
      }}>
        {dialogue.content}
      </div>
    </div>
  );
}
