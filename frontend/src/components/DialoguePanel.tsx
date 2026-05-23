import { useEffect, useRef } from 'react';
import type { RobotDialogue } from '../types';

interface Props {
  dialogues: RobotDialogue[];
  style?: React.CSSProperties;
}

const STAGE_CONFIG: Record<string, { label: string; color: string; accent: string; icon: string }> = {
  self_description: { label: 'Self-Description', color: '#f59e0b', accent: '#fbbf24', icon: '📝' },
  task_allocation_bidding: { label: 'Planning & Bidding', color: '#8b5cf6', accent: '#a78bfa', icon: '📋' },
  leader_election: { label: 'Leader Election', color: '#ec4899', accent: '#f472b6', icon: '🗳️' },
  execution_reflection: { label: 'Execution', color: '#06b6d4', accent: '#22d3ee', icon: '⚡' },
  completed: { label: 'Completed', color: '#22c55e', accent: '#4ade80', icon: '✅' },
};

const ROBOT_EMOJIS: Record<string, string> = {
  Alice: '🦾',
  Bob: '🦿',
  David: '🚗',
  Lucy: '🚁',
  '[SYSTEM]': '💻',
};

// ============ Key content extraction ============

function extractKeyContent(dialogue: RobotDialogue): { key: string; detail: string } {
  const { stage, content } = dialogue;

  // Helper: extract the 'Contents:' portion from LLM output (skip Thoughts/Reasons headers)
  const extractContents = (txt: string): string => {
    const lines = txt.split('\n');
    // Find the Contents section: look for lines after 'Contents:' or '2) **Contents**:' etc.
    let inContent = false;
    const contentLines: string[] = [];
    // Match section headers with optional number prefix, bold markers **, and colon
    const isContentsHeader = (s: string) => /^\s*\d*[\.\)]?\s*\*{0,2}Contents\*{0,2}\s*[:\-]?/i.test(s);
    const isBreakHeader = (s: string) => /^\s*\d*[\.\)]?\s*\*{0,2}(Thoughts?|Reasons?|Leader|Summary|Plan)\*{0,2}\s*[:\-]?/i.test(s);

    for (const line of lines) {
      const clean = line.trim();
      if (!clean) continue;
      if (isContentsHeader(clean)) {
        inContent = true;
        // Strip the header prefix, keep the text after colon/dash
        const afterHeader = clean.replace(/^\s*\d*[\.\)]?\s*\*{0,2}Contents\*{0,2}\s*[:\-]?\s*/, '');
        if (afterHeader) contentLines.push(afterHeader);
        continue;
      }
      if (isBreakHeader(clean)) {
        inContent = false;
        continue;
      }
      if (inContent) {
        contentLines.push(line);
      }
    }
    return contentLines.join('\n').trim() || txt; // fallback to full text
  };

  // Leader vote extraction
  const ROBOT_NAMES = new Set(['Alice', 'Bob', 'David', 'Lucy']);
  if (stage === 'leader_election') {
    // 策略1: 找结尾处的 **Leader:** 名 或 Leader: 名
    const voteRegex = /(?:^|[\n.])\s*\d*[\.\)]?\s*\*{0,2}Leader\*{0,2}\s*:\s*\*{0,2}\s*(\w+)/i;
    const voteMatch = content.match(voteRegex);
    if (voteMatch) {
      const name = voteMatch[1];
      // 只接受已知的机器人名
      if (ROBOT_NAMES.has(name)) {
        const reasons = content
          .replace(voteRegex, '')
          .replace(/^\d+[\)\.]\s*/gm, '')
          .replace(/\*{0,2}(Thoughts?|Reasons?)\*{0,2}\s*:.*/g, '')
          .trim();
        return { 
          key: `🗳️ I vote for **${name}**`,
          detail: reasons 
        };
      }
    }
    // 策略2: 换行格式（Leader: 单独一行，名字在下一行）
    const lines = content.split('\n').filter(l => l.trim());
    for (let i = 0; i < lines.length; i++) {
      if (/^\d*[\.\)]?\s*\*{0,2}Leader\*{0,2}\s*:?\s*$/i.test(lines[i].trim())) {
        if (i + 1 < lines.length) {
          const nameMatch = lines[i + 1].trim().match(/^(\w+)/);
          if (nameMatch && ROBOT_NAMES.has(nameMatch[1])) {
            return { key: `🗳️ I vote for **${nameMatch[1]}**`, detail: content };
          }
        }
      }
    }
    // 策略3: 全文搜索已知机器人名在 "Leader" 附近
    for (const robotName of ROBOT_NAMES) {
      const idx = content.lastIndexOf(robotName);
      if (idx > 0) {
        const before = content.slice(Math.max(0, idx - 30), idx);
        if (/Leader/i.test(before)) {
          return { key: `🗳️ I vote for **${robotName}**`, detail: content };
        }
      }
    }
    return { key: '🗳️ Casting vote...', detail: content };
  }

  // Execution action extraction
  if (stage === 'execution_reflection') {
    const lines = content.split('\n');
    const actionLine = lines[0] || '';
    const feedbackLines = lines.slice(1).filter(l => l.trim());
    return { key: `⚡ ${actionLine}`, detail: feedbackLines.join('\n') };
  }

  // Self description: just the intro paragraph (no Thoughts/Contents split)
  if (stage === 'self_description') {
    const intro = content.replace(/^(Thoughts?:\s*)?/i, '').trim();
    return { key: `🤝 ${intro}`, detail: '' };
  }

  // Task allocation: highlight the collaboration plan / task assignments
  if (stage === 'task_allocation_bidding') {
    // Try to extract plan portion (before campaign speech)
    const beforeCampaign = content.split(/Campaign Speech/i)[0] || content;
    const planPortion = extractContents(beforeCampaign);
    const displayText = planPortion && planPortion.length > 20
      ? planPortion.replace(/^\d+[\)\.\]\s]*/gm, '').trim()
      : content.replace(/^.*?Contents?:\s*/is, '').replace(/^\d+[\)\.\]\s]*/gm, '').trim().slice(0, 200);
    return { key: `📋 ${displayText}`, detail: content };
  }

  // Reflection
  if (content.includes('[REFLECTION]') || content.includes('[LEADER UPDATE]')) {
    const planMatch = content.match(/Plan:\s*(.+)/i);
    const summaryMatch = content.match(/Summary:\s*(.+)/i);
    const keyLine = planMatch
      ? `🔄 ${planMatch[1].trim()}`
      : summaryMatch
        ? `🔄 ${summaryMatch[1].trim()}`
        : `🔄 Plan update`;
    return { key: keyLine, detail: content };
  }

  // Completed
  if (stage === 'completed') {
    return { key: `✅ ${content}`, detail: '' };
  }

  // Fallback
  const firstLine = content.split('\n')[0] || content;
  return { key: firstLine.slice(0, 80), detail: content };
}

// ============ Component ============

export function DialoguePanel({ dialogues, style }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [dialogues]);

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
        <StageProgressBar currentStage={dialogues[dialogues.length - 1]?.stage || 'self_description'} />

        {stageOrder.map(stage => {
          const msgs = grouped[stage];
          if (!msgs || msgs.length === 0) return null;
          const cfg = STAGE_CONFIG[stage] || { label: stage, color: '#64748b', accent: '#94a3b8', icon: '•' };

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

// ============ Progress Bar ============

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
              <div style={{ flex: 1, height: 2, background: done ? cfg.color : '#1e293b', minWidth: 8 }} />
            )}
          </div>
        );
      })}
    </div>
  );
}

// ============ Dialogue Card ============

function DialogueCard({ dialogue }: { dialogue: RobotDialogue }) {
  const cfg = STAGE_CONFIG[dialogue.stage] || { label: '', color: '#64748b', accent: '#94a3b8', icon: '•' };
  const emoji = ROBOT_EMOJIS[dialogue.robotName] || '🤖';
  const isSystem = dialogue.robotName === '[SYSTEM]';
  const isLeader = dialogue.robotName.includes('[LEADER]');

  // Extract key content
  const { key, detail } = extractKeyContent(dialogue);

  // Determine accent color for the key highlight
  const keyColor = dialogue.stage === 'execution_reflection'
    ? '#22d3ee'
    : dialogue.stage === 'leader_election'
      ? '#f472b6'
      : dialogue.stage === 'self_description'
        ? '#fbbf24'
        : dialogue.stage === 'task_allocation_bidding'
          ? '#a78bfa'
          : '#4ade80';

  return (
    <div style={{
      background: isLeader ? '#1e293b' : '#151d2d',
      border: `1px solid ${isLeader ? cfg.color + '40' : '#1e293b'}`,
      borderLeft: `3px solid ${cfg.color}`,
      borderRadius: 6,
      padding: 10,
      fontSize: 13,
    }}>
      {/* Header: robot name */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6,
        marginBottom: 6,
      }}>
        <span>{emoji}</span>
        <strong style={{ color: isSystem ? '#64748b' : '#f1f5f9', fontSize: 13 }}>
          {dialogue.robotName.replace('[LEADER]', ' 👑')}
        </strong>
        {isLeader && <span style={{ fontSize: 11, color: '#fbbf24' }}>LEADER</span>}
      </div>

      {/* Key highlight - the most important output */}
      {key && (
        <div style={{
          background: `${keyColor}15`,
          border: `1px solid ${keyColor}40`,
          borderRadius: 6,
          padding: '8px 10px',
          marginBottom: detail ? 8 : 0,
          fontSize: 13,
          color: keyColor,
          fontWeight: 600,
          lineHeight: 1.4,
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}>
          {key}
        </div>
      )}

      {/* Thoughts (collapsible with small text) */}
      {dialogue.thoughts && !isSystem && (
        <details style={{ marginBottom: detail ? 6 : 0 }}>
          <summary style={{
            fontSize: 11, color: '#64748b', cursor: 'pointer',
            padding: '2px 0',
          }}>
            💭 Reasoning
          </summary>
          <div style={{
            background: '#0f172a', borderRadius: 4, padding: '6px 8px',
            marginTop: 4, fontSize: 12, color: '#94a3b8',
            borderLeft: '2px solid #f59e0b', lineHeight: 1.4,
            whiteSpace: 'pre-wrap',
          }}>
            {dialogue.thoughts}
          </div>
        </details>
      )}

      {/* Detail (secondary text) */}
      {detail && (
        <div style={{
          color: '#64748b',
          whiteSpace: 'pre-wrap',
          lineHeight: 1.5,
          fontSize: 11,
          marginTop: 4,
        }}>
          {detail}
        </div>
      )}
    </div>
  );
}
