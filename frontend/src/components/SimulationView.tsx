import { useRef, useEffect, useState } from 'react';
import type { SimulationState, SceneObject, RobotStatus } from '../types';

interface Props {
  state: SimulationState | null;
  style?: React.CSSProperties;
}

const CANVAS_W = 800;
const CANVAS_H = 600;

const GRID_COLOR = '#1e293b';
const BG_COLOR = '#0f172a';
const FURNITURE_COLOR = '#334155';
const ITEM_COLOR = '#f59e0b';
const ROBOT_COLORS: Record<string, string> = {
  Alice: '#3b82f6',
  Bob: '#22c55e',
  David: '#a855f7',
  Lucy: '#ec4899',
};

const ROBOT_SHAPES: Record<string, string> = {
  Alice: '⬡',
  Bob: '⬢',
  David: '⬠',
  Lucy: '⟐',
};

export function SimulationView({ state, style }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);
  const [showGrid] = useState(true);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const dpr = window.devicePixelRatio || 1;
    canvas.width = CANVAS_W * dpr;
    canvas.height = CANVAS_H * dpr;
    ctx.scale(dpr, dpr);

    // Clear
    ctx.fillStyle = BG_COLOR;
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

    if (!state) {
      ctx.fillStyle = '#475569';
      ctx.font = '16px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Start a run to see simulation', CANVAS_W / 2, CANVAS_H / 2);
      return;
    }

    // Grid
    if (showGrid) {
      ctx.strokeStyle = GRID_COLOR;
      ctx.lineWidth = 0.5;
      for (let x = 0; x <= CANVAS_W; x += 40) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, CANVAS_H);
        ctx.stroke();
      }
      for (let y = 0; y <= CANVAS_H; y += 40) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(CANVAS_W, y);
        ctx.stroke();
      }
    }

    // Convert world to canvas coordinates
    const toCanvas = (x: number, y: number): [number, number] => {
      return [
        40 + (x / 10) * (CANVAS_W - 80),
        40 + (y / 10) * (CANVAS_H - 80),
      ];
    };

    const scale = (val: number) => val * 40;

    // Draw furniture
    const scene = state.scene;
    if (scene?.objects) {
      for (const obj of Object.values(scene.objects)) {
        if (obj.category === 'furniture') {
          const [cx, cy] = toCanvas(obj.posX, obj.posY);
          const w = scale(obj.width);
          const h = scale(obj.height);

          ctx.fillStyle = obj.isContainer && !obj.isOpen ? '#475569' : FURNITURE_COLOR;
          ctx.strokeStyle = '#64748b';
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.roundRect(cx - w / 2, cy - h / 2, w, h, 4);
          ctx.fill();
          ctx.stroke();

          // Label
          ctx.fillStyle = '#94a3b8';
          ctx.font = '10px Inter, sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(obj.name, cx, cy + h / 2 + 12);

          // Container contents
          if (obj.isOpen && obj.contains.length > 0) {
            ctx.fillStyle = '#fbbf24';
            ctx.font = '8px Inter, sans-serif';
            ctx.fillText(`[${obj.contains.join(', ')}]`, cx, cy - h / 2 - 6);
          }

          // Stand pose indicator
          if (obj.standPoseX != null && obj.standPoseY != null) {
            const [sx, sy] = toCanvas(obj.standPoseX, obj.standPoseY);
            ctx.fillStyle = '#22d3ee';
            ctx.beginPath();
            ctx.arc(sx, sy, 3, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }

      // Draw items
      for (const obj of Object.values(scene.objects)) {
        if (obj.category === 'item') {
          const [cx, cy] = toCanvas(obj.posX, obj.posY);
          ctx.fillStyle = ITEM_COLOR;
          ctx.beginPath();
          ctx.arc(cx, cy, 5, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillStyle = '#e2e8f0';
          ctx.font = '9px Inter, sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(obj.name, cx, cy + 14);
        }
      }
    }

    // Draw robots
    if (state.robots) {
      for (const robot of Object.values(state.robots)) {
        const [rx, ry] = toCanvas(robot.posX, robot.posY);
        const color = ROBOT_COLORS[robot.robotType] || '#3b82f6';

        // Robot body
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(rx, ry, 14, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(rx, ry, 14, 0, Math.PI * 2);
        ctx.stroke();

        // Robot name
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 10px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(robot.name, rx, ry + 4);

        // Gripper indicator
        if (robot.gripperOccupied && robot.graspingObject) {
          ctx.fillStyle = '#fbbf24';
          ctx.font = '8px Inter, sans-serif';
          ctx.fillText(`📦${robot.graspingObject}`, rx, ry - 22);
        }
      }
    }

    // Legend
    ctx.fillStyle = '#1e293b';
    ctx.fillRect(10, CANVAS_H - 90, 180, 80);
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.strokeRect(10, CANVAS_H - 90, 180, 80);

    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'left';
    let ly = CANVAS_H - 78;
    for (const [name, color] of Object.entries(ROBOT_COLORS)) {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(22, ly + 3, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#e2e8f0';
      ctx.fillText(name, 34, ly + 7);
      ly += 16;
    }
    if (state.leader) {
      ctx.fillStyle = '#fbbf24';
      ctx.fillText(`👑 Leader: ${state.leader}`, 22, ly + 7);
    }

    // Status overlay
    ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
    ctx.fillRect(10, 10, 180, 40);
    ctx.fillStyle = '#e2e8f0';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`Step: ${state.step}`, 20, 28);
    ctx.fillText(state.taskProgress, 20, 44);

  }, [state, showGrid]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !state) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    // Check robots
    if (state.robots) {
      const toCanvas = (x: number, y: number): [number, number] => {
        return [40 + (x / 10) * (CANVAS_W - 80), 40 + (y / 10) * (CANVAS_H - 80)];
      };
      for (const robot of Object.values(state.robots)) {
        const [rx, ry] = toCanvas(robot.posX, robot.posY);
        const dist = Math.sqrt((mx - rx) ** 2 + (my - ry) ** 2);
        if (dist < 16) {
          setTooltip({
            x: mx + 14,
            y: my - 10,
            text: `${robot.name} (${robot.robotType})\nPos: (${robot.posX.toFixed(1)}, ${robot.posY.toFixed(1)})\nGripper: ${robot.graspingObject || 'empty'}`,
          });
          return;
        }
      }
    }
    setTooltip(null);
  };

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', overflow: 'hidden',
      ...style,
    }}>
      <div style={{
        padding: '8px 12px', background: '#1e293b',
        borderBottom: '1px solid #334155', fontSize: 13,
        color: '#94a3b8', fontWeight: 500,
      }}>
        🎮 Simulation View
      </div>
      <div style={{
        flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
        position: 'relative', overflow: 'hidden',
      }}>
        <canvas
          ref={canvasRef}
          style={{ width: CANVAS_W, height: CANVAS_H, maxWidth: '100%', maxHeight: '100%' }}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setTooltip(null)}
        />
        {tooltip && (
          <div style={{
            position: 'absolute', left: tooltip.x, top: tooltip.y,
            background: '#1e293b', border: '1px solid #475569',
            borderRadius: 6, padding: '6px 10px', fontSize: 11,
            color: '#e2e8f0', whiteSpace: 'pre-wrap',
            pointerEvents: 'none', zIndex: 10,
            boxShadow: '0 4px 12px rgba(0,0,0,0.3)',
          }}>
            {tooltip.text}
          </div>
        )}
      </div>
    </div>
  );
}
