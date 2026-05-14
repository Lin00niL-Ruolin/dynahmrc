import { useRef, useEffect, useState, useCallback } from 'react';
import type { SimulationState } from '../types';

interface Props {
  state: SimulationState | null;
  style?: React.CSSProperties;
}

const CANVAS_W = 800;
const CANVAS_H = 600;

const COLORS = {
  bg: '#0f172a', grid: '#1e293b', furniture: '#334155',
  container_closed: '#475569', container_open: '#1e3a5f',
  item: '#f59e0b', item_placed: '#22c55e', label: '#94a3b8',
  text: '#e2e8f0', accent: '#22d3ee', leader: '#fbbf24',
  stand_pose: '#22d3ee', zone: 'rgba(34, 211, 238, 0.08)',
};

const ROBOT_STYLE: Record<string, { color: string; shape: 'circle' | 'diamond' | 'hexagon' | 'square' }> = {
  Alice: { color: '#3b82f6', shape: 'circle' },
  Bob:    { color: '#22c55e', shape: 'square' },
  David:  { color: '#a855f7', shape: 'hexagon' },
  Lucy:   { color: '#ec4899', shape: 'diamond' },
};

const posHistory: Record<string, Array<[number, number]>> = {};

// Smoothed robot positions for animation
let animatedPositions: Record<string, { x: number; y: number; targetX: number; targetY: number }> = {};
let lastStateVersion = 0;

export function SimulationView({ state, style }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; text: string } | null>(null);
  const animRef = useRef(0);
  const lastRenderRef = useRef(0);

  const toCanvas = useCallback((x: number, y: number): [number, number] => {
    return [40 + (x / 10) * (CANVAS_W - 80), 40 + (y / 10) * (CANVAS_H - 80)];
  }, []);

  const scale = useCallback((val: number) => val * 40, []);

  // Update animation targets when new state arrives
  useEffect(() => {
    if (!state?.robots) return;
    const version = state.step + (state.leader ? 100 : 0);
    if (version <= lastStateVersion) return;
    lastStateVersion = version;

    for (const [name, robot] of Object.entries(state.robots)) {
      if (!animatedPositions[name]) {
        animatedPositions[name] = { x: robot.posX, y: robot.posY, targetX: robot.posX, targetY: robot.posY };
      } else {
        animatedPositions[name].targetX = robot.posX;
        animatedPositions[name].targetY = robot.posY;
      }
    }

    // Track position history
    for (const [name, robot] of Object.entries(state.robots)) {
      if (!posHistory[name]) posHistory[name] = [];
      const last = posHistory[name][posHistory[name].length - 1];
      if (!last || last[0] !== robot.posX || last[1] !== robot.posY) {
        posHistory[name].push([robot.posX, robot.posY]);
        if (posHistory[name].length > 20) posHistory[name].shift();
      }
    }
  }, [state]);

  // Animation loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = CANVAS_W * dpr;
    canvas.height = CANVAS_H * dpr;
    ctx.scale(dpr, dpr);

    let running = true;

    const render = () => {
      if (!running) return;

      // Interpolate positions
      for (const name of Object.keys(animatedPositions)) {
        const p = animatedPositions[name];
        const dx = p.targetX - p.x;
        const dy = p.targetY - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > 0.01) {
          const speed = 0.12;
          p.x += dx * speed;
          p.y += dy * speed;
        } else {
          p.x = p.targetX;
          p.y = p.targetY;
        }
      }

      // Clear
      ctx.fillStyle = COLORS.bg;
      ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

      // Static elements
      drawGrid(ctx);
      if (state) drawScene(ctx, state, toCanvas, scale);
      if (state) drawTrails(ctx, state, toCanvas);
      drawRobots(ctx);
      if (state) drawOverlays(ctx, state, toCanvas);

      animRef.current = requestAnimationFrame(render);
    };

    render();
    lastRenderRef.current = Date.now();

    return () => { running = false; cancelAnimationFrame(animRef.current); };
  }, [state, toCanvas, scale]);

  // ===== DRAWING FUNCTIONS =====

  function drawGrid(ctx: CanvasRenderingContext2D) {
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 0.5;
    for (let x = 0; x <= CANVAS_W; x += 40) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, CANVAS_H); ctx.stroke();
    }
    for (let y = 0; y <= CANVAS_H; y += 40) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(CANVAS_W, y); ctx.stroke();
    }
  }

  function drawScene(ctx: CanvasRenderingContext2D, s: SimulationState, toC: typeof toCanvas, sc: typeof scale) {
    if (!s.scene?.objects) return;

    // Task zone
    const [zx, zy] = toC(5, 5);
    ctx.fillStyle = COLORS.zone;
    ctx.beginPath(); ctx.arc(zx, zy, 60, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = 'rgba(34, 211, 238, 0.15)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.arc(zx, zy, 60, 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([]);

    // Restricted zones
    const restricted = (s as any).restrictedZones || [];
    for (const zone of restricted) {
      const [rzx, rzy] = toC(zone.x, zone.y);
      const r = zone.radius * 40;
      ctx.fillStyle = 'rgba(239, 68, 68, 0.12)';
      ctx.beginPath(); ctx.arc(rzx, rzy, r, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.arc(rzx, rzy, r, 0, Math.PI * 2); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#ef4444';
      ctx.font = 'bold 10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('⛔ RESTRICTED', rzx, rzy - r - 6);
    }

    // Furniture
    for (const obj of Object.values(s.scene.objects)) {
      if (obj.category !== 'furniture') continue;
      const [cx, cy] = toC(obj.posX, obj.posY);
      const w = sc(obj.width);
      const h = sc(obj.height);

      ctx.fillStyle = obj.isContainer ? (obj.isOpen ? COLORS.container_open : COLORS.container_closed) : COLORS.furniture;
      ctx.strokeStyle = '#64748b';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      (ctx as any).roundRect(cx - w / 2, cy - h / 2, w, h, 4);
      ctx.fill(); ctx.stroke();

      ctx.fillStyle = COLORS.label;
      ctx.font = '10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(obj.name, cx, cy + h / 2 + 12);

      if (obj.isContainer) {
        ctx.fillStyle = obj.isOpen ? '#4ade80' : '#f87171';
        ctx.font = '9px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(obj.isOpen ? '🔓' : '🔒', cx, cy - h / 2 - 8);
      }
      if (obj.isContainer && obj.isOpen && obj.contains.length > 0) {
        ctx.fillStyle = '#fbbf24';
        ctx.font = '8px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`[${obj.contains.join(', ')}]`, cx, cy);
      }
      if (obj.standPoseX != null && obj.standPoseY != null) {
        const [sx, sy] = toC(obj.standPoseX, obj.standPoseY);
        ctx.fillStyle = COLORS.stand_pose;
        ctx.globalAlpha = 0.4;
        ctx.beginPath(); ctx.arc(sx, sy, 3, 0, Math.PI * 2); ctx.fill();
        ctx.globalAlpha = 1;
      }
    }

    // Items
    for (const obj of Object.values(s.scene.objects)) {
      if (obj.category !== 'item') continue;
      const [cx, cy] = toC(obj.posX, obj.posY);
      const grad = ctx.createRadialGradient(cx, cy, 2, cx, cy, 8);
      grad.addColorStop(0, COLORS.item);
      grad.addColorStop(1, 'rgba(245, 158, 11, 0)');
      ctx.fillStyle = grad;
      ctx.beginPath(); ctx.arc(cx, cy, 8, 0, Math.PI * 2); ctx.fill();

      ctx.fillStyle = COLORS.item;
      ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = '#fbbf24';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2); ctx.stroke();

      ctx.fillStyle = COLORS.text;
      ctx.font = '9px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(obj.name, cx, cy + 14);
    }
  }

  function drawTrails(ctx: CanvasRenderingContext2D, s: SimulationState, toC: typeof toCanvas) {
    if (!s.robots) return;
    for (const robot of Object.values(s.robots)) {
      const trail = posHistory[robot.name] || [];
      if (trail.length < 2) continue;
      const style = ROBOT_STYLE[robot.robotType] || ROBOT_STYLE.Alice;
      ctx.strokeStyle = style.color + '40';
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      for (const [px, py] of trail) {
        const [cx, cy] = toC(px, py);
        ctx.lineTo(cx, cy);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  function drawRobots(ctx: CanvasRenderingContext2D) {
    for (const [name, p] of Object.entries(animatedPositions)) {
      const s = state?.robots?.[name];
      if (!s) continue;
      const style = ROBOT_STYLE[s.robotType] || ROBOT_STYLE.Alice;
      const [rx, ry] = toCanvas(p.x, p.y);

      // Glow
      const glow = ctx.createRadialGradient(rx, ry, 5, rx, ry, 22);
      glow.addColorStop(0, style.color + '60');
      glow.addColorStop(1, style.color + '00');
      ctx.fillStyle = glow;
      ctx.beginPath(); ctx.arc(rx, ry, 22, 0, Math.PI * 2); ctx.fill();

      // Body
      ctx.fillStyle = style.color;
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;

      switch (style.shape) {
        case 'circle':
          ctx.beginPath(); ctx.arc(rx, ry, 14, 0, Math.PI * 2); break;
        case 'square':
          ctx.beginPath(); (ctx as any).roundRect(rx - 12, ry - 12, 24, 24, 4); break;
        case 'diamond':
          ctx.beginPath();
          ctx.moveTo(rx, ry - 14); ctx.lineTo(rx + 14, ry);
          ctx.lineTo(rx, ry + 14); ctx.lineTo(rx - 14, ry);
          ctx.closePath(); break;
        default: // hexagon
          ctx.beginPath();
          for (let i = 0; i < 6; i++) {
            const a = (Math.PI / 3) * i - Math.PI / 6;
            i === 0 ? ctx.moveTo(rx + 14 * Math.cos(a), ry + 14 * Math.sin(a))
                    : ctx.lineTo(rx + 14 * Math.cos(a), ry + 14 * Math.sin(a));
          }
          ctx.closePath(); break;
      }
      ctx.fill(); ctx.stroke();

      if (s.name === state?.leader) {
        ctx.font = '14px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('👑', rx, ry - 22);
      }

      ctx.fillStyle = '#fff';
      ctx.font = 'bold 10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(s.name, rx, ry + 4);

      if (s.gripperOccupied && s.graspingObject) {
        ctx.fillStyle = '#fbbf24';
        ctx.font = '8px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`📦${s.graspingObject}`, rx, ry - 36);
      }
    }
  }

  function drawOverlays(ctx: CanvasRenderingContext2D, s: SimulationState, toC: typeof toCanvas) {
    // Top-left status panel
    const px = 10, py = 10, pw = 200, ph = 88;
    ctx.fillStyle = 'rgba(30, 41, 59, 0.9)';
    ctx.beginPath(); (ctx as any).roundRect(px, py, pw, ph, 6); ctx.fill();
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.beginPath(); (ctx as any).roundRect(px, py, pw, ph, 6); ctx.stroke();

    ctx.fillStyle = COLORS.text;
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`Step ${s.step}`, px + 10, py + 18);
    ctx.fillStyle = COLORS.label;
    ctx.fillText(s.taskProgress || '', px + 10, py + 36);
    if (s.leader) {
      ctx.fillStyle = COLORS.leader;
      ctx.fillText(`👑 Leader: ${s.leader}`, px + 10, py + 54);
    }
    ctx.fillStyle = COLORS.label;
    ctx.font = '10px Inter, sans-serif';
    ctx.fillText(`Dialogues: ${s.dialogues?.length || 0}`, px + 10, py + 72);

    // Legend
    const lx = 10, ly = CANVAS_H - 100, lw = 180, lh = 90;
    ctx.fillStyle = 'rgba(30, 41, 59, 0.9)';
    ctx.beginPath(); (ctx as any).roundRect(lx, ly, lw, lh, 6); ctx.fill();
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.beginPath(); (ctx as any).roundRect(lx, ly, lw, lh, 6); ctx.stroke();

    let textY = ly + 16;
    ctx.fillStyle = COLORS.label;
    ctx.font = '9px Inter, sans-serif';
    ctx.textAlign = 'left';
    for (const [name, st] of Object.entries(ROBOT_STYLE)) {
      ctx.fillStyle = st.color;
      ctx.beginPath(); ctx.arc(lx + 10, textY, 5, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = COLORS.text;
      ctx.fillText(name, lx + 22, textY + 4);
      textY += 16;
    }
    if (s.leader) {
      ctx.fillStyle = COLORS.leader;
      ctx.fillText(`👑 ${s.leader}`, lx + 22, textY + 4);
    }
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !state?.robots) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    for (const [name, p] of Object.entries(animatedPositions)) {
      const robot = state.robots[name];
      if (!robot) continue;
      const [rx, ry] = toCanvas(p.x, p.y);
      const dist = Math.sqrt((mx - rx) ** 2 + (my - ry) ** 2);
      if (dist < 16) {
        setTooltip({
          x: mx + 14, y: my - 10,
          text: `${robot.name} (${robot.robotType})\nPos: (${robot.posX.toFixed(1)}, ${robot.posY.toFixed(1)})\nGripper: ${robot.graspingObject || 'empty'}`,
        });
        return;
      }
    }
    setTooltip(null);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden', ...style }}>
      <div style={{
        padding: '8px 12px', background: '#1e293b',
        borderBottom: '1px solid #334155', fontSize: 13,
        color: '#94a3b8', fontWeight: 500,
      }}>
        🎮 Simulation View {state?.leader ? `| Leader: ${state.leader}` : ''}
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
