import express from 'express';
import cors from 'cors';
import { createServer } from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import { DynaHMRCEngine } from './dynahmrc.js';
import { RobotType, WSMessage } from './types.js';

const app = express();
app.use(cors());
app.use(express.json());

const server = createServer(app);
const wss = new WebSocketServer({ server });

const engines = new Map<string, DynaHMRCEngine>();
const engineConnections = new Map<string, Set<WebSocket>>();

// === REST API ===

app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', message: 'DynaHMRC API is running' });
});

app.get('/api/config', (_req, res) => {
  res.json({
    taskTypes: [
      { id: 'pack_objects', name: 'Pack Objects', desc: 'Pack bowl, fork, soap, apple into tray' },
      { id: 'sort_solids', name: 'Sort Solids', desc: 'Sort colored solids onto matching panels' },
      { id: 'make_sandwich', name: 'Make Sandwich', desc: 'Stack ingredients on cutting board' },
    ],
    layouts: [
      { id: 'kitchen', name: 'Kitchen', desc: 'Kitchen with fridge, cabinet, tables' },
      { id: 'living_room', name: 'Living Room', desc: 'Living room with sofa, bookshelf, TV stand' },
    ],
    robotTypes: [
      { id: 'Alice', name: 'Alice (Mobile Manipulation)', desc: 'Wheeled chassis + arm, can navigate, open, pick, place' },
      { id: 'Bob', name: 'Bob (Fixed Arm)', desc: 'Desktop robotic arm, precise manipulation' },
      { id: 'David', name: 'David (Mobile)', desc: 'Wheeled robot, navigation and exploration only' },
      { id: 'Lucy', name: 'Lucy (Drone)', desc: 'Quadrotor drone with gripper, aerial navigation and manipulation' },
    ],
    dynamicVariations: [
      { id: 'goal_change', name: 'Goal Change', desc: 'Task target changes mid-execution' },
      { id: 'restricted_zone', name: 'Restricted Zone', desc: 'Area becomes inaccessible' },
      { id: 'team_change', name: 'Team Change', desc: 'Robot joins or leaves team' },
      { id: 'action_constraint', name: 'Action Constraint', desc: 'Some actions become unavailable' },
    ],
    defaultApiKeyConfigured: !!process.env.DEEPSEEK_API_KEY,
  });
});

app.post('/api/run', (req, res) => {
  const { taskType = 'pack_objects', layout = 'kitchen', robots = [],
          dynamicVariations = [], maxSteps = 50 } = req.body;

  const runId = `run_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

  const rtypeMap: Record<string, RobotType> = {
    Alice: RobotType.ALICE,
    Bob: RobotType.BOB,
    David: RobotType.DAVID,
    Lucy: RobotType.LUCY,
  };

  let robotConfigs: Array<[string, RobotType]> = [];
  if (Array.isArray(robots) && robots.length > 0) {
    for (const r of robots) {
      const rtype = rtypeMap[r.name];
      if (rtype) robotConfigs.push([r.name, rtype]);
    }
  }

  if (robotConfigs.length === 0) {
    robotConfigs = [
      ['Alice', RobotType.ALICE],
      ['Bob', RobotType.BOB],
      ['David', RobotType.DAVID],
      ['Lucy', RobotType.LUCY],
    ];
  }

  const engine = new DynaHMRCEngine(taskType, layout, robotConfigs);
  engine.maxSteps = maxSteps;
  engine.dynamicVariations = dynamicVariations;

  engines.set(runId, engine);
  engineConnections.set(runId, new Set());

  res.json({ runId, status: 'created', robots: robotConfigs.length });
});

app.get('/api/run/:runId', (req, res) => {
  const engine = engines.get(req.params.runId);
  if (!engine) {
    return res.status(404).json({ error: `Run ${req.params.runId} not found` });
  }
  res.json({
    runId: req.params.runId,
    stage: engine.stage,
    leader: engine.leader,
    step: engine.stepCount,
    running: engine.running,
    paused: engine.paused,
    completed: engine.sim.taskCompleted,
    dialoguesCount: engine.dialogues.length,
    actionsCount: engine.allActions.length,
  });
});

// === WebSocket ===

wss.on('connection', (ws, req) => {
  const url = new URL(req.url || '', 'http://localhost');
  const pathParts = url.pathname.split('/');
  const runId = pathParts[pathParts.length - 1];

  if (!runId || !engines.has(runId)) {
    ws.send(JSON.stringify({ type: 'error', data: { message: `Run ${runId} not found` } }));
    ws.close();
    return;
  }

  const engine = engines.get(runId)!;
  const connections = engineConnections.get(runId)!;
  connections.add(ws);

  engine.setUpdateCallback(async (msg: WSMessage) => {
    const deadConns: WebSocket[] = [];
    for (const conn of connections) {
      try {
        conn.send(JSON.stringify(msg));
      } catch {
        deadConns.push(conn);
      }
    }
    for (const dc of deadConns) connections.delete(dc);
  });

  ws.on('message', (data) => {
    try {
      const msg = JSON.parse(data.toString());
      const { command } = msg;

      switch (command) {
        case 'start':
          (async () => { await engine.run(); })();
          ws.send(JSON.stringify({ type: 'control', data: { status: 'started', runId } }));
          break;
        case 'pause':
          engine.pause();
          ws.send(JSON.stringify({ type: 'control', data: { status: 'paused' } }));
          break;
        case 'resume':
          engine.resume();
          ws.send(JSON.stringify({ type: 'control', data: { status: 'resumed' } }));
          break;
        case 'stop':
          engine.stop();
          ws.send(JSON.stringify({ type: 'control', data: { status: 'stopped' } }));
          break;
        case 'get_state': {
          const state = engine.sim.getState();
          state.stage = engine.stage;
          state.leader = engine.leader;
          state.collaborationPlan = engine.collaborationPlan;
          state.dialogues = engine.dialogues;
          state.actions = engine.allActions;
          ws.send(JSON.stringify({ type: 'state', data: state as any }));
          break;
        }
      }
    } catch (e) {
      console.error('[WS Parse Error]', e);
    }
  });

  ws.on('close', () => {
    connections.delete(ws);
  });
});

// === Start ===

const PORT = parseInt(process.env.PORT || '3001', 10);
server.listen(PORT, () => {
  console.log(`[DynaHMRC Server] Running on http://localhost:${PORT}`);
  console.log(`[DynaHMRC Server] WebSocket ws://localhost:${PORT}/ws/<runId>`);
  console.log(`[DynaHMRC Server] DeepSeek API: ${process.env.DEEPSEEK_API_KEY ? 'Configured' : 'NOT configured (using mock)'}`);
});
