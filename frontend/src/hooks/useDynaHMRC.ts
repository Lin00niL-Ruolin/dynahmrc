import { useEffect, useRef, useCallback, useState } from 'react';
import type { SimulationState, RobotDialogue } from '../types';

export function useDynaHMRC() {
  const [connected, setConnected] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const [state, setState] = useState<SimulationState | null>(null);
  const [dialogues, setDialogues] = useState<RobotDialogue[]>([]);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const dialogRef = useRef<RobotDialogue[]>([]);

  const connectWebSocket = useCallback((id: string, autoStart = false) => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    // 通过当前 host 连接（后端也提供了前端静态文件，同端口）
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${id}`;

    console.log('[WS] Connecting to:', wsUrl);
    console.log('[WS] Location:', window.location.href);
    console.log('[WS] Protocol:', window.location.protocol);
    console.log('[WS] Host:', window.location.host);

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setRunId(id);
      // Auto-start the engine
      if (autoStart) {
        console.log('[WS] Auto-starting engine...');
        ws.send(JSON.stringify({ command: 'start' }));
      }
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        console.log('[WS] Received:', msg.type, msg.data?.stage || '');

        switch (msg.type) {
          case 'state':
            setState(msg.data);
            setDialogues(dialogRef.current);
            break;
          case 'dialogue':
            dialogRef.current = [...dialogRef.current, msg.data];
            setDialogues(dialogRef.current);
            break;
          case 'control':
            console.log('[Control]', msg.data);
            break;
          case 'error':
            setError(msg.data.message);
            break;
        }
      } catch (e) {
        console.error('[WS Parse]', e);
      }
    };

    ws.onclose = () => {
      setConnected(false);
    };

    ws.onerror = (e) => {
      console.error('[WS Error]', e);
      setError('WebSocket connection error - check browser console for details');
    };

    return ws;
  }, []);

  const [debugInfo, setDebugInfo] = useState<string[]>([]);

  const addDebug = (msg: string) => {
    console.log('[DynaDEBUG]', msg);
    setDebugInfo(prev => [...prev.slice(-9), msg]);
  };

  const createRun = useCallback(async (config: {
    taskType?: string;
    layout?: string;
    robots?: string[];
    dynamicVariations?: string[];
    maxSteps?: number;
  }) => {
    addDebug('Creating run...');
    try {
      addDebug(`POST /api/run config=${config.taskType} layout=${config.layout}`);
      const resp = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          taskType: config.taskType || 'pack_objects',
          layout: config.layout || 'kitchen',
          robots: (config.robots || ['Alice', 'Bob', 'David', 'Lucy']).map(name => ({ name })),
          dynamicVariations: config.dynamicVariations || [],
          maxSteps: config.maxSteps || 50,
        }),
      });

      if (!resp.ok) {
        const text = await resp.text();
        addDebug(`API error: HTTP ${resp.status} ${text}`);
        throw new Error(`HTTP ${resp.status}: ${text}`);
      }
      const data = await resp.json();
      addDebug(`Run created: ${data.runId}`);

      // Reset state
      dialogRef.current = [];
      setDialogues([]);
      setState(null);
      setError(null);

      // Connect WebSocket with auto-start
      addDebug(`Connecting WS: ${window.location.host}/ws/${data.runId}`);
      connectWebSocket(data.runId, true);
      return data.runId;
    } catch (e: any) {
      addDebug(`Error: ${e.message}`);
      setError(e.message);
      return null;
    }
  }, [connectWebSocket]);

  const sendCommand = useCallback((command: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command }));
    }
  }, []);

  const start = useCallback(() => sendCommand('start'), [sendCommand]);
  const pause = useCallback(() => sendCommand('pause'), [sendCommand]);
  const resume = useCallback(() => sendCommand('resume'), [sendCommand]);
  const stop = useCallback(() => sendCommand('stop'), [sendCommand]);

  const loadConfig = useCallback(async () => {
    try {
      const resp = await fetch('/api/config');
      return await resp.json();
    } catch {
      return null;
    }
  }, []);

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return {
    connected,
    runId,
    state,
    dialogues,
    error,
    debugInfo,
    createRun,
    start,
    pause,
    resume,
    stop,
    loadConfig,
  };
}
