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

  const connectWebSocket = useCallback((id: string) => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.hostname}:3001/ws/${id}`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setRunId(id);
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);

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
      setError('WebSocket connection error');
    };

    return ws;
  }, []);

  const createRun = useCallback(async (config: {
    taskType?: string;
    layout?: string;
    robots?: string[];
    dynamicVariations?: string[];
    maxSteps?: number;
  }) => {
    try {
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

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();

      // Reset state
      dialogRef.current = [];
      setDialogues([]);
      setState(null);
      setError(null);

      // Connect WebSocket
      connectWebSocket(data.runId);
      return data.runId;
    } catch (e: any) {
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
    createRun,
    start,
    pause,
    resume,
    stop,
    loadConfig,
  };
}
