/**
 * BestMan 3D 仿真桥接模块
 * 
 * 功能：
 * 1. 管理 BestMan Python 微服务（启动/停止）
 * 2. 将 DynaHMRC 引擎的动作（navigate/pick/place）转发给 BestMan
 * 3. 从 BestMan 获取状态并返回给引擎
 */

import { execSync, ChildProcess, spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import http from 'http';

const BESTMAN_SERVICE_PORT = 5001;
const BESTMAN_SERVICE_URL = `http://localhost:${BESTMAN_SERVICE_PORT}`;

let bestmanProcess: ChildProcess | null = null;
let serviceReady = false;

/**
 * 获取 bestman-service 目录路径
 */
function getServiceDir(): string {
  const serverDir = path.dirname(new URL(import.meta.url).pathname);
  return path.resolve(serverDir, '../../bestman-service');
}

/**
 * 检查 Python 和必要的包是否可用
 */
export function checkEnvironment(): { ok: boolean; message: string } {
  const pythonPath = '/home/developer/miniconda/bin/python3';
  
  try {
    // 检查 Python
    const pyVer = execSync(`${pythonPath} --version`, { encoding: 'utf-8' }).trim();
    
    // 检查 pybullet
    try {
      execSync(`${pythonPath} -c "import pybullet" 2>/dev/null`, { encoding: 'utf-8' });
    } catch {
      return { ok: false, message: `PyBullet not installed in ${pythonPath}` };
    }
    
    // 检查 fastapi
    try {
      execSync(`${pythonPath} -c "import fastapi; import uvicorn" 2>/dev/null`, { encoding: 'utf-8' });
    } catch {
      return { ok: false, message: `FastAPI/Uvicorn not installed.` };
    }
    
    return { ok: true, message: `${pyVer} (miniconda)` };
  } catch (e: any) {
    return { ok: false, message: `Python not found: ${e.message}` };
  }
}

/**
 * 启动 BestMan 微服务
 */
export async function startService(scene: string = 'scene1', gui: boolean = true): Promise<boolean> {
  if (serviceReady) {
    console.log('[BestMan] Service already running');
    return true;
  }

  const envCheck = checkEnvironment();
  if (!envCheck.ok) {
    console.error('[BestMan]', envCheck.message);
    return false;
  }

  const serviceDir = getServiceDir();
  const serviceScript = path.join(serviceDir, 'service.py');

  if (!fs.existsSync(serviceScript)) {
    console.error(`[BestMan] Service script not found: ${serviceScript}`);
    return false;
  }

  return new Promise((resolve) => {
    console.log(`[BestMan] Starting service on port ${BESTMAN_SERVICE_PORT}...`);
    
    const pythonPath = '/home/developer/miniconda/bin/python3';
    bestmanProcess = spawn(pythonPath, [serviceScript], {
      cwd: serviceDir,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });

    let output = '';
    
    bestmanProcess.stdout?.on('data', (data: Buffer) => {
      const text = data.toString();
      output += text;
      process.stdout.write(`[BestMan] ${text}`);
    });

    bestmanProcess.stderr?.on('data', (data: Buffer) => {
      const text = data.toString();
      output += text;
      process.stderr.write(`[BestMan:err] ${text}`);
    });

    bestmanProcess.on('close', (code) => {
      console.log(`[BestMan] Process exited with code ${code}`);
      serviceReady = false;
      bestmanProcess = null;
    });

    // 等待服务启动（最多 15 秒）
    let attempts = 0;
    const maxAttempts = 30;
    const checkInterval = setInterval(async () => {
      attempts++;
      try {
        const resp = await fetch(`${BESTMAN_SERVICE_URL}/`);
        if (resp.ok) {
          clearInterval(checkInterval);
          serviceReady = true;
          console.log(`[BestMan] Service started ✅ (port ${BESTMAN_SERVICE_PORT})`);

          // 初始化场景
          const initResult = await initScene(scene, gui);
          if (initResult) {
            resolve(true);
          } else {
            console.error('[BestMan] Scene initialization failed');
            resolve(false);
          }
        }
      } catch {
        // 服务还没就绪，继续等待
      }

      if (attempts >= maxAttempts) {
        clearInterval(checkInterval);
        console.error(`[BestMan] Service failed to start after ${maxAttempts} attempts`);
        console.error(`[BestMan] Output: ${output.slice(-500)}`);
        resolve(false);
      }
    }, 500);
  });
}

/**
 * 初始化场景
 */
async function initScene(scene: string, gui: boolean): Promise<boolean> {
  try {
    const resp = await fetch(`${BESTMAN_SERVICE_URL}/init`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scene, gui }),
    });
    const data = await resp.json() as any;
    console.log(`[BestMan] Scene initialized: ${data.message}`);
    return true;
  } catch (e: any) {
    console.error(`[BestMan] Init error: ${e.message}`);
    return false;
  }
}

/**
 * 发送动作到 BestMan
 */
export async function sendAction(
  robotName: string,
  actionType: string,
  params: Record<string, any> = {}
): Promise<{ success: boolean; message: string }> {
  if (!serviceReady) {
    return { success: false, message: 'BestMan service not running' };
  }

  try {
    // 映射机器人名称到 BestMan 中的 ID
    const robotMap: Record<string, string> = {
      'Alice': 'alice_base',
      'Bob': 'bob_arm',
      'David': 'david',
      'Lucy': 'drone_body',
    };

    const robotId = robotMap[robotName] || robotName;

    const resp = await fetch(`${BESTMAN_SERVICE_URL}/act`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        robot_id: robotId,
        action: actionType,
        params,
      }),
    });

    const result = await resp.json() as any;
    return { success: result.success !== false, message: result.message || '' };
  } catch (e: any) {
    return { success: false, message: `BestMan action error: ${e.message}` };
  }
}

/**
 * 停止 BestMan 服务
 */
export function stopService(): void {
  if (bestmanProcess) {
    console.log('[BestMan] Stopping service...');
    bestmanProcess.kill('SIGTERM');
    bestmanProcess = null;
    serviceReady = false;
  }
}

/**
 * 获取服务状态
 */
export function getStatus(): { running: boolean; port: number } {
  return { running: serviceReady, port: BESTMAN_SERVICE_PORT };
}
