/**
 * 场景二: 命令行运行器
 * 
 * 直接运行: npx tsx src/run-scenario2.ts
 * 不经过 WebSocket，独立运行，输出详细日志到控制台
 */

// 优先加载 .env
import { readFileSync } from 'fs';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
try {
  const __dirname = dirname(fileURLToPath(import.meta.url));
  const envPath = resolve(__dirname, '../.env');
  const envContent = readFileSync(envPath, 'utf-8');
  for (const line of envContent.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const eqIdx = trimmed.indexOf('=');
    if (eqIdx > 0) {
      const key = trimmed.slice(0, eqIdx).trim();
      const value = trimmed.slice(eqIdx + 1).trim();
      if (!process.env[key]) {
        process.env[key] = value;
      }
    }
  }
  console.log('[ENV] Loaded .env file');
} catch {
  console.warn('[ENV] No .env file found, using mock LLM');
}

import { runScenario2, validateScenario2, SCENARIO_2_CONFIG } from './scenarios/scenario2.js';

async function main() {
  console.log('========================================');
  console.log(`  ${SCENARIO_2_CONFIG.name}`);
  console.log(`  ${SCENARIO_2_CONFIG.description}`);
  console.log(`  Task: ${SCENARIO_2_CONFIG.taskType} | Layout: ${SCENARIO_2_CONFIG.layout}`);
  console.log(`  Target: ${SCENARIO_2_CONFIG.smallCubeName} → ${SCENARIO_2_CONFIG.largeCubeName}`);
  console.log('========================================\n');

  // 1. 验证场景配置
  console.log('[1/4] Validating scenario configuration...');
  const issues = validateScenario2();
  if (issues.length > 0) {
    console.warn('  ⚠️  Validation issues:');
    for (const issue of issues) {
      console.warn(`     - ${issue}`);
    }
  } else {
    console.log('  ✅ All objects validated');
  }
  console.log('');

  // 2. 打印预期工作流
  console.log('[2/4] Expected robot collaboration workflow:');
  for (const step of SCENARIO_2_CONFIG.workflow) {
    console.log(`  Step ${step.step}: ${step.actor} → ${step.action}(${step.target}) — ${step.purpose}`);
  }
  console.log('');

  // 3. 创建引擎
  console.log('[3/4] Creating DynaHMRC engine...');
  const engine = await runScenario2(async (msg) => {
    const data = msg.data || {};
    switch (msg.type) {
      case 'dialogue': {
        const d = data as any;
        const stage = d.stage || '';
        const robot = d.robotName || '';
        const content = (d.content || '').replace(/\n/g, '\n        ');
        
        const stageLabels: Record<string, string> = {
          self_description: '📋 自我介绍',
          task_allocation_bidding: '📝 分工提议',
          leader_election: '🗳️ 投票',
          execution_reflection: '🤖 执行反馈',
          completed: '✅ 完成',
        };
        
        if (d.stage === 'self_description') {
          console.log(`  [${stageLabels[stage] || stage}] ${robot}: ${(d.content || '').slice(0, 120)}...`);
        } else if (d.stage === 'task_allocation_bidding') {
          console.log(`  [${stageLabels[stage] || stage}] ${robot}: ${(d.content || '').slice(0, 150)}...`);
        } else if (d.stage === 'leader_election') {
          const vote = d.vote ? ` → 投票: ${d.vote}` : '';
          console.log(`  [${stageLabels[stage] || stage}] ${robot}${vote}`);
        } else if (stage === 'execution_reflection') {
          if (robot === '[SYSTEM]') {
            console.log(`  [📢 系统] ${content}`);
          } else if (content.includes('[REFLECTION]')) {
            // Skip detailed reflection in concise mode — print only summary
            const summary = content.split('\n')[1]?.replace('Summary: ', '') || '';
            console.log(`  [🔍 反思] ${robot}: ${summary.slice(0, 80)}`);
          } else if (content.includes('[LEADER UPDATE]')) {
            const plan = content.split('\n').slice(1).join(' ').slice(0, 100);
            console.log(`  [👑 领导更新] ${robot}: ${plan}`);
          } else {
            console.log(`  [🤖 ${robot}] ${content.slice(0, 120)}`);
          }
        }
        break;
      }
      case 'state': {
        const s = data as any;
        if (s.stage && s.stage !== 'execution_reflection') {
          const stageLabels: Record<string, string> = {
            self_description: '📋',
            task_allocation_bidding: '📝',
            leader_election: '🗳️',
            execution_reflection: '🤖',
            completed: '✅',
            stopped: '⏹️',
          };
          console.log(`\n  [🔄 阶段] ${stageLabels[s.stage] || '—'} ${s.stage}`);
          if (s.leader) console.log(`  [👑 领导] ${s.leader}`);
          if (s.stage === 'completed') {
            console.log(`  ${s.taskCompleted ? '✅ 任务完成!' : '⚠️ 部分完成'} (${s.taskProgress})`);
          }
        }
        break;
      }
      case 'error':
        console.error('  ❌', data);
        break;
    }
  });
  console.log('');

  // 4. 运行
  console.log('[4/4] Running DynaHMRC...\n');
  const startTime = Date.now();
  
  try {
    await engine.run();
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`\n  ⏱️  Total time: ${elapsed}s`);
    console.log(`  📊 Steps: ${engine.stepCount}`);
    console.log(`  🎯 Task: ${engine.sim.taskCompleted ? '✅ Completed' : '⚠️ Partial'}`);
    console.log(`  📦 Placed: ${engine.sim.placedObjects.length}/${engine.sim.taskTargets.length}`);
    console.log(`  🏆 Leader: ${engine.leader}`);
  } catch (e: any) {
    console.error('\n  ❌ Run failed:', e.message);
  }
}

main().catch(console.error);
