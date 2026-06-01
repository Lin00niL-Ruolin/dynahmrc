/**
 * 场景二 (Scenario 2): Sort Solids - 绿色小方块分类任务
 * 对应场景: kitchen (BestMan scene2) L型厨房
 * 对应任务: sort_solids
 * 
 * 机器人协作流程:
 * 1. Alice (移动机械臂) → 移动到书架下方指引位置
 * 2. Lucy (无人机)     → 飞到书架上拿取绿色小方块
 * 3. Lucy             → 飞到 Bob 所在的桌子 (table2) 放下小方块
 * 4. Bob (固定机械臂)   → 捡起小方块放到颜色一致的大方块 (cube_green) 上面
 */

import { RobotType } from '../types.js';
import { DynaHMRCEngine } from '../dynahmrc.js';
import { SimEnvironment } from '../simulation.js';

/**
 * 场景二配置定义
 */
export const SCENARIO_2_CONFIG = {
  id: 'scenario2',
  name: '场景二: 绿色方块分类',
  description: 'Alice引导 → Lucy空中取物 → Bob精准放置',
  taskType: 'sort_solids' as const,
  layout: 'kitchen',
  targetCube: 'green',          // 本次场景目标颜色
  smallCubeName: 'small_cube_green',
  largeCubeName: 'cube_green',
  workflow: [
    { step: 1, actor: 'Alice', action: 'navigate', target: 'bookcase', purpose: '移动到书架下占据有利位置' },
    { step: 2, actor: 'Lucy',  action: 'navigate', target: 'bookcase', purpose: '飞到书架上方准备取物' },
    { step: 3, actor: 'Lucy',  action: 'pick',     target: 'small_cube_green', purpose: '从书架上拿取绿色小方块' },
    { step: 4, actor: 'Lucy',  action: 'navigate', target: 'table2',   purpose: '飞到 Bob 的工作台' },
    { step: 5, actor: 'Lucy',  action: 'place',    object: 'small_cube_green', target: "Bob's table", purpose: '将小方块放到 Bob 桌子上' },
    { step: 6, actor: 'Bob',   action: 'pick',     target: 'small_cube_green', purpose: 'Bob 捡起绿色小方块' },
    { step: 7, actor: 'Bob',   action: 'place',    object: 'small_cube_green', target: 'cube_green', purpose: '将小方块放到大的绿色方块上完成分类' },
  ],
};

/**
 * 运行场景二
 * 创建一个独立的 DynaHMRC 实例，并覆盖 sort_solids 的随机行为，
 * 固定为只匹配绿色方块。
 */
export async function runScenario2(
  onUpdate?: (msg: any) => Promise<void>,
  maxSteps: number = 50,
): Promise<DynaHMRCEngine> {
  // 创建引擎并设置回调
  const engine = new DynaHMRCEngine('sort_solids', 'kitchen', [
    ['Alice', RobotType.ALICE],
    ['Bob', RobotType.BOB],
    ['David', RobotType.DAVID],
    ['Lucy', RobotType.LUCY],
  ]);
  engine.maxSteps = maxSteps;

  if (onUpdate) {
    engine.setUpdateCallback(onUpdate);
  }

  // === 覆盖引擎的 sort_solids 随机行为，固定为绿色方块 ===
  // 设置场景的任务描述为固定绿色配对
  const greenPair = { small: 'small_cube_green', large: 'cube_green' };
  const pairText = `${greenPair.small} → ${greenPair.large}`;
  engine.taskDescription = `Match 1 pair: ${pairText}. 
Alice navigates to guide near the bookshelf. 
Lucy flies to the bookcase, picks up the green small cube, and brings it to Bob's table (table2). 
Bob places it on the matching large green cube (cube_green).
This is a scripted scenario: mobile robots explore, Lucy does aerial retrieval, Bob does precision placement.`;

  // 覆盖任务目标为只有 small_cube_green
  const prompts = await import('../prompts.js');
  prompts.TASK_DESCRIPTIONS['sort_solids'] = engine.taskDescription;
  prompts.TASK_GOALS['sort_solids'] = ['small_cube_green'];
  engine.sim.taskTargets = ['small_cube_green'];

  // 场景二专用提示
  prompts.TASK_ALLOCATION_HINTS['sort_solids'] = `This is Scenario 2 — Green Cube Sorting.
The target is a single green small cube (small_cube_green) on the bookcase shelf.
A large green cube (cube_green) is on table_2 where Bob works.

Expected collaboration:
- Alice: navigate to bookcase area, scout the environment
- Lucy (drone): fly to the bookcase, pick small_cube_green, fly to table_2, place on Bob's table
- David: explore the area and provide environment information
- Bob: pick small_cube_green from his table, place on cube_green

This is a precision sorting task requiring aerial retrieval and precise placement.`;

  prompts.REFLECTION_TASK_HINTS['sort_solids'] = `Task: Place small_cube_green on cube_green. 
Lucy must retrieve the small cube from the bookcase and bring it to Bob's table.
Bob places it on the large matching cube.`;

  prompts.LEADER_REFLECTION_HINTS['sort_solids'] = `Match small_cube_green → cube_green. 
Lucy is the aerial retrieval specialist. Bob does precision placement.`;

  console.log(`[Scenario 2] Green cube sort task configured: ${pairText}`);

  return engine;
}

/**
 * 回调构造器 - 用于 WebSocket 广播
 */
export function createScenario2Callback(ws: { send: (data: string) => void }): (msg: any) => Promise<void> {
  return async (msg: any) => {
    try {
      ws.send(JSON.stringify(msg));
    } catch {
      // connection closed
    }
  };
}

/**
 * 验证场景二的配置完整性
 */
export function validateScenario2(): string[] {
  const issues: string[] = [];
  
  // 检查场景对象
  const sim = new SimEnvironment('kitchen');
  sim.reset('sort_solids', [
    ['Alice', RobotType.ALICE],
    ['Bob', RobotType.BOB],
    ['David', RobotType.DAVID],
    ['Lucy', RobotType.LUCY],
  ]);

  // 验证必要对象存在
  const requiredObjects = [
    'bookcase',
    'table2',
    'small_cube_green',
    'cube_green',
    'cube_red', 'cube_blue', 'cube_yellow', 'cube_purple', 'cube_orange',
  ];

  for (const name of requiredObjects) {
    if (!sim.scene.objects[name]) {
      issues.push(`Missing object: ${name}`);
    }
  }

  // 验证 small_cube_green 在 bookshelf 位置
  const greenCube = sim.scene.objects['small_cube_green'];
  const bookcase = sim.scene.objects['bookcase'];
  if (greenCube && bookcase) {
    const dx = Math.abs(greenCube.posX - bookcase.posX);
    const dy = Math.abs(greenCube.posY - bookcase.posY);
    if (dx > 1.0 || dy > 1.0) {
      issues.push(`small_cube_green (${greenCube.posX}, ${greenCube.posY}) is far from bookcase (${bookcase.posX}, ${bookcase.posY})`);
    }
  }

  // 验证 cube_green 在 table2 位置
  const largeGreen = sim.scene.objects['cube_green'];
  const table2 = sim.scene.objects['table2'];
  if (largeGreen && table2) {
    const dx = Math.abs(largeGreen.posX - table2.posX);
    const dy = Math.abs(largeGreen.posY - table2.posY);
    if (dx > 1.0 || dy > 1.0) {
      issues.push(`cube_green (${largeGreen.posX}, ${largeGreen.posY}) is far from table2 (${table2.posX}, ${table2.posY})`);
    }
  }

  // 验证 Bob 位置在 table2 旁
  const bobPos = sim.robotPositions['Bob'];
  if (bobPos && table2) {
    const dx = Math.abs(bobPos[0] - table2.posX);
    const dy = Math.abs(bobPos[1] - table2.posY);
    if (dx > 2.0 || dy > 2.0) {
      issues.push(`Bob (${bobPos[0]}, ${bobPos[1]}) is far from table2 (${table2.posX}, ${table2.posY})`);
    }
  }

  return issues;
}
