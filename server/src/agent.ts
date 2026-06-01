import {
  RobotType, ActionType, RobotStatus, RobotAction, Feedback,
  RobotMessage, SceneGraph, LLMMessage, DynaHMRCStage, RobotDialogue,
} from './types.js';
import { getLLMClient, DeepSeekClient } from './llm.js';
import * as prompts from './prompts.js';

export class RobotAgent {
  name: string;
  robotType: RobotType;
  taskType: string;
  status: RobotStatus;
  private llm: DeepSeekClient;
  feedbackHistory: Feedback[] = [];
  actionHistory: RobotAction[] = [];
  receivedMessages: RobotMessage[] = [];
  selfDescription = '';
  taskPlan = '';
  campaignSpeech = '';
  currentPlan = '';
  failedPickItems: Map<string, number> = new Map(); // item → stepCount when failed
  pendingPlaceWait = 0; // steps to wait after placing on Bob's table
  /** 上一步操作是否由反循环逻辑强制注入（防止无限循环） */
  private _lastActionWasForced = false;
  /** 连续被强制注入的次数（超过阈值时触发应急策略） */
  private _consecutiveForcedCount = 0;

  private roleTypeName: string;

  constructor(name: string, robotType: RobotType, taskType: string) {
    this.name = name;
    this.robotType = robotType;
    this.taskType = taskType;
    this.llm = getLLMClient();
    this.status = {
      name,
      robotType,
      posX: 0,
      posY: 0,
      gripperOccupied: false,
      graspingObject: null,
      reachableRange: 0.5,
    };
    const map: Record<string, string> = {
      [RobotType.ALICE]: 'Alice',
      [RobotType.BOB]: 'Bob',
      [RobotType.DAVID]: 'David',
      [RobotType.LUCY]: 'Lucy',
    };
    this.roleTypeName = map[robotType];
  }

  getCapabilityDescription(): string {
    return prompts.ROBOT_CAPABILITIES[this.roleTypeName] || '';
  }

  getPrinciples(): string {
    return prompts.PRINCIPLES[this.roleTypeName] || '';
  }

  getTaskDescription(): string {
    return prompts.TASK_DESCRIPTIONS[this.taskType] || this.taskType;
  }

  async selfDescribe(teammates: string[]): Promise<[string, string]> {
    const msgs: LLMMessage[] = [
      { role: 'system', content: prompts.SELF_DESCRIPTION_SYSTEM },
      {
        role: 'user',
        content: prompts.selfDescriptionUser(
          this.getTaskDescription(),
          teammates.join(', '),
          this.getCapabilityDescription(),
          this.taskType,
        ),
      },
    ];
    const response = await this.llm.chat(msgs);
    this.selfDescription = response.content;
    return [response.thoughts, response.content];
  }

  async proposePlanAndCampaign(selfIntroductions: string): Promise<[string, string]> {
    const msgs: LLMMessage[] = [
      { role: 'system', content: prompts.TASK_ALLOCATION_SYSTEM },
      {
        role: 'user',
        content: prompts.taskAllocationUser(this.name, selfIntroductions, this.taskType),
      },
    ];
    const response = await this.llm.chat(msgs);
    this.taskPlan = response.content;
    this.campaignSpeech = response.content;
    return [response.thoughts, response.content];
  }

  async voteLeader(plansAndSpeeches: string): Promise<[string, string, string]> {
    const msgs: LLMMessage[] = [
      { role: 'system', content: prompts.LEADER_ELECTION_SYSTEM },
      {
        role: 'user',
        content: prompts.leaderElectionUser(this.name, plansAndSpeeches),
      },
    ];
    const response = await this.llm.chat(msgs);
    const leader = this.extractLeader(response.content);
    return [response.thoughts, response.content, leader];
  }

  async act(
    leader: string,
    plan: string,
    scene: SceneGraph,
    taskProgress: string,
    taskTargets?: string[],
    placedObjects?: string[],
    sharedStatus?: string,
  ): Promise<[string, RobotAction]> {
    const sceneGraphStr = this.sceneGraphToText(scene);
    const feedbackStr = this.feedbackToText();
    const actionStr = this.actionHistoryToText();
    const messageStr = this.messagesToText();
    const gripperStatus = this.status.gripperOccupied ? 'occupied' : 'empty';
    const graspObj = this.status.graspingObject || 'nothing';

    const sysContent = prompts.executionSystem(
      this.getCapabilityDescription(),
      this.getTaskDescription(),
      '',
      leader,
      plan,
      this.getPrinciples(),
      this.taskType,
      this.roleTypeName,
      this.status.gripperOccupied,
    );

    const userContent = prompts.executionUser(
      sceneGraphStr,
      this.status.posX,
      this.status.posY,
      gripperStatus,
      graspObj,
      feedbackStr || 'No previous feedback.',
      actionStr || 'No previous actions.',
      messageStr || 'No messages received.',
      taskProgress,
      taskTargets || prompts.TASK_GOALS[this.taskType],
      placedObjects || [],
      sharedStatus,
    );

    const msgs: LLMMessage[] = [
      { role: 'system', content: sysContent },
      { role: 'user', content: userContent },
    ];

    const response = await this.llm.chat(msgs);
    let action = this.parseAction(response.content);

    // ⛔ 机器人能力约束：不允许的操作直接拦截
    const FORBIDDEN_ACTIONS: Record<string, ActionType[]> = {
      'David': [ActionType.PICK, ActionType.PLACE, ActionType.OPEN, ActionType.MOVE],
      'Lucy': [ActionType.OPEN],
      'Alice': [],
      'Bob': [ActionType.NAVIGATE, ActionType.OPEN, ActionType.MOVE],
    };
    const blocked = FORBIDDEN_ACTIONS[this.roleTypeName] || [];
    if (blocked.includes(action.actionType)) {
      console.log(`[Agent ${this.name}] Capability constraint: ${this.roleTypeName} cannot ${action.actionType}. Forcing communicate.`);
      action = {
        robotName: this.name, actionType: ActionType.COMMUNICATE,
        params: { content: `I cannot ${action.actionType}. ${
          this.roleTypeName === 'David' ? 'I am a navigation-only robot. Please handle object manipulation.' : ''
        }`, recipient: '*' },
        timestamp: Date.now(),
      };
    }

    // ⛔ Mobile robots: never wait or communicate when actionable steps remain
    if (this.roleTypeName !== 'Bob') {
      const lastType = this.actionHistory.length > 0 ? this.actionHistory[this.actionHistory.length - 1].actionType : null;
      const lastTarget = this.actionHistory.length > 0 ? (this.actionHistory[this.actionHistory.length - 1].params.target as string || '') : '';
      
      // 应急策略：连续强制注入超过阈值后，直接执行有意义的动作
      // 这样即使 LLM 一直输出 wait() 也能有进展
      if (this._lastActionWasForced && this._consecutiveForcedCount >= 3) {
        console.log(`[Agent ${this.name}] ⚠️ Emergency: ${this._consecutiveForcedCount} consecutive forced actions. Taking initiative.`);
        
        if (this.status.gripperOccupied && this.status.graspingObject) {
          // 持有物品 → 送到 Bob 桌子
          action = {
            robotName: this.name, actionType: ActionType.NAVIGATE,
            params: { target: 'table_new_2' }, timestamp: Date.now(),
          };
        } else {
          // 空手 → 去书架（场景二目标位置）或 Bob 桌子
          const targets = ['table_new_1', 'table2', 'bookcase'];
          const idx = this._consecutiveForcedCount % targets.length;
          action = {
            robotName: this.name, actionType: ActionType.NAVIGATE,
            params: { target: targets[idx] }, timestamp: Date.now(),
          };
        }
        this._consecutiveForcedCount = 0;
        this._lastActionWasForced = false;
      }
      
      const isLLMOutput = !this._lastActionWasForced;
      
      // 阻止重复通信（仅 LLM 自然输出）
      if (isLLMOutput && action.actionType === ActionType.COMMUNICATE && lastType === ActionType.COMMUNICATE) {
        console.log(`[Agent ${this.name}] Repeated communicate(). Guiding to productive action.`);
        action = { robotName: this.name, actionType: ActionType.NAVIGATE, params: { target: 'table_new_1' }, timestamp: Date.now() };
        this._lastActionWasForced = true;
        this._consecutiveForcedCount++;
      }
      // 阻止重复导航到同一个目标
      else if (isLLMOutput && action.actionType === ActionType.NAVIGATE && lastTarget && action.params.target === lastTarget) {
        console.log(`[Agent ${this.name}] Repeated navigate to ${lastTarget}. Switching target.`);
        action = { robotName: this.name, actionType: ActionType.NAVIGATE, params: { target: 'table_new_2' }, timestamp: Date.now() };
        this._lastActionWasForced = true;
        this._consecutiveForcedCount++;
      }
      // 阻止 wait (核心防卡死)
      else if (action.actionType === ActionType.WAIT) {
        if (this.status.gripperOccupied && this.status.graspingObject) {
          console.log(`[Agent ${this.name}] Holding ${this.status.graspingObject} but waiting. Forcing go to Bob's table.`);
          action = { robotName: this.name, actionType: ActionType.NAVIGATE, params: { target: 'table_new_2' }, timestamp: Date.now() };
        } else {
          console.log(`[Agent ${this.name}] Blocked wait(). Guiding to explore.`);
          action = { robotName: this.name, actionType: ActionType.NAVIGATE, params: { target: 'table_new_1' }, timestamp: Date.now() };
        }
        this._lastActionWasForced = true;
        this._consecutiveForcedCount++;
      }
      // LLM 自然输出（有意义的动作）→ 重置强制计数器
      else if (isLLMOutput) {
        this._lastActionWasForced = false;
        this._consecutiveForcedCount = 0;
      }
    }

    // ⛔ Bob: never communicate twice in a row — force wait instead
    if (this.roleTypeName === 'Bob' && action.actionType === ActionType.COMMUNICATE) {
      const lastType = this.actionHistory.length > 0 ? this.actionHistory[this.actionHistory.length - 1].actionType : null;
      if (lastType === ActionType.COMMUNICATE) {
        console.log(`[Agent ${this.name}] Repeated communicate(). Forcing wait.`);
        action = {
          robotName: this.name, actionType: ActionType.WAIT,
          params: {}, timestamp: Date.now(),
        };
      }
    }

    // ⛔ Bob: never try to pick items he recently failed to reach
    const pickTarget = action.params.object as string || '';
    if (this.roleTypeName === 'Bob' && action.actionType === ActionType.PICK && !this.status.gripperOccupied) {
      // Expire old failures (>5 actions ago — item may have moved)
      const failedAgo = this.failedPickItems.get(pickTarget);
      if (failedAgo !== undefined) {
        const currentStep = this.actionHistory.length;
        if (currentStep - failedAgo < 1) {
          console.log(`[Agent ${this.name}] Blocked pick(${pickTarget}) — failed ${currentStep - failedAgo} steps ago. Forcing wait.`);
          action = { robotName: this.name, actionType: ActionType.WAIT, params: {}, timestamp: Date.now() };
        } else {
          // Expired — allow retry
          this.failedPickItems.delete(pickTarget);
        }
      } else {
        // Check the last feedback for failures
        const lastFeedback = this.feedbackHistory.length > 0 ? this.feedbackHistory[this.feedbackHistory.length - 1].description : '';
        if (lastFeedback.includes('out of reach') || lastFeedback.includes('already being held')) {
          this.failedPickItems.set(pickTarget, this.actionHistory.length);
          console.log(`[Agent ${this.name}] Blocked pick(${pickTarget}). Will retry next step.`);
          action = { robotName: this.name, actionType: ActionType.WAIT, params: {}, timestamp: Date.now() };
        }
      }
    }

    // ⛔ Non-Bob robots: NEVER place on final target or wrong table — only place on Bob's table
    const FINAL_TARGETS: Record<string, string[]> = {
      make_sandwich: ['cutting_board'],
      sort_solids: ['cube_red', 'cube_green', 'cube_blue', 'cube_yellow', 'cube_purple', 'cube_orange'],
      pack_objects: ['tray'],
    };
    // Wrong tables for each layout (robots should NOT place items here — only Bob's table)
    const WRONG_TABLES = ['packing_table', 'table_dining', 'source_table_2'];
    if (this.roleTypeName !== 'Bob' && action.actionType === ActionType.PLACE) {
      const target = (action.params.target as string || '').toLowerCase();
      const validTargets = FINAL_TARGETS[this.taskType] || FINAL_TARGETS.pack_objects;
      const isFinal = validTargets.some(t => target.includes(t.toLowerCase()));
      const isWrongTable = WRONG_TABLES.some(t => target.includes(t.toLowerCase()));
      if (isFinal || isWrongTable) {
        console.log(`[Agent ${this.name}] Blocked place() on "${target}". Non-Bob robots must only place on Bob's table. Forcing place on Bob's table.`);
        action.params.target = "Bob's table";
      }
    }

    // ⛔ Physical constraint: can't pick() while already holding something
    if (this.status.gripperOccupied && action.actionType === ActionType.PICK) {
      const held = this.status.graspingObject || 'item';
      console.log(`[Agent ${this.name}] Physical constraint: holding ${held}, pick() blocked.`);
      
      if (this.roleTypeName === 'Bob') {
        // Bob: force place on cutting_board
        action = {
          robotName: this.name, actionType: ActionType.PLACE,
          params: { object: held, target: 'cutting_board' },
          timestamp: Date.now(),
        };
      } else {
        // Check if already at Bob's table (hardcoded positions for known layouts)
        const atBobTable = 
          (Math.abs(this.status.posX - 8.5) < 1.5 && Math.abs(this.status.posY - 5.5) < 1.5) || // scene1 (table_new_2)
          (Math.abs(this.status.posX - 3.5) < 1.5 && Math.abs(this.status.posY - 5.0) < 1.5) ||  // scene2/kitchen (table2 stand_pose)
          (Math.abs(this.status.posX - 2.5) < 1.5 && Math.abs(this.status.posY - 4.0) < 1.5);   // scene3/living_room (source_table_1 stand_pose)
        if (atBobTable) {
          // Already there — force place on Bob's table
          action = {
            robotName: this.name, actionType: ActionType.PLACE,
            params: { object: held, target: "Bob's table" },
            timestamp: Date.now(),
          };
        } else {
          // Not at Bob's table — navigate there first
          action = {
            robotName: this.name, actionType: ActionType.NAVIGATE,
            params: { target: 'table_new_2' },
            timestamp: Date.now(),
          };
        }
      }
    }

    // ⛔ Non-Bob: if holding an item and at Bob's table, force place (don't wander off)
    if (this.roleTypeName !== 'Bob' && this.status.gripperOccupied && action.actionType === ActionType.NAVIGATE) {
      const atBobTable = 
        (Math.abs(this.status.posX - 8.5) < 1.5 && Math.abs(this.status.posY - 5.5) < 1.5) ||
        (Math.abs(this.status.posX - 3.5) < 1.5 && Math.abs(this.status.posY - 5.0) < 1.5) ||
        (Math.abs(this.status.posX - 2.5) < 1.5 && Math.abs(this.status.posY - 4.0) < 1.5);
      if (atBobTable) {
        const held = this.status.graspingObject || 'item';
        console.log(`[Agent ${this.name}] Holding ${held} at Bob's table but trying to navigate. Forcing place.`);
        action = {
          robotName: this.name, actionType: ActionType.PLACE,
          params: { object: held, target: "Bob's table" },
          timestamp: Date.now(),
        };
      }
    }

    // ⛔ Non-Bob: pending wait after placing on Bob's table
    if (this.roleTypeName !== 'Bob' && this.pendingPlaceWait > 0) {
      console.log(`[Agent ${this.name}] Pending place wait: ${this.pendingPlaceWait} steps. Forcing wait.`);
      this.pendingPlaceWait--;
      action = { robotName: this.name, actionType: ActionType.WAIT, params: {}, timestamp: Date.now() };
    }

    // ⛔ Bob: if stuck in wait() loop and objects are on his table, force pick
    if (this.roleTypeName === 'Bob' && action.actionType === ActionType.WAIT && !this.status.gripperOccupied) {
      // Check sharedStatus for objects on Bob's table
      const bobTableRegex = /On Bob's table \(delivered\):\s*\[([^\]]+)\]/i;
      const match = (sharedStatus || '').match(bobTableRegex);
      if (match && match[1].trim()) {
        const objName = match[1].trim();
        console.log(`[Agent ${this.name}] Bob stuck in wait loop, forcing pick(${objName}) from Bob's table.`);
        action = {
          robotName: this.name, actionType: ActionType.PICK,
          params: { object: objName },
          timestamp: Date.now(),
        };
      }
    }

    if (response.thoughts) {
      this.currentPlan = response.thoughts;
    }

    return [response.thoughts, action];
  }

  async reflect(
    taskStatus: string,
    teamReflections = '',
    isLeader = false,
  ): Promise<[string, string, string]> {
    let msgs: LLMMessage[];
    if (isLeader) {
      msgs = [
        { role: 'system', content: prompts.REFLECTION_LEADER_SYSTEM },
        {
          role: 'user',
          content: prompts.reflectionLeaderUser(this.name, teamReflections, this.taskType),
        },
      ];
    } else {
      msgs = [
        { role: 'system', content: prompts.REFLECTION_PARTICIPANT_SYSTEM },
        {
          role: 'user',
          content: prompts.reflectionParticipantUser(
            this.name,
            this.roleTypeName,
            taskStatus,
            this.getTaskDescription(),
            this.getCapabilityDescription(),
            this.taskType,
          ),
        },
      ];
    }

    const response = await this.llm.chat(msgs);
    
    // Parse Summary and Plan from the response content
    const content = response.content;
    let summary = content;
    let plan = content;
    
    const summaryMatch = content.match(/Summary:\s*([\s\S]*?)(?=\n\s*Plan:|$)/i);
    const planMatch = content.match(/Plan:\s*([\s\S]*?)(?=\n\s*(CoT:|\n\s*Thoughts:|$))/i);
    if (summaryMatch) summary = summaryMatch[1].trim();
    if (planMatch) plan = planMatch[1].trim();
    
    return [response.thoughts, summary, plan];
  }

  addFeedback(feedback: Feedback): void {
    this.feedbackHistory.push(feedback);
    if (this.feedbackHistory.length > 10) {
      this.feedbackHistory = this.feedbackHistory.slice(-10);
    }
    // Detect Place Success on Bob's table → set pending wait so Bob has time to pick
    if (this.roleTypeName !== 'Bob' && feedback.success && feedback.actionType === ActionType.PLACE) {
      const desc = feedback.description.toLowerCase();
      if (desc.includes('bob') || desc.includes('table') || desc.includes('place success')) {
        this.pendingPlaceWait = 2; // wait 2 steps for Bob to pick
        console.log(`[Agent ${this.name}] Just placed on Bob's table. Will wait ${this.pendingPlaceWait} steps.`);
      }
    }
  }

  addAction(action: RobotAction): void {
    this.actionHistory.push(action);
    if (this.actionHistory.length > 10) {
      this.actionHistory = this.actionHistory.slice(-10);
    }
  }

  addMessage(message: RobotMessage): void {
    this.receivedMessages.push(message);
    if (this.receivedMessages.length > 10) {
      this.receivedMessages = this.receivedMessages.slice(-10);
    }
  }

  private extractLeader(content: string): string {
    const ROBOTS = ['Alice', 'Bob', 'David', 'Lucy'];

    // 策略1: 匹配 **Leader:** 名 或 Leader: 名（处理 markdown 粗体）
    const leaderRegex = /\*{0,2}Leader\*{0,2}\s*:\s*\*{0,2}\s*(\w+)/i;
    const leaderMatch = content.match(leaderRegex);
    if (leaderMatch) {
      const name = leaderMatch[1];
      if (ROBOTS.includes(name)) return name;
      // 检查名字是否被包含在另一个词中
      for (const r of ROBOTS) {
        if (name.toLowerCase().includes(r.toLowerCase())) return r;
      }
    }

    // 策略2: 换行格式（Leader: 单独一行，名字在下一行）
    const lines = content.split('\n');
    for (let i = 0; i < lines.length; i++) {
      const trimmed = lines[i].trim().replace(/^\d*[\.\)]?\s*/, '').replace(/\*+/g, '').toLowerCase();
      if (trimmed === 'leader' || trimmed.startsWith('leader:')) {
        if (i + 1 < lines.length) {
          const nextLine = lines[i + 1].trim();
          for (const r of ROBOTS) {
            if (nextLine.toLowerCase().includes(r.toLowerCase())) return r;
          }
        }
        // 或者当前行冒号后直接有名字
        const afterColon = lines[i].split(':')[1]?.trim() || '';
        for (const r of ROBOTS) {
          if (afterColon.toLowerCase().includes(r.toLowerCase())) return r;
        }
      }
    }

    // 策略3: 全文找最后一个出现的机器人名（前面有 Leader 关键词）
    for (const r of ROBOTS) {
      const idx = content.lastIndexOf(r);
      if (idx > 0) {
        const before = content.slice(Math.max(0, idx - 40), idx).toLowerCase();
        if (before.includes('leader')) return r;
      }
    }

    // 真正的 fallback: 用第一个机器人（不硬编码 Alice）
    console.warn(`[extractLeader] Could not extract leader from: ${content.slice(0, 100)}...`);
    return 'Alice';
  }

  private parseAction(content: string): RobotAction {
    const action: RobotAction = {
      robotName: this.name,
      actionType: ActionType.WAIT,
      params: {},
      timestamp: Date.now(),
    };

    const allText = content;
    
    // Search ALL text for action patterns, not just by line
    // Match patterns like: action_name(param1, param2)
    for (const at of Object.values(ActionType)) {
      // Create regex to find the action anywhere in text
      // Look for: actionName( or actionName ( or Contents: actionName(
      const regex = new RegExp(`(?:^|\\s|[:\"]+)${at}\\s*\\(`, 'i');
      const match = allText.match(regex);
      if (!match) continue;
      
      const startIdx = allText.indexOf(match[0]) + match[0].length - 1;
      // Find matching closing paren
      let depth = 1;
      let endIdx = startIdx + 1;
      while (depth > 0 && endIdx < allText.length) {
        if (allText[endIdx] === '(') depth++;
        else if (allText[endIdx] === ')') depth--;
        if (depth > 0) endIdx++;
      }
      
      if (depth !== 0) continue;
      
      const paramsStr = allText.substring(startIdx + 1, endIdx).trim();
      // Split by comma, handling quoted strings
      const paramsList = paramsStr.split(',').map(p => p.trim().replace(/['"\`]/g, ''));
      action.actionType = at;

      switch (at) {
        case ActionType.NAVIGATE:
          action.params = { target: paramsList[0] };
          break;
        case ActionType.OPEN:
          action.params = { container: paramsList[0] };
          break;
        case ActionType.PICK:
          action.params = { object: paramsList[0] };
          break;
        case ActionType.PLACE:
          action.params = { object: paramsList[0], target: paramsList[1] || 'tray' };
          break;
        case ActionType.MOVE:
          action.params = {
            dx: parseFloat(paramsList[0]) || 0.1,
            dy: parseFloat(paramsList[1]) || 0.1,
          };
          break;
        case ActionType.COMMUNICATE:
          action.params = { content: paramsList[0] || '', recipient: paramsList[1] || '*' };
          break;
      }
      break;
    }
    return action;
  }

  private sceneGraphToText(scene: SceneGraph): string {
    const lines: string[] = [];
    for (const obj of Object.values(scene.objects)) {
      const parts = [`${obj.name}: pos=(${obj.posX.toFixed(1)}, ${obj.posY.toFixed(1)})`];
      // Category: 'furniture' (can navigate to) or 'item' (cannot navigate to, must pick)
      parts.push(`[${obj.category}]`);
      if (obj.standPoseX != null && obj.standPoseY != null) {
        parts.push(`stand_pose=(${obj.standPoseX.toFixed(1)}, ${obj.standPoseY.toFixed(1)})`);
      }
      if (obj.isContainer) {
        parts.push(`state=${obj.isOpen ? 'open' : 'close'}`);
      }
      // Show contents for ANY object that has items on it (even non-container furniture like tables)
      if (obj.contains && obj.contains.length > 0) {
        parts.push(`contains=[${obj.contains.join(', ')}]`);
      }
      lines.push(parts.join(', '));
    }
    return lines.length > 0 ? lines.join('\n') : 'Empty scene.';
  }

  private feedbackToText(): string {
    if (this.feedbackHistory.length === 0) return '';
    return this.feedbackHistory.slice(-5).map(fb =>
      `[${fb.success ? 'SUCCESS' : 'FAILED'}] ${fb.description}`
    ).join('\n');
  }

  private actionHistoryToText(): string {
    if (this.actionHistory.length === 0) return '';
    return this.actionHistory.slice(-5).map(a => {
      const params = Object.values(a.params).join(', ');
      return `${a.actionType}(${params})`;
    }).join(', ');
  }

  private messagesToText(): string {
    if (this.receivedMessages.length === 0) return '';
    return this.receivedMessages.slice(-5).map(m =>
      `From ${m.sender}: ${m.content}`
    ).join('\n');
  }
}
