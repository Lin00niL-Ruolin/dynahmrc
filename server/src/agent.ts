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
        // Check if already at Bob's table (table_new_2 @ 8.5, 5.5)
        const atBobTable = Math.abs(this.status.posX - 8.5) < 1.5 && Math.abs(this.status.posY - 5.5) < 1.5;
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
      if (obj.isContainer) {
        parts.push(`state=${obj.isOpen ? 'open' : 'close'}`);
        if (obj.standPoseX != null && obj.standPoseY != null) {
          parts.push(`stand_pose=(${obj.standPoseX.toFixed(1)}, ${obj.standPoseY.toFixed(1)})`);
        }
        if (obj.contains.length > 0) {
          parts.push(`contains=[${obj.contains.join(', ')}]`);
        }
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
