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
    );

    const msgs: LLMMessage[] = [
      { role: 'system', content: sysContent },
      { role: 'user', content: userContent },
    ];

    const response = await this.llm.chat(msgs);
    const action = this.parseAction(response.content);

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
    return [response.thoughts, response.content, response.content];
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
    for (const line of content.split('\n')) {
      const trimmed = line.trim().toLowerCase();
      if (trimmed.startsWith('leader')) {
        const namePart = line.includes(':') ? line.split(':')[1].trim() : '';
        for (const name of ['Alice', 'Bob', 'David', 'Lucy']) {
          if (namePart.toLowerCase().includes(name.toLowerCase())) {
            return name;
          }
        }
      }
    }
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
