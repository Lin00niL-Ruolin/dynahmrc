import {
  RobotType, DynaHMRCStage, RobotAction, RobotDialogue,
  RobotMessage, WSMessage, ActionType,
} from './types.js';
import { RobotAgent } from './agent.js';
import { SimEnvironment } from './simulation.js';
import { TASK_DESCRIPTIONS } from './prompts.js';

type UpdateCallback = (msg: WSMessage) => Promise<void>;

export class DynaHMRCEngine {
  taskType: string;
  layout: string;
  robotConfigs: Array<[string, RobotType]>;
  stage = DynaHMRCStage.SELF_DESCRIPTION;
  leader: string | null = null;
  collaborationPlan = '';
  dialogues: RobotDialogue[] = [];
  allActions: RobotAction[] = [];
  stepCount = 0;
  maxSteps = 50;
  running = false;
  paused = false;
  private onUpdate: UpdateCallback | null = null;
  dynamicVariations: string[] = [];
  dynamicStep = 3;

  agents: Record<string, RobotAgent> = {};
  sim: SimEnvironment;

  constructor(
    taskType = 'pack_objects',
    layout = 'kitchen',
    robotConfigs?: Array<[string, RobotType]>,
  ) {
    this.taskType = taskType;
    this.layout = layout;
    this.robotConfigs = robotConfigs || [
      ['Alice', RobotType.ALICE],
      ['Bob', RobotType.BOB],
      ['David', RobotType.DAVID],
      ['Lucy', RobotType.LUCY],
    ];

    for (const [name, rtype] of this.robotConfigs) {
      this.agents[name] = new RobotAgent(name, rtype, taskType);
    }

    this.sim = new SimEnvironment(layout);
  }

  setUpdateCallback(callback: UpdateCallback): void {
    this.onUpdate = callback;
  }

  private async emit(msg: WSMessage): Promise<void> {
    if (this.onUpdate) {
      try {
        await this.onUpdate(msg);
      } catch (e) {
        console.error('[Emit Error]', e);
      }
    }
  }

  private async emitDialogue(dialogue: RobotDialogue): Promise<void> {
    this.dialogues.push(dialogue);
    await this.emit({ type: 'dialogue', data: dialogue as any });
  }

  private async emitState(): Promise<void> {
    const state = this.sim.getState();
    state.stage = this.stage;
    state.leader = this.leader;
    state.collaborationPlan = this.collaborationPlan;
    state.dialogues = this.dialogues;
    state.actions = this.allActions;
    await this.emit({ type: 'state', data: state as any });
  }

  async run(): Promise<void> {
    this.running = true;
    this.sim.reset(this.taskType, this.robotConfigs);

    try {
      await this.stageSelfDescription();
      if (!this.running) return;

      await this.stageTaskAllocation();
      if (!this.running) return;

      await this.stageLeaderElection();
      if (!this.running) return;

      await this.stageExecution();
    } catch (e) {
      console.error('[ERROR] DynaHMRC run failed:', e);
      await this.emit({ type: 'error', data: { message: String(e) } });
    } finally {
      this.stage = DynaHMRCStage.COMPLETED;
      this.running = false;
      await this.emitState();
    }
  }

  private async stageSelfDescription(): Promise<void> {
    this.stage = DynaHMRCStage.SELF_DESCRIPTION;
    await this.emitState();

    const teammates = this.robotConfigs.map(([name]) => name);
    // Run all self-descriptions in parallel
    const results = await Promise.all(
      this.robotConfigs.map(async ([name, rtype]) => {
        const agent = this.agents[name];
        const [thoughts, content] = await agent.selfDescribe(teammates);
        return { stage: DynaHMRCStage.SELF_DESCRIPTION, robotName: name, robotType: rtype, thoughts, content, timestamp: Date.now() };
      })
    );
    for (const r of results) {
      await this.emitDialogue(r);
    }
    await this.sleep(200);
  }

  private async stageTaskAllocation(): Promise<void> {
    this.stage = DynaHMRCStage.TASK_ALLOCATION_BIDDING;
    await this.emitState();

    const introTexts = this.robotConfigs.map(
      ([name]) => `${name}'s self-introduction:\n${this.agents[name].selfDescription}`
    );
    const allIntroductions = introTexts.join('\n\n');

    // Run all proposals in parallel
    const results = await Promise.all(
      this.robotConfigs.map(async ([name, rtype]) => {
        const agent = this.agents[name];
        const [thoughts, content] = await agent.proposePlanAndCampaign(allIntroductions);
        return { stage: DynaHMRCStage.TASK_ALLOCATION_BIDDING, robotName: name, robotType: rtype, thoughts, content, timestamp: Date.now() };
      })
    );
    for (const r of results) {
      await this.emitDialogue(r);
    }
    await this.sleep(200);
  }

  private async stageLeaderElection(): Promise<void> {
    this.stage = DynaHMRCStage.LEADER_ELECTION;
    await this.emitState();

    const plansTexts = this.robotConfigs.map(
      ([name]) => `${name}'s plan and campaign:\n${this.agents[name].taskPlan}`
    );
    const allPlans = plansTexts.join('\n\n');

    // Run all votes in parallel
    const voteResults = await Promise.all(
      this.robotConfigs.map(async ([name, rtype]) => {
        const agent = this.agents[name];
        const [thoughts, content, vote] = await agent.voteLeader(allPlans);
        return { stage: DynaHMRCStage.LEADER_ELECTION, robotName: name, robotType: rtype, thoughts, content, vote, timestamp: Date.now() };
      })
    );

    const votes: Record<string, string> = {};
    for (const r of voteResults) {
      votes[r.robotName] = r.vote;
      await this.emitDialogue(r);
    }

    const voteCounts: Record<string, number> = {};
    for (const candidate of Object.values(votes)) {
      voteCounts[candidate] = (voteCounts[candidate] || 0) + 1;
    }

    this.leader = Object.keys(voteCounts).length > 0
      ? Object.entries(voteCounts).sort((a, b) => b[1] - a[1])[0][0]
      : this.robotConfigs[0][0];

    const leaderAgent = this.agents[this.leader];
    if (leaderAgent) {
      this.collaborationPlan = leaderAgent.taskPlan;
    }

    await this.emitState();
  }

  private async stageExecution(): Promise<void> {
    this.stage = DynaHMRCStage.EXECUTION_REFLECTION;
    const agentNames = this.robotConfigs.map(([name]) => name);

    while (this.stepCount < this.maxSteps && this.running && !this.sim.taskCompleted) {
      if (this.paused) {
        await this.sleep(500);
        continue;
      }

      this.stepCount++;

      // Dynamic variations - trigger at configured step
      for (const varType of this.dynamicVariations) {
        if (this.stepCount === this.dynamicStep) {
          const msg = this.sim.enableDynamicVariation(varType);
          await this.emitDialogue({
            stage: DynaHMRCStage.EXECUTION_REFLECTION,
            robotName: '[SYSTEM]',
            robotType: RobotType.ALICE,
            thoughts: '',
            content: msg,
            timestamp: Date.now(),
          });
        }
      }

      const taskProgress = `Step ${this.stepCount}: Placed ${this.sim.placedObjects.length}/${this.sim.taskTargets.length} objects: [${this.sim.placedObjects.join(', ')}] Remaining: [${this.sim.taskTargets.filter(t => !this.sim.placedObjects.includes(t)).join(', ')}]`;

      // Run all robot actions in parallel
      const actionResults = await Promise.all(
        agentNames.map(async (name) => {
          if (!this.running) return null;
          const agent = this.agents[name];
          const [thoughts, action] = await agent.act(
            this.leader || 'Alice',
            this.collaborationPlan,
            this.sim.scene,
            taskProgress,
            this.sim.taskTargets,
            this.sim.placedObjects,
          );
          return { name, agent, thoughts, action };
        })
      );

      // Apply actions to simulation sequentially (to avoid conflicts)
      for (const result of actionResults) {
        if (!result || !this.running) break;
        const { name, agent, thoughts, action } = result;

        const feedback = this.sim.step(action);
        agent.addFeedback(feedback);
        agent.addAction(action);
        this.allActions.push(action);

        await this.emitDialogue({
          stage: DynaHMRCStage.EXECUTION_REFLECTION,
          robotName: name,
          robotType: agent.robotType,
          thoughts,
          content: `Action: ${action.actionType}(${JSON.stringify(action.params)})\nFeedback: ${feedback.description}`,
          timestamp: Date.now(),
        });

        if (action.actionType === ActionType.COMMUNICATE) {
          const msgContent = action.params.content as string || '';
          const recipient = action.params.recipient as string || '*';
          for (const otherName of agentNames) {
            if (otherName !== name) {
              this.agents[otherName].addMessage({
                sender: name,
                receiver: recipient,
                content: msgContent,
                timestamp: Date.now(),
              });
            }
          }
        }
      }

      // Periodic reflection every 5 steps
      if (this.stepCount % 5 === 0 && this.running) {
        await this.doReflection();
      }

      await this.emitState();
    }

    // Completion
    if (this.sim.taskCompleted) {
      await this.emitDialogue({
        stage: DynaHMRCStage.COMPLETED,
        robotName: '[SYSTEM]',
        robotType: RobotType.ALICE,
        thoughts: '',
        content: `Task completed! All ${this.sim.taskTargets.length} objects placed successfully.`,
        timestamp: Date.now(),
      });
    } else {
      await this.emitDialogue({
        stage: DynaHMRCStage.COMPLETED,
        robotName: '[SYSTEM]',
        robotType: RobotType.ALICE,
        thoughts: '',
        content: `Max steps (${this.maxSteps}) reached. Task partially completed.`,
        timestamp: Date.now(),
      });
    }
  }

  private async doReflection(): Promise<void> {
    const agentNames = this.robotConfigs.map(([name]) => name);
    const taskStatus = `Placed ${this.sim.placedObjects.length}/${this.sim.taskTargets.length} objects`;

    const reflections: Record<string, { summary: string; plan: string }> = {};

    // Run all reflections in parallel
    const reflectionResults = await Promise.all(
      agentNames.map(async (name) => {
        const agent = this.agents[name];
        const [thoughts, summary, plan] = await agent.reflect(taskStatus, '', false);
        return { name, agent, thoughts, summary, plan };
      })
    );

    for (const { name, agent, thoughts, summary, plan } of reflectionResults) {
      reflections[name] = { summary, plan };
      await this.emitDialogue({
        stage: DynaHMRCStage.EXECUTION_REFLECTION,
        robotName: name,
        robotType: agent.robotType,
        thoughts,
        content: `[REFLECTION]\nSummary: ${summary}\nPlan: ${plan}`,
        timestamp: Date.now(),
      });
      await this.sleep(300);
    }

    if (this.leader && this.agents[this.leader]) {
      const leaderAgent = this.agents[this.leader];
      const teamText = Object.entries(reflections)
        .map(([n, r]) => `${n}:\nSummary: ${r.summary}\nPlan: ${r.plan}`)
        .join('\n\n');
      const [thoughts, newPlan] = await leaderAgent.reflect(taskStatus, teamText, true);
      this.collaborationPlan = newPlan;

      await this.emitDialogue({
        stage: DynaHMRCStage.EXECUTION_REFLECTION,
        robotName: `${this.leader}[LEADER]`,
        robotType: leaderAgent.robotType,
        thoughts,
        content: `[LEADER UPDATE]\nUpdated plan:\n${newPlan}`,
        timestamp: Date.now(),
      });
    }
  }

  pause(): void { this.paused = true; }
  resume(): void { this.paused = false; }
  stop(): void {
    this.running = false;
    this.stage = DynaHMRCStage.STOPPED;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
