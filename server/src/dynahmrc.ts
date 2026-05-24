import {
  RobotType, DynaHMRCStage, RobotAction, RobotDialogue,
  RobotMessage, WSMessage, ActionType,
} from './types.js';
import { RobotAgent } from './agent.js';
import { SimEnvironment } from './simulation.js';
import { TASK_DESCRIPTIONS, TASK_GOALS } from './prompts.js';
import { sendAction as bestmanSendAction } from './bestman-bridge.js';

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

  useBestMan = false;
  private bestManStarted = false;
  agents: Record<string, RobotAgent> = {};
  sim: SimEnvironment;

  taskDescription = '';

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
    state.paused = this.paused;
    await this.emit({ type: 'state', data: state as any });
  }

  async run(): Promise<void> {
    this.running = true;
    this.sim.reset(this.taskType, this.robotConfigs);

    // Randomize sort_solids: pick 1-2 random color pairs
    if (this.taskType === 'sort_solids') {
      const colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange'];
      const count = Math.random() < 0.6 ? 1 : 2;
      const shuffled = [...colors].sort(() => Math.random() - 0.5).slice(0, count);
      const pairs = shuffled.map(c => ({ small: `small_cube_${c}`, large: `cube_${c}` }));
      const pairTexts = pairs.map(p => `${p.small} → ${p.large}`);
      this.taskDescription = `Match ${count} pair(s): ${pairTexts.join(', ')}. Mobile robots find the small cubes and bring them to Bob's table. Bob places each on its matching large cube.`;
      this.sim.taskTargets = pairs.map(p => p.small);
      const allPrompts = await import('./prompts.js');
      allPrompts.TASK_DESCRIPTIONS['sort_solids'] = this.taskDescription;
      allPrompts.TASK_GOALS['sort_solids'] = [...this.sim.taskTargets];
      // Update static hints to match randomized colors
      const first = pairs[0];
      const pairList = pairs.map(p => `${p.small} → ${p.large}`).join(', ');
      allPrompts.TASK_ALLOCATION_HINTS['sort_solids'] = `Match: ${pairList}. Mobile robots search and transport, Bob does precision placement.`;
      allPrompts.REFLECTION_TASK_HINTS['sort_solids'] = `Task: Place ${pairList}. Mobile robots find and bring to Bob's table, Bob places on matching large cube.`;
      allPrompts.LEADER_REFLECTION_HINTS['sort_solids'] = `Match ${pairList}. Bob can reach his table. Mobile robots fetch small cubes and bring to Bob.`;
      console.log(`[DynaHMRC] Sort task: ${pairList}`);
    }

    // 如果启用 BestMan，启动服务
    if (this.useBestMan && !this.bestManStarted) {
      try {
        const { startService } = await import('./bestman-bridge.js');
        const ok = await startService(this.layout);
        this.bestManStarted = ok;
        if (ok) {
          console.log('[DynaHMRC] BestMan 3D simulation started ✅');
          await this.emitDialogue({
            stage: DynaHMRCStage.SELF_DESCRIPTION,
            robotName: '[SYSTEM]',
            robotType: RobotType.ALICE,
            thoughts: '',
            content: '🎮 BestMan 3D 仿真已启动，PyBullet 窗口已打开',
            timestamp: Date.now(),
          });
        }
      } catch (e) {
        console.warn('[DynaHMRC] BestMan start failed:', e);
      }
    }

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

      // 如果启用了 BestMan，清理资源
      if (this.useBestMan && this.bestManStarted) {
        try {
          const { stopService } = await import('./bestman-bridge.js');
          stopService();
          this.bestManStarted = false;
        } catch { /* ignore */ }
      }

      await this.emitState();
    }
  }

  private async stageSelfDescription(): Promise<void> {
    this.stage = DynaHMRCStage.SELF_DESCRIPTION;
    await this.emitState();

    const teammates = this.robotConfigs.map(([name]) => name);
    // Run all self-descriptions in parallel, but emit each one as soon as it arrives
    await Promise.all(
      this.robotConfigs.map(async ([name, rtype]) => {
        const agent = this.agents[name];
        const [thoughts, content] = await agent.selfDescribe(teammates);
        await this.emitDialogue({ stage: DynaHMRCStage.SELF_DESCRIPTION, robotName: name, robotType: rtype, thoughts, content, timestamp: Date.now() });
        await this.sleep(600); // delay so each appears one by one
      })
    );
  }

  private async stageTaskAllocation(): Promise<void> {
    this.stage = DynaHMRCStage.TASK_ALLOCATION_BIDDING;
    await this.emitState();

    const introTexts = this.robotConfigs.map(
      ([name]) => `${name}'s self-introduction:\n${this.agents[name].selfDescription}`
    );
    const allIntroductions = introTexts.join('\n\n');

    // Run all proposals in parallel, emit each one as it arrives
    await Promise.all(
      this.robotConfigs.map(async ([name, rtype]) => {
        const agent = this.agents[name];
        const [thoughts, content] = await agent.proposePlanAndCampaign(allIntroductions);
        await this.emitDialogue({ stage: DynaHMRCStage.TASK_ALLOCATION_BIDDING, robotName: name, robotType: rtype, thoughts, content, timestamp: Date.now() });
        await this.sleep(600);
      })
    );
  }

  private async stageLeaderElection(): Promise<void> {
    this.stage = DynaHMRCStage.LEADER_ELECTION;
    await this.emitState();

    const plansTexts = this.robotConfigs.map(
      ([name]) => `${name}'s plan and campaign:\n${this.agents[name].taskPlan}`
    );
    const allPlans = plansTexts.join('\n\n');

    // Run all votes in parallel, emit each one as it arrives
    const votes: Record<string, string> = {};
    await Promise.all(
      this.robotConfigs.map(async ([name, rtype]) => {
        const agent = this.agents[name];
        const [thoughts, content, vote] = await agent.voteLeader(allPlans);
        votes[name] = vote;
        await this.emitDialogue({ stage: DynaHMRCStage.LEADER_ELECTION, robotName: name, robotType: rtype, thoughts, content, vote, timestamp: Date.now() });
        await this.sleep(600);
      })
    );

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

      // Build shared status (task blackboard — everyone sees this)
      const gripperStatuses = agentNames.map(n => {
        const a = this.agents[n];
        const holding = a.status.graspingObject || 'empty';
        return `${n}:${holding === 'empty' ? 'empty' : `holding ${holding}`}`;
      }).join(', ');
      const placedStr = this.sim.placedObjects.length > 0
        ? `Placed on cutting_board: [${this.sim.placedObjects.join(', ')}]`
        : 'Nothing placed yet';
      const remainingStr = this.sim.taskTargets.filter(t => !this.sim.placedObjects.includes(t));
      const unclaimedStr = remainingStr.length > 0
        ? `Unclaimed items: [${remainingStr.join(', ')}]`
        : 'All items placed!';
      // On Bob's table: items physically near Bob's position (delivered but not final-placed)
      const bobPos = this.sim.robotPositions['Bob'] || null;
      const onBobTable = this.sim.taskTargets.filter(t => {
        if (this.sim.placedObjects.includes(t)) return false;
        if (Object.values(this.sim.robotGrippers).includes(t)) return false;
        const obj = this.sim.scene.objects[t];
        if (!obj || !bobPos) return false;
        const dx = Math.abs(obj.posX - bobPos[0]);
        const dy = Math.abs(obj.posY - bobPos[1]);
        return dx < 0.8 && dy < 0.8;
      });
      const bobTableStr = onBobTable.length > 0
        ? `On Bob's table (delivered): [${onBobTable.join(', ')}]`
        : 'Nothing on Bob\'s table yet';

      // Real-time item locations with coordinates
      const realtimeLocations: string[] = [];
      for (const t of this.sim.taskTargets) {
        const obj = this.sim.scene.objects[t];
        if (!obj) continue;
        const pos = `(${obj.posX.toFixed(1)},${obj.posY.toFixed(1)})`;
        let status = '';
        if (this.sim.placedObjects.includes(t)) {
          status = '✅ placed on final target';
        } else if (Object.values(this.sim.robotGrippers).includes(t)) {
          const holder = Object.entries(this.sim.robotGrippers).find(([,v]) => v === t)?.[0] || '?';
          status = `carried by ${holder}`;
        } else {
          // Check if item is near any robot's position (delivered to their table)
          const originals: Record<string, Record<string, string>> = {
            make_sandwich: { bread_0: 'table_new_2', bacon: 'table_new_1', bread_1: 'table_new_1' },
            sort_solids: { small_cube_red: 'scattered' },
            pack_objects: { fork_0: 'kitchen_cabinet', apple: 'source_table_2', book_0: 'bookcase', soap: 'wall_shelf' },
          };
          let atRobotTable = '';
          for (const [rName, rPos] of Object.entries(this.sim.robotPositions)) {
            if (Math.abs(obj.posX - rPos[0]) < 0.5 && Math.abs(obj.posY - rPos[1]) < 0.5) {
              atRobotTable = `${rName}'s table`;
              break;
            }
          }
          status = atRobotTable
            ? `delivered to ${atRobotTable}`
            : (originals[this.taskType]?.[t] || 'original position');
        }
        realtimeLocations.push(`${t}@${pos}:${status}`);
      }
      const locationStr = realtimeLocations.join(' | ');

      // Robot positions
      const robotPosStrs = agentNames.map(n => {
        const p = this.sim.robotPositions[n];
        return p ? `${n}@(${p[0].toFixed(1)},${p[1].toFixed(1)})` : `${n}:unknown`;
      }).join(', ');

      const sharedStatus = `Robots: [${robotPosStrs}]\nGrippers: ${gripperStatuses}\n${placedStr}\n${bobTableStr}\n${unclaimedStr}\nLocations: ${locationStr}`;

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
            sharedStatus,
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

        // Sync agent state from simulation (positions + gripper)
        const simPos = this.sim.robotPositions[name];
        if (simPos) {
          agent.status.posX = simPos[0];
          agent.status.posY = simPos[1];
        }
        const gripObj = this.sim.robotGrippers[name];
        agent.status.gripperOccupied = gripObj != null;
        agent.status.graspingObject = gripObj || null;

        // 如果启用 BestMan，将动作转发到 3D 仿真
        if (this.useBestMan && this.bestManStarted) {
          try {
            const actParams: Record<string, any> = { ...action.params };
            const actTypeMap: Record<string, string> = {
              'navigate': 'navigate',
              'open': 'pick',
              'pick': 'pick',
              'place': 'place',
              'move': 'navigate',
              'communicate': 'communicate',
              'wait': 'wait',
            };
            const bmAction = actTypeMap[action.actionType] || action.actionType;
            const result = await bestmanSendAction(name, bmAction, actParams);
            if (!result.success) {
              console.warn(`[BestMan] Action failed: ${name} ${bmAction} - ${result.message}`);
            }
          } catch (e: any) {
            console.warn(`[BestMan] sendAction error: ${e.message}`);
          }
        }

        await this.emitDialogue({
          stage: DynaHMRCStage.EXECUTION_REFLECTION,
          robotName: name,
          robotType: agent.robotType,
          thoughts,
          content: `Action: ${action.actionType}(${Object.values(action.params).join(", ")})\nFeedback: ${feedback.description}`,
          timestamp: Date.now(),
        });
        await this.sleep(400);

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

      // Periodic reflection every 10 steps (better action-to-reflection ratio)
      if (this.stepCount % 10 === 0 && this.running) {
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

    // Build detailed task status with placed items, remaining items, and gripper states
    const placedStr = this.sim.placedObjects.length > 0 ? this.sim.placedObjects.join(', ') : 'none';
    const remainingStr = this.sim.taskTargets.filter(t => !this.sim.placedObjects.includes(t)).join(', ') || 'none';
    const gripStrs = agentNames.map(n => {
      const g = this.sim.robotGrippers[n];
      return `${n}:${g ? `holding ${g}` : 'empty'}`;
    }).join(', ');
    const isDone = this.sim.taskCompleted;

    // Build real-time object locations
    const bobPos = this.sim.robotPositions['Bob'] || null;
    const objLocStrs: string[] = [];
    for (const t of this.sim.taskTargets) {
      const obj = this.sim.scene.objects[t];
      if (!obj) continue;
      let where = 'unknown';
      if (this.sim.placedObjects.includes(t)) {
        where = '✅ ON FINAL TARGET';
      } else if (Object.values(this.sim.robotGrippers).includes(t)) {
        const holder = Object.entries(this.sim.robotGrippers).find(([,v]) => v === t)?.[0];
        where = `carried by ${holder}`;
      } else if (bobPos && Math.abs(obj.posX - bobPos[0]) < 0.8 && Math.abs(obj.posY - bobPos[1]) < 0.8) {
        where = `at Bob\'s table (${obj.posX.toFixed(1)},${obj.posY.toFixed(1)}) — Bob can pick!`;
      } else {
        where = `at (${obj.posX.toFixed(1)},${obj.posY.toFixed(1)}) — needs transport`;
      }
      objLocStrs.push(`${t}: ${where}`);
    }

    const taskStatus = `${isDone ? '✅ TASK COMPLETE - ' : ''}Placed ${this.sim.placedObjects.length}/${this.sim.taskTargets.length} objects on final target: [${placedStr}]
Remaining: [${remainingStr}]
Grippers: ${gripStrs}
Object Locations:
${objLocStrs.join('\n')}
${isDone ? 'NOTE: All objects are already placed. No further action needed.' : ''}`;

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
