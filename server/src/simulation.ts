import {
  RobotType, ActionType, SceneGraph, SceneObject,
  RobotStatus, RobotAction, Feedback, SimulationState,
  DynaHMRCStage,
} from './types.js';
import { TASK_GOALS } from './prompts.js';

export class SimEnvironment {
  scene: SceneGraph;
  robotPositions: Record<string, [number, number]> = {};
  robotGrippers: Record<string, string | null> = {};
  robotTypes: Record<string, RobotType> = {};
  taskType = '';
  taskTargets: string[] = [];
  placedObjects: string[] = [];
  restrictedZones: Array<{ x: number; y: number; radius: number }> = [];
  stepCount = 0;
  maxSteps = 50;
  taskCompleted = false;

  private layoutName: string;

  constructor(layoutName = 'kitchen') {
    this.layoutName = layoutName;
    this.scene = this.buildScene(layoutName);
  }

  reset(
    taskType = 'pack_objects',
    robots?: Array<[string, RobotType]>,
  ): SimulationState {
    this.taskType = taskType;
    this.scene = this.buildScene(this.layoutName);
    this.robotPositions = {};
    this.robotGrippers = {};
    this.robotTypes = {};
    this.placedObjects = [];
    this.stepCount = 0;
    this.taskCompleted = false;

    if (robots) {
      const startPositions: Array<[number, number]> = [[4, 4], [4, 6], [6, 4], [6, 6]];
      for (let i = 0; i < robots.length; i++) {
        const [name, rtype] = robots[i];
        const pos = startPositions[i % startPositions.length];
        this.robotPositions[name] = pos;
        this.robotGrippers[name] = null;
        this.robotTypes[name] = rtype;
      }
    }

    this.taskTargets = TASK_GOALS[taskType] || [];
    this.addTaskObjects(taskType);

    return this.getState();
  }

  private buildScene(layoutName: string): SceneGraph {
    const objects: Record<string, SceneObject> = {};

    if (layoutName === 'kitchen') {
      const furniture: Array<[string, number, number, number, number, boolean, string?, string[]?]> = [
        ['table_0', 3, 5, 1.5, 1.0, false],
        ['table_1', 7, 5, 1.5, 1.0, false],
        ['fridge', 1, 3, 0.8, 0.8, true, 'close'],
        ['cabinet', 5, 8, 0.8, 0.8, true, 'close'],
        ['drawer', 2, 7, 0.6, 0.6, true, 'close'],
        ['shelf', 8, 2, 1.0, 0.6, true, 'open'],
        ['counter', 5, 1, 2.0, 0.6, false],
        ['tray', 5, 5, 0.4, 0.4, false],
      ];
      for (const f of furniture) {
        objects[f[0]] = {
          name: f[0], category: 'furniture',
          posX: f[1], posY: f[2], width: f[3], height: f[4],
          isContainer: f[5],
          isOpen: f[6] === 'open',
          contains: f[7] || [],
          standPoseX: f[1] + 0.5,
          standPoseY: f[2],
        };
      }
    } else if (layoutName === 'living_room') {
      const furniture: Array<[string, number, number, number, number, boolean, string?, string[]?]> = [
        ['table_0', 4, 4, 1.5, 1.0, false],
        ['sofa', 2, 2, 2.0, 0.8, false],
        ['bookshelf', 8, 3, 1.0, 0.6, true, 'close'],
        ['tv_stand', 8, 7, 1.2, 0.6, false],
        ['coffee_table', 5, 5, 1.0, 0.8, false],
        ['tray', 5, 5, 0.4, 0.4, false],
      ];
      for (const f of furniture) {
        objects[f[0]] = {
          name: f[0], category: 'furniture',
          posX: f[1], posY: f[2], width: f[3], height: f[4],
          isContainer: f[5],
          isOpen: f[6] === 'open',
          contains: f[7] || [],
          standPoseX: f[1] + 0.5,
          standPoseY: f[2],
        };
      }
    } else {
      const furniture: Array<[string, number, number, number, number, boolean, string?]> = [
        ['table_0', 4, 4, 1.5, 1.0, false],
        ['table_1', 6, 6, 1.5, 1.0, false],
        ['cabinet', 2, 7, 0.8, 0.8, true, 'close'],
        ['counter', 5, 2, 2.0, 0.6, false],
        ['tray', 5, 5, 0.4, 0.4, false],
      ];
      for (const f of furniture) {
        objects[f[0]] = {
          name: f[0], category: 'furniture',
          posX: f[1], posY: f[2], width: f[3], height: f[4],
          isContainer: f[5],
          isOpen: f[6] === 'open',
          contains: [],
          standPoseX: f[1] + 0.5,
          standPoseY: f[2],
        };
      }
    }

    return { objects };
  }

  private addTaskObjects(taskType: string): void {
    const addItem = (name: string, x: number, y: number, container: string) => {
      this.scene.objects[name] = {
        name, category: 'item',
        posX: x, posY: y, width: 0.2, height: 0.2,
        isContainer: false, isOpen: false, contains: [],
        standPoseX: null, standPoseY: null,
      };
      if (this.scene.objects[container]) {
        this.scene.objects[container].contains.push(name);
      }
    };

    if (taskType === 'pack_objects') {
      addItem('bowl', 3.5, 5.5, 'table_0');
      addItem('fork', 6.5, 5.5, 'table_1');
      addItem('soap', 5, 8, 'cabinet');
      addItem('apple', 1, 3, 'fridge');
    } else if (taskType === 'sort_solids') {
      addItem('red_cube', 3.5, 5.5, 'table_0');
      addItem('blue_sphere', 6.5, 5.5, 'table_1');
      addItem('green_cylinder', 5, 8, 'cabinet');
    } else if (taskType === 'make_sandwich') {
      addItem('bread_bottom', 3.5, 5.5, 'table_0');
      addItem('lettuce', 1, 3, 'fridge');
      addItem('tomato', 5, 8, 'cabinet');
      addItem('cheese', 6.5, 5.5, 'table_1');
      addItem('ham', 3.5, 4.5, 'table_0');
      addItem('bread_top', 6.5, 4.5, 'table_1');
    }

    // Distractor items
    const distractors: Array<[string, number, number]> = [
      ['phone', 2, 2],
      ['book', 7, 3],
      ['toy_duck', 8, 8],
    ];
    for (const [name, x, y] of distractors) {
      if (!this.scene.objects[name]) {
        this.scene.objects[name] = {
          name, category: 'item',
          posX: x, posY: y, width: 0.2, height: 0.2,
          isContainer: false, isOpen: false, contains: [],
          standPoseX: null, standPoseY: null,
        };
      }
    }
  }

  step(action: RobotAction): Feedback {
    this.stepCount++;
    const robotName = action.robotName;
    const pos = this.robotPositions[robotName] || [5, 5];
    const gripper = this.robotGrippers[robotName] || null;

    switch (action.actionType) {
      case ActionType.WAIT:
        return {
          actionType: ActionType.WAIT,
          success: true,
          description: `${robotName} is waiting.`,
          details: {},
        };

      case ActionType.NAVIGATE: {
        const target = action.params.target as string;
        const obj = this.scene.objects[target];
        if (obj && obj.standPoseX != null && obj.standPoseY != null) {
          this.robotPositions[robotName] = [obj.standPoseX, obj.standPoseY];
          return {
            actionType: ActionType.NAVIGATE,
            success: true,
            description: `${robotName} navigated to ${target} at (${obj.standPoseX.toFixed(1)}, ${obj.standPoseY.toFixed(1)}).`,
            details: { target, posX: obj.standPoseX, posY: obj.standPoseY },
          };
        }
        return {
          actionType: ActionType.NAVIGATE,
          success: false,
          description: `Navigation failed: target ${target} not found.`,
          details: {},
        };
      }

      case ActionType.OPEN: {
        const container = action.params.container as string;
        const obj = this.scene.objects[container];
        if (obj && obj.isContainer) {
          if (obj.isOpen) {
            return {
              actionType: ActionType.OPEN,
              success: true,
              description: `${container} is already open. Contains: ${obj.contains.join(', ')}`,
              details: {},
            };
          }
          obj.isOpen = true;
          return {
            actionType: ActionType.OPEN,
            success: true,
            description: `${robotName} opened ${container}. Found: ${obj.contains.join(', ')}`,
            details: { container, contents: obj.contains },
          };
        }
        return {
          actionType: ActionType.OPEN,
          success: false,
          description: `Failed to open ${container}: not a container.`,
          details: {},
        };
      }

      case ActionType.PICK: {
        const objName = action.params.object as string;
        const obj = this.scene.objects[objName];
        if (!obj) {
          return {
            actionType: ActionType.PICK,
            success: false,
            description: `Pick failed: ${objName} not found.`,
            details: {},
          };
        }
        const dx = pos[0] - obj.posX;
        const dy = pos[1] - obj.posY;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (gripper) {
          return {
            actionType: ActionType.PICK,
            success: false,
            description: `Pick failed: already holding ${gripper}.`,
            details: {},
          };
        }
        if (dist > 2.0) {
          return {
            actionType: ActionType.PICK,
            success: false,
            description: `Pick failed: ${objName} too far (dist=${dist.toFixed(2)}).`,
            details: { distance: dist },
          };
        }
        this.robotGrippers[robotName] = objName;
        return {
          actionType: ActionType.PICK,
          success: true,
          description: `${robotName} picked up ${objName}.`,
          details: {},
        };
      }

      case ActionType.PLACE: {
        const objName = action.params.object as string;
        const target = action.params.target as string;
        if (gripper !== objName) {
          return {
            actionType: ActionType.PLACE,
            success: false,
            description: `Place failed: not holding ${objName}. Holding: ${gripper}`,
            details: {},
          };
        }
        this.robotGrippers[robotName] = null;
        if (this.scene.objects[objName]) {
          this.scene.objects[objName].posX = pos[0] + 0.3;
          this.scene.objects[objName].posY = pos[1];
        }
        if (this.taskTargets.includes(objName) && !this.placedObjects.includes(objName)) {
          this.placedObjects.push(objName);
        }
        if (this.placedObjects.length === this.taskTargets.length &&
            [...this.placedObjects].sort().join() === [...this.taskTargets].sort().join()) {
          this.taskCompleted = true;
        }
        return {
          actionType: ActionType.PLACE,
          success: true,
          description: `${robotName} placed ${objName} at ${target}. Progress: ${this.placedObjects.length}/${this.taskTargets.length}`,
          details: { placed: this.placedObjects, completed: this.taskCompleted },
        };
      }

      case ActionType.MOVE: {
        const dx = (action.params.dx as number) || 0.1;
        const dy = (action.params.dy as number) || 0.1;
        this.robotPositions[robotName] = [pos[0] + dx, pos[1] + dy];
        return {
          actionType: ActionType.MOVE,
          success: true,
          description: `${robotName} moved to (${(pos[0] + dx).toFixed(2)}, ${(pos[1] + dy).toFixed(2)}).`,
          details: {},
        };
      }

      case ActionType.COMMUNICATE: {
        const content = action.params.content as string || '';
        const recipient = action.params.recipient as string || '*';
        return {
          actionType: ActionType.COMMUNICATE,
          success: true,
          description: `${robotName} -> ${recipient}: ${content}`,
          details: {},
        };
      }

      default:
        return {
          actionType: action.actionType,
          success: false,
          description: `Unknown action: ${action.actionType}`,
          details: {},
        };
    }
  }

  getState(): SimulationState {
    const robots: Record<string, RobotStatus> = {};
    for (const name of Object.keys(this.robotPositions)) {
      const rtype = this.robotTypes[name] || RobotType.ALICE;
      const pos = this.robotPositions[name] || [0, 0];
      const gripObj = this.robotGrippers[name] || null;
      robots[name] = {
        name,
        robotType: rtype,
        posX: pos[0],
        posY: pos[1],
        gripperOccupied: gripObj !== null,
        graspingObject: gripObj,
        reachableRange: 0.5,
      };
    }

    return {
      step: this.stepCount,
      robots,
      scene: this.scene,
      stage: DynaHMRCStage.EXECUTION_REFLECTION,
      leader: null,
      collaborationPlan: '',
      dialogues: [],
      actions: [],
      taskProgress: `Placed ${this.placedObjects.length}/${this.taskTargets.length} objects`,
      taskCompleted: this.taskCompleted,
    };
  }

  enableDynamicVariation(varType: string): void {
    if (varType === 'goal_change') {
      if (this.taskTargets.length > 0) {
        const old = this.taskTargets[this.taskTargets.length - 1];
        this.taskTargets[this.taskTargets.length - 1] = 'phone';
        console.log(`[DYNAMIC] Task goal changed: ${old} -> phone`);
      }
    } else if (varType === 'restricted_zone') {
      this.restrictedZones.push({ x: 3, y: 3, radius: 1.5 });
      console.log('[DYNAMIC] Restricted zone activated at (3, 3)');
    } else if (varType === 'team_change') {
      console.log('[DYNAMIC] Team change requested');
    } else if (varType === 'action_constraint') {
      console.log('[DYNAMIC] Action constraint applied');
    }
  }
}
