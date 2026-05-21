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
  roomWidth = 10;
  roomHeight = 8;

  constructor(layoutName = 'kitchen') {
    this.layoutName = layoutName;
    this.scene = this.buildScene(layoutName);
    // Set room dimensions based on layout
    if (layoutName === 'scene1') {
      this.roomWidth = 10;
      this.roomHeight = 8;
    }
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
      let startPositions: Array<[number, number]>;
      if (this.layoutName === 'scene1') {
        // Scene1 robot positions matching 3D BestMan scene: Alice(6,6), Bob(8.5,5.5), David(4,6), Lucy(3,2)
        const scene1Positions: Record<string, [number, number]> = {
          'Alice': [6, 6],
          'Bob': [8.5, 5.5],
          'David': [4, 6],
          'Lucy': [3, 2],
        };
        startPositions = [];
        for (const [name] of robots) {
          startPositions.push(scene1Positions[name] || [5, 5]);
        }
      } else {
        startPositions = [[4, 4], [4, 6], [6, 4], [6, 6]];
      }
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
        ['desk', 4, 8, 1.0, 0.6, false],
        ['bed', 2, 7, 1.5, 1.0, false],
        ['wardrobe', 1, 8, 0.8, 0.6, true, 'close'],
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
    } else if (layoutName === 'scene1') {
      // Scene 1: 10m x 8m room matching 3D BestMan scene1 layout
      // Internal walls: vertical at x=5 from y=0 to y=5, horizontal at y=4 from x=0 to x=3, vertical at x=5 from y=7 to y=8
      // Furniture definitions matching scene1.json
      const furniture: Array<[string, number, number, number, number, boolean, string?, string[]?]> = [
        // Kitchen area (bottom-right, x=5..10, y=0..4)
        ['fridge', 9.4, 0.5, 0.8, 0.8, true, 'close'],
        ['counter_elementA', 7.4, 0.5, 0.8, 0.6, false],
        ['counter_elementB', 5.9, 0.5, 0.8, 0.6, false],
        ['dishwasher', 8.6, 0.5, 0.6, 0.6, false],
        ['microwave', 8.1, 0.3, 0.6, 0.6, true, 'close'],
        // Dining table (center-left, x=0..5, y=0..4)
        ['table_dining', 3, 2, 1.0, 0.6, false],
        ['chair_bottom', 3, 1, 0.3, 0.3, false],
        ['chair_top', 3, 3, 0.3, 0.3, false],
        // Bookshelves (far left wall)
        ['bookshelf_1', 0.5, 0.5, 0.8, 0.6, true, 'open'],
        ['bookshelf_2', 0.5, 1.5, 0.8, 0.6, true, 'open'],
        ['bookshelf_3', 0.5, 2.5, 0.8, 0.6, true, 'open'],
        // Bob's table area (top-right, x=5..10, y=4..8)
        ['table_bob', 8.5, 5.5, 1.0, 0.6, false],
        ['table_extra', 8.5, 4, 1.0, 0.6, false],
        ['chair_bob_1', 8.5, 3, 0.3, 0.3, false],
        ['chair_bob_2', 7.5, 5, 0.3, 0.3, false],
        // Cutting board on Bob's table (8.5, 5.8)
        ['cutting_board', 8.5, 5.5, 0.4, 0.4, false],
        // Bathroom area (top-left, x=0..5, y=4..8)
        ['toilet', 1.5, 7, 0.6, 0.6, false],
        ['bathtub', 1.0, 7, 0.6, 0.6, false],
      ];
      for (const f of furniture) {
        // Walls (name starts with 'wall_') have no standPose (can't navigate to walls)
        const isWall = f[0].startsWith('wall_');
        objects[f[0]] = {
          name: f[0], category: 'furniture',
          posX: f[1], posY: f[2], width: f[3], height: f[4],
          isContainer: f[5],
          isOpen: f[6] === 'open',
          contains: f[7] || [],
          standPoseX: isWall ? null : f[1] + 0.5,
          standPoseY: isWall ? null : f[2],
        };
      }

      // Internal walls matching 3D BestMan scene1 layout
      const walls: Array<[string, number, number, number, number]> = [
        // Vertical wall: from (5,0) to (5,5), center at (5, 2.5)
        ['wall_inner_v1', 5, 2.5, 0.15, 5.0],
        // Horizontal wall: from (0,4) to (3,4), center at (1.5, 4)
        ['wall_inner_h1', 1.5, 4, 3.0, 0.15],
        // Vertical short wall: from (5,7) to (5,8), center at (5, 7.5)
        ['wall_inner_v2', 5, 7.5, 0.15, 1.0],
      ];
      for (const w of walls) {
        objects[w[0]] = {
          name: w[0], category: 'furniture',
          posX: w[1], posY: w[2], width: w[3], height: w[4],
          isContainer: false, isOpen: false, contains: [],
          standPoseX: null,
          standPoseY: null,
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

    if (this.layoutName === 'scene1') {
      // Scene 1 item placements (matching 3D BestMan scene)
      if (taskType === 'pack_objects') {
        addItem('bowl', 8.5, 5.8, 'cutting_board');
        addItem('fork', 9.4, 0.5, 'fridge');
        addItem('soap', 5.9, 0.5, 'counter_elementB');
        addItem('apple', 3, 2, 'table_dining');
      } else if (taskType === 'sort_solids') {
        addItem('red_cube', 8.5, 5.8, 'cutting_board');
        addItem('blue_sphere', 3, 2, 'table_dining');
        addItem('green_cylinder', 7.4, 0.5, 'counter_elementA');
      } else if (taskType === 'make_sandwich') {
        addItem('bread_bottom', 8.5, 5.2, 'table_bob');
        addItem('lettuce', 9.4, 0.5, 'fridge');
        addItem('tomato', 7.4, 0.5, 'counter_elementA');
        addItem('cheese', 3, 2, 'table_dining');
        addItem('ham', 8.45, 4, 'table_extra');
        addItem('bread_top', 8.55, 5.82, 'table_bob');
      }

      // Scene 1 distractors (placed at logical 3D scene locations)
      const distractors: Array<[string, number, number]> = [
        ['phone', 8.6, 0.5],
        ['book', 0.5, 0.5],
        ['toy_duck', 1.5, 7],
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
    } else {
      // Original layouts
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

      // Distractor items (original)
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
        let obj = this.scene.objects[target];
        
        // Fuzzy match
        if (!obj) {
          for (const [, o] of Object.entries(this.scene.objects)) {
            if (o.category === 'furniture' && (o.name.includes(target) || target.includes(o.name))) {
              obj = o;
              break;
            }
          }
        }
        
        if (!obj) {
          for (const o of Object.values(this.scene.objects)) {
            if (o.category === 'furniture' && o.standPoseX != null) {
              obj = o;
              break;
            }
          }
        }
        
        if (obj && obj.standPoseX != null && obj.standPoseY != null) {
          // Check if destination is in a restricted zone
          const inRestricted = this.restrictedZones.some(z => {
            const dx = obj!.standPoseX! - z.x;
            const dy = obj!.standPoseY! - z.y;
            return Math.sqrt(dx*dx + dy*dy) < z.radius;
          });
          
          if (inRestricted) {
            return {
              actionType: ActionType.NAVIGATE,
              success: false,
              description: `Navigation Failed: ${obj.name} is inside a RESTRICTED ZONE. Cannot navigate there. Choose a different target outside the restricted area.`,
              details: { error_code: 'RESTRICTED_ZONE', restricted: true },
            };
          }
          
          const oldPos = [...this.robotPositions[robotName]];
          this.robotPositions[robotName] = [obj.standPoseX, obj.standPoseY];
          const distTraveled = Math.sqrt((oldPos[0]-obj.standPoseX)**2 + (oldPos[1]-obj.standPoseY)**2);
          return {
            actionType: ActionType.NAVIGATE,
            success: true,
            description: `Navigation Success: ${robotName} completed global path planning and arrived at ${obj.name} stand_pose. Traveled ${distTraveled.toFixed(2)}m. Position: (${obj.standPoseX.toFixed(1)}, ${obj.standPoseY.toFixed(1)}).`,
            details: { target: obj.name, posX: obj.standPoseX, posY: obj.standPoseY },
          };
        }
        return {
          actionType: ActionType.NAVIGATE,
          success: false,
          description: `Navigation Failed: The target ${target} is either invalid (does not exist in scene graph) or exceeds map boundaries. Valid navigation targets: ${Object.values(this.scene.objects).filter(o=>o.standPoseX!=null).map(o=>o.name).join(', ')}.`,
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
              description: `Open Success: ${container} is already in open state. Contents inside: [${obj.contains.join(', ')}].`,
              details: { state: 'already_open', contents: obj.contains },
            };
          }
          obj.isOpen = true;
          return {
            actionType: ActionType.OPEN,
            success: true,
            description: `Open Success: ${robotName} successfully opened ${container}. Items discovered inside: [${obj.contains.join(', ')}]. This information has been added to team knowledge.`,
            details: { container, contents: obj.contains, state: 'opened' },
          };
        }
        return {
          actionType: ActionType.OPEN,
          success: false,
          description: `Open Failed: ${container} is either not a container, already in open state, or positioned beyond the robot's operational range.`,
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
            description: `Pick failed: target ${objName} does not exist in the scene graph. Valid targets are: ${Object.values(this.scene.objects).filter(o=>o.category==='item').map(o=>o.name).join(', ')}`,
            details: { error_code: 'INVALID_TARGET' },
          };
        }
        
        const dx = pos[0] - obj.posX;
        const dy = pos[1] - obj.posY;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (gripper) {
          return {
            actionType: ActionType.PICK,
            success: false,
            description: `Pick failed: gripper is already occupied with ${gripper}. The ${gripper} must be placed first before picking ${objName}.`,
            details: { error_code: 'GRIPPER_OCCUPIED', current_object: gripper },
          };
        }
        
        if (dist > 2.0) {
          return {
            actionType: ActionType.PICK,
            success: false,
            description: `Pick failed: ${objName} is out of reach (distance=${dist.toFixed(2)}m, max reach=2.0m). Try navigating closer first. Relative distance: dx=${dx.toFixed(2)}m, dy=${dy.toFixed(2)}m.`,
            details: { error_code: 'OUT_OF_REACH', distance: dist, dx, dy },
          };
        }
        
        this.robotGrippers[robotName] = objName;
        return {
          actionType: ActionType.PICK,
          success: true,
          description: `Pick Success: ${robotName} successfully picked up ${objName} at (${obj.posX.toFixed(1)}, ${obj.posY.toFixed(1)}). The object is now held in the gripper.`,
          details: { object: objName, posX: obj.posX, posY: obj.posY },
        };
      }

      case ActionType.PLACE: {
        const objName = action.params.object as string;
        const target = action.params.target as string;
        
        if (!gripper) {
          return {
            actionType: ActionType.PLACE,
            success: false,
            description: `Place Failed: Gripper is empty. Cannot place ${objName} because nothing is being held. Current gripper status: empty.`,
            details: { error_code: 'EMPTY_GRIPPER' },
          };
        }
        if (gripper !== objName) {
          return {
            actionType: ActionType.PLACE,
            success: false,
            description: `Place Failed: Object mismatch. Currently holding ${gripper}, but attempting to place ${objName}. Please place ${gripper} first.`,
            details: { error_code: 'OBJECT_MISMATCH', holding: gripper, requested: objName },
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
        
        const allPlaced = this.placedObjects.length === this.taskTargets.length &&
          [...this.placedObjects].sort().join() === [...this.taskTargets].sort().join();
        if (allPlaced) {
          this.taskCompleted = true;
          return {
            actionType: ActionType.PLACE,
            success: true,
            description: `Place Success: ${robotName} placed ${objName} at ${target}. ✅ TASK COMPLETE! All ${this.taskTargets.length}/${this.taskTargets.length} objects have been placed successfully.`,
            details: { placed: this.placedObjects, completed: true },
          };
        }
        
        return {
          actionType: ActionType.PLACE,
          success: true,
            description: `Place Success: ${robotName} placed ${objName} at ${target}. Task progress: ${this.placedObjects.length}/${this.taskTargets.length} objects placed. Remaining: ${this.taskTargets.filter(t => !this.placedObjects.includes(t)).join(', ')}.`,
          details: { placed: this.placedObjects, completed: false, remaining: this.taskTargets.filter(t => !this.placedObjects.includes(t)) },
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
      roomWidth: this.roomWidth,
      roomHeight: this.roomHeight,
      restrictedZones: this.restrictedZones,
    } as any;
  }

  enableDynamicVariation(varType: string): string {
    let msg = '';
    if (varType === 'goal_change') {
      if (this.taskTargets.length > 0) {
        const old = this.taskTargets[this.taskTargets.length - 1];
        const newTarget = 'toothpaste';
        this.taskTargets[this.taskTargets.length - 1] = newTarget;
        
        // Also add the new target object to scene if not present
        if (!this.scene.objects[newTarget]) {
          this.scene.objects[newTarget] = {
            name: newTarget, category: 'item',
            posX: 7, posY: 3, width: 0.2, height: 0.2,
            isContainer: false, isOpen: false, contains: [],
            standPoseX: null, standPoseY: null,
          };
        }
        
        msg = `🚨 DYNAMIC: Task goal CHANGED! Old goal "${old}" replaced with new goal "${newTarget}". All robots must adapt their plan to find and place the ${newTarget}! Current progress: ${this.placedObjects.length}/${this.taskTargets.length} objects.`;
        console.log(`[DYNAMIC] Goal changed: ${old} -> ${newTarget}`);
      } else {
        msg = 'Goal change requested but no targets exist.';
      }
    } else if (varType === 'restricted_zone') {
      // Mark area near cabinets as restricted
      const zoneObj = this.scene.objects['cabinet'] || this.scene.objects['bookshelf'];
      if (zoneObj) {
        const zx = zoneObj.posX;
        const zy = zoneObj.posY;
        this.restrictedZones.push({ x: zx, y: zy, radius: 1.5 });
        msg = `🚨 DYNAMIC: RESTRICTED ZONE activated at (${zx}, ${zy}) radius 1.5m! Robots cannot navigate into this area. Use caution and find alternative routes.`;
        console.log(`[DYNAMIC] Restricted zone at (${zx}, ${zy})`);
      } else {
        this.restrictedZones.push({ x: 3, y: 3, radius: 1.5 });
        msg = '🚨 DYNAMIC: RESTRICTED ZONE activated at (3, 3)! Avoid this area.';
      }
    } else if (varType === 'team_change') {
      console.log('[DYNAMIC] Team change requested');
      msg = '🔄 DYNAMIC: Team composition change detected.';
    } else if (varType === 'action_constraint') {
      console.log('[DYNAMIC] Action constraint applied');
      msg = '🔧 DYNAMIC: Action constraint applied - some actions temporarily unavailable.';
    }
    return msg;
  }
}
