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

  constructor(layoutName = 'scene2') {
    this.layoutName = layoutName;
    this.scene = this.buildScene(layoutName);
    // Set room dimensions based on layout
    if (layoutName === 'scene1') {
      this.roomWidth = 10;
      this.roomHeight = 8;
    } else if (layoutName === 'scene2') {
      this.roomWidth = 10;
      this.roomHeight = 10;
    } else if (layoutName === 'scene3') {
      this.roomWidth = 10;
      this.roomHeight = 10;
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
      const scenePositions: Record<string, Record<string, [number, number]>> = {
        'scene1': {
          'Alice': [6, 6],
          'Bob': [8.5, 5.5],
          'David': [4, 6],
          'Lucy': [3, 2],
        },
        'scene2': {
          'Alice': [2, 2],
          'Bob': [3, 5],
          'David': [7, 2],
          'Lucy': [8, 5],
        },
        'scene3': {
          'Alice': [5, 2],
          'Bob': [2, 4],
          'David': [8, 6],
          'Lucy': [3, 3],
        },
      };
      const robotPosMap = scenePositions[this.layoutName] || {};
      if (Object.keys(robotPosMap).length > 0) {
        startPositions = [];
        for (const [name] of robots) {
          startPositions.push(robotPosMap[name] || [5, 5]);
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

    if (layoutName === 'scene1') {
      // Scene 1: 10m x 8m room matching 3D BestMan scene1 layout (Make Sandwich)
      // Internal walls: vertical at x=5 from y=0 to y=5, horizontal at y=4 from x=0 to x=3, vertical at x=5 from y=7 to y=8
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
        // Bookshelves (far left wall, y=0..3)
        ['bookshelf_1', 0.5, 0.5, 0.8, 0.6, true, 'open'],
        ['bookshelf_2', 0.5, 1.5, 0.8, 0.6, true, 'open'],
        ['bookshelf_3', 0.5, 2.5, 0.8, 0.6, true, 'open'],
        // Bob's table area (top-right, x=5..10, y=4..8)
        ['table_bob', 8.5, 5.5, 1.0, 0.6, false],
        ['table_extra', 8.5, 4, 1.0, 0.6, false],
        ['chair_bob_1', 8.5, 3, 0.3, 0.3, false],
        ['chair_bob_2', 7.5, 5, 0.3, 0.3, false],
        ['cutting_board', 8.5, 5.5, 0.4, 0.4, false],
        ['tray', 8.8, 5.5, 0.3, 0.3, false],
        // Colored sorting panels for sort_solids task
        ['red_panel', 1.5, 2.5, 0.4, 0.3, false],
        ['blue_panel', 2.5, 2.5, 0.4, 0.3, false],
        ['green_panel', 3.5, 2.5, 0.4, 0.3, false],
        // Bathroom area (top-left, x=0..5, y=4..8)
        ['toilet', 1.5, 7, 0.6, 0.6, false],
        ['bathtub', 1.0, 7, 0.6, 0.6, false],
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

      // Internal walls matching 3D BestMan scene1 layout
      const walls: Array<[string, number, number, number, number]> = [
        ['wall_inner_v1', 5, 2.5, 0.15, 5.0],   // vertical: (5,0) to (5,5)
        ['wall_inner_h1', 1.5, 4, 3.0, 0.15],    // horizontal: (0,4) to (3,4)
        ['wall_inner_v2', 5, 7.5, 0.15, 1.0],    // vertical: (5,7) to (5,8)
      ];
      for (const w of walls) {
        objects[w[0]] = {
          name: w[0], category: 'furniture',
          posX: w[1], posY: w[2], width: w[3], height: w[4],
          isContainer: false, isOpen: false, contains: [],
          standPoseX: null, standPoseY: null,
        };
      }

    } else if (layoutName === 'kitchen') {
      // Scene 2: 10m x 10m L-shaped room matching 3D BestMan scene2 layout (Sort Solids)
      // Furniture from scene2.json
      const furniture: Array<[string, number, number, number, number, boolean, string?, string[]?]> = [
        // Kitchen counters/cabinets along bottom wall (y=0..2)
        ['cabinet_elementB', 1, 0.55, 0.8, 0.6, true, 'close'],
        ['counter_elementA', 2.5, 1.48, 0.8, 0.6, false],
        ['microwave', 3.2, 1.1, 0.6, 0.4, true, 'close'],
        ['dishwasher', 3.7, 0.35, 0.6, 0.4, false],
        ['fridge', 4.5, 1.06, 0.8, 0.8, true, 'close'],
        // Tables
        ['table_1', 1, 4, 1.2, 0.8, false],
        ['table_2', 3, 5, 1.2, 0.8, false],
        // Left area furniture
        ['bookcase', 1, 6.5, 0.8, 0.6, true, 'open'],
        // Top area (L alcove)
        ['sofa', 7.5, 9, 1.5, 0.8, false],
        ['bed', 8, 9, 1.5, 1.0, false],
        // Right wall shelf
        ['shelf_table', 9.5, 7.5, 1.0, 3.0, false],
        ['blackboard', 7, 0.15, 0.15, 2.0, false],
                ['large_red_cube', 3, 5.3, 0.35, 0.35, false],
        ['large_green_cube', 3, 5, 0.35, 0.35, false],
        ['large_blue_cube', 3, 4.7, 0.35, 0.35, false],
        ['large_yellow_cube', 2.5, 5.3, 0.35, 0.35, false],
        ['large_purple_cube', 2.5, 5, 0.35, 0.35, false],
        ['large_orange_cube', 2.5, 4.7, 0.35, 0.35, false],
        ['tray', 3, 4.7, 0.3, 0.3, false], // Colored sorting panels for sort_solids task
        ['red_panel', 1.5, 8, 0.4, 0.3, false],
        ['blue_panel', 2.5, 8, 0.4, 0.3, false],
        ['green_panel', 3.5, 8, 0.4, 0.3, false],
        // Floor rug
        ['rug', 6, 2, 0.6, 0.6, false],
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

      // Internal walls matching 3D BestMan scene2 layout
      const walls: Array<[string, number, number, number, number]> = [
        ['wall_inner_v1', 5, 2, 0.15, 4.0],       // vertical: (5,0) to (5,4)
        ['wall_inner_h1', 3, 8, 6.0, 0.15],        // horizontal: (0,8) to (6,8)
        ['wall_inner_v2', 6, 9, 0.15, 2.0],        // vertical: (6,8) to (6,10)
      ];
      for (const w of walls) {
        objects[w[0]] = {
          name: w[0], category: 'furniture',
          posX: w[1], posY: w[2], width: w[3], height: w[4],
          isContainer: false, isOpen: false, contains: [],
          standPoseX: null, standPoseY: null,
        };
      }

    } else if (layoutName === 'living_room') {
      // Scene 3: 10m x 10m room matching 3D BestMan scene3 layout (Pack Objects)
      // Furniture from scene3.json
      const furniture: Array<[string, number, number, number, number, boolean, string?, string[]?]> = [
        // Kitchen line along bottom (y=0..2)
        ['kitchen_cabinet', 1.2, 0.55, 0.8, 0.6, true, 'close'],
        ['kitchen_counter', 3.1, 1.6, 0.8, 0.6, false],
        ['microwave', 3.8, 1.1, 0.6, 0.4, true, 'close'],
        ['dishwasher', 4.6, 0.43, 0.6, 0.4, false],
        ['fridge', 5.5, 1.06, 0.8, 0.8, true, 'close'],
        ['cabinet_2', 7.3, 0.55, 0.8, 0.6, true, 'close'],
        ['sofa', 8.6, 0.42, 1.5, 0.8, false],
        // Tables and chairs (center area)
        ['packing_table', 8, 3, 1.2, 0.8, false],
        ['source_table_1', 2, 4, 1.0, 0.8, false],
        ['source_table_2', 4, 4, 1.0, 0.8, false],
        ['chair', 2, 3, 0.3, 0.3, false],
        ['bookcase', 7.5, 5.5, 0.8, 0.6, true, 'open'],
        // Bathroom area (top area)
        ['bathtub', 1.5, 9.4, 1.0, 0.6, false],
        ['sink_base', 4.7, 9.5, 0.8, 0.6, false],
        ['tray', 8, 2.7, 0.3, 0.3, false],
        // Colored sorting panels for sort_solids task
        ['red_panel', 5, 2, 0.4, 0.3, false],
        ['blue_panel', 6, 2, 0.4, 0.3, false],
        ['green_panel', 7, 2, 0.4, 0.3, false],
        // Floor items
        ['rug', 8, 2.4, 0.6, 0.6, false],
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

      // Internal walls matching 3D BestMan scene3 layout
      const walls: Array<[string, number, number, number, number]> = [
        ['wall_inner_v1', 6, 1, 0.15, 2.0],        // vertical: (6,0) to (6,2)
        ['wall_inner_v2', 6, 6, 0.15, 4.0],        // vertical: (6,4) to (6,8)
        ['wall_inner_h1', 8, 6, 4.0, 0.15],         // horizontal: (6,6) to (10,6)
        ['wall_inner_v3', 5.5, 9, 0.15, 2.0],       // vertical: (5.5,8) to (5.5,10)
        ['wall_inner_h2', 1.5, 8, 3.0, 0.15],       // horizontal: (0,8) to (3,8)
        ['wall_inner_h3', 5.75, 8, 0.5, 0.15],      // horizontal: (5.5,8) to (6,8)
      ];
      for (const w of walls) {
        objects[w[0]] = {
          name: w[0], category: 'furniture',
          posX: w[1], posY: w[2], width: w[3], height: w[4],
          isContainer: false, isOpen: false, contains: [],
          standPoseX: null, standPoseY: null,
        };
      }
    } else {
      // Fallback default scene
      const furniture: Array<[string, number, number, number, number, boolean, string?]> = [
        ['table_0', 4, 4, 1.5, 1.0, false],
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
      // Scene 1 item placements (matching 3D BestMan scene1)
      if (taskType === 'pack_objects') {
        addItem('fork', 9.4, 0.5, 'fridge');
        addItem('apple', 3, 2, 'table_dining');
        addItem('book', 0.5, 0.5, 'bookshelf_1');
        addItem('soap', 5.9, 0.5, 'counter_elementB');
      } else if (taskType === 'sort_solids') {
        addItem('small_red_cube', 3, 2, 'table_dining');
      } else if (taskType === 'make_sandwich') {
        addItem('ham_bottom', 8.2, 5.5, 'table_bob');
        addItem('bacon', 8.5, 4, 'table_extra');
        addItem('ham_top', 8.55, 4.2, 'table_extra');
      }

      // Scene 1 distractors + 3D scene books
      const distractors: Array<[string, number, number, string?]> = [
        ['phone', 8.7, 0.5],
        ['toy_duck', 1.5, 7],
        // Books on bookshelf_1 matching scene1.json (book_holder, book_1, book_2, book_3)
        ['book_holder', 0.3, 0.5, 'bookshelf_1'],
        ['book_1', 0.4, 0.7, 'bookshelf_1'],
        ['book_2', 0.6, 0.7, 'bookshelf_1'],
        ['book_3', 0.8, 0.5, 'bookshelf_1'],
      ];
      for (const [name, x, y, container] of distractors) {
        if (!this.scene.objects[name]) {
          this.scene.objects[name] = {
            name, category: 'item',
            posX: x, posY: y, width: 0.2, height: 0.2,
            isContainer: false, isOpen: false, contains: [],
            standPoseX: null, standPoseY: null,
          };
          if (container && this.scene.objects[container]) {
            this.scene.objects[container].contains.push(name);
          }
        }
      }

    } else if (this.layoutName === 'kitchen') {
      // Scene 2 (sort_solids) item placements matching 3D scene2.json
      if (taskType === 'pack_objects') {
        addItem('bowl', 2.5, 1.48, 'counter_elementA');
        addItem('fork', 4.5, 1.06, 'fridge');
        addItem('soap', 1, 4, 'table_1');
        addItem('apple', 3, 5, 'table_2');
      } else if (taskType === 'sort_solids') {
        // Small red cube scattered in scene, large cubes already on table_2
        addItem('small_red_cube', 9.5, 7.5, 'shelf_table');
      } else if (taskType === 'make_sandwich') {
        addItem('bread_bottom', 2.5, 1.48, 'counter_elementA');
        addItem('ham', 1, 4, 'table_1');
        addItem('bread_top', 3, 5, 'table_2');
      }

      // Scene 2 distractors
      const distractors: Array<[string, number, number]> = [
        ['book', 1, 6.5],
        ['toy_duck', 7.5, 9],
        ['phone', 9.5, 7.5],
        ['lemon', 5, 1.5],
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

    } else if (this.layoutName === 'living_room') {
      // Scene 3 (pack_objects) item placements matching 3D scene3.json
      if (taskType === 'pack_objects') {
        // Items match scene3.json positions
        addItem('fork', 1.2, 0.55, 'kitchen_cabinet');      // fork_0
        addItem('apple', 4.15, 4, 'source_table_2');          // apple on source_table_2
        addItem('book', 7.5, 5.8, 'bookcase');                // book_0 on bookcase
        addItem('soap', 5.7, 2, 'kitchen_counter');            // soap on wall shelf near wall4
      } else if (taskType === 'sort_solids') {
        addItem('small_red_cube', 2, 4, 'source_table_1');
      } else if (taskType === 'make_sandwich') {
        addItem('bread_bottom', 8, 3, 'packing_table');
        addItem('ham', 3.1, 1.6, 'kitchen_counter');
        addItem('bread_top', 8, 2.7, 'packing_table');
      }

      // Scene 3 distractors matching scene3.json items
      const distractors: Array<[string, number, number]> = [
        ['lemon', 5.5, 1.0],       // near fridge (from scene3.json)
        ['cup', 4.15, 4.2],        // on source_table_2 (from scene3.json)
        ['book_0', 7.5, 5.8],      // on bookcase (from scene3.json)
        ['toy_duck', 1.5, 9.4],    // near bathtub
        ['phone', 8.6, 0.42],      // near sofa
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
      // Fallback default layout
      if (taskType === 'pack_objects') {
        addItem('bowl', 3.5, 5.5, 'table_0');
        addItem('fork', 4, 5.5, 'table_0');
        addItem('soap', 2, 7, 'cabinet');
        addItem('apple', 3, 5, 'table_0');
      } else if (taskType === 'sort_solids') {
        addItem('red_cube', 3.5, 5.5, 'table_0');
        addItem('blue_sphere', 4.5, 5.5, 'table_0');
        addItem('green_cylinder', 2, 7, 'cabinet');
      } else if (taskType === 'make_sandwich') {
        addItem('bread_bottom', 3.5, 5.5, 'table_0');
        addItem('lettuce', 5.5, 5.5, 'table_0');
        addItem('tomato', 2, 7, 'cabinet');
        addItem('cheese', 4, 5.5, 'table_0');
        addItem('ham', 4.5, 5.5, 'table_0');
        addItem('bread_top', 3, 5.5, 'table_0');
      }

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

  isManipulator(name: string): boolean {
    // David is pure mobile (no manipulation), Bob is fixed arm (no navigation)
    const rtype = this.robotTypes[name];
    return rtype !== RobotType.DAVID;
  }

  canNavigate(name: string): boolean {
    const rtype = this.robotTypes[name];
    return rtype !== RobotType.BOB;
  }

  step(action: RobotAction): Feedback {
    this.stepCount++;
    const robotName = action.robotName;
    const pos = this.robotPositions[robotName] || [5, 5];
    const gripper = this.robotGrippers[robotName] || null;

    // === Capability enforcement ===
    if (action.actionType === ActionType.NAVIGATE && !this.canNavigate(robotName)) {
      return {
        actionType: ActionType.NAVIGATE,
        success: false,
        description: `${robotName} is a fixed arm and cannot navigate or move from position (${pos[0].toFixed(2)}, ${pos[1].toFixed(2)}). Only mobile robots can navigate.`,
        details: {},
      };
    }
    if ((action.actionType === ActionType.PICK || action.actionType === ActionType.PLACE)
      && !this.isManipulator(robotName)) {
      return {
        actionType: action.actionType,
        success: false,
        description: `${robotName} is a navigation-only robot without a manipulator arm. Cannot pick or place objects. Only communicate or navigate.`,
        details: {},
      };
    }
    if (action.actionType === ActionType.OPEN && !this.isManipulator(robotName)) {
      return {
        actionType: ActionType.OPEN,
        success: false,
        description: `${robotName} is a navigation-only robot and cannot open containers. This needs a robot with a manipulator arm.`,
        details: {},
      };
    }

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
      taskType: this.taskType,
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
      // Mark area near cabinets as restricted (use layout-appropriate target)
      const zoneObj = this.scene.objects['cabinet']
        || this.scene.objects['bookshelf_1']
        || this.scene.objects['cabinet_elementB']
        || this.scene.objects['kitchen_cabinet'];
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
