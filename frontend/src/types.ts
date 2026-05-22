export type RobotType = 'Alice' | 'Bob' | 'David' | 'Lucy';

export type DynaHMRCStage =
  | 'self_description'
  | 'task_allocation_bidding'
  | 'leader_election'
  | 'execution_reflection'
  | 'completed'
  | 'stopped';

export interface RobotStatus {
  name: string;
  robotType: RobotType;
  posX: number;
  posY: number;
  gripperOccupied: boolean;
  graspingObject: string | null;
  reachableRange: number;
}

export interface SceneObject {
  name: string;
  category: 'furniture' | 'item';
  posX: number;
  posY: number;
  width: number;
  height: number;
  isContainer: boolean;
  isOpen: boolean;
  contains: string[];
  standPoseX: number | null;
  standPoseY: number | null;
}

export interface SceneGraph {
  objects: Record<string, SceneObject>;
}

export interface RobotAction {
  robotName: string;
  actionType: string;
  params: Record<string, any>;
  timestamp: number;
}

export interface RobotDialogue {
  stage: DynaHMRCStage;
  robotName: string;
  robotType: RobotType;
  thoughts: string;
  content: string;
  timestamp: number;
}

export interface SimulationState {
  step: number;
  robots: Record<string, RobotStatus>;
  scene: SceneGraph;
  stage: DynaHMRCStage;
  leader: string | null;
  taskType?: string;
  collaborationPlan: string;
  dialogues: RobotDialogue[];
  actions: RobotAction[];
  taskProgress: string;
  taskCompleted: boolean;
  roomWidth?: number;
  roomHeight?: number;
  restrictedZones?: Array<{ x: number; y: number; radius: number }>;
}

export interface ConfigOption {
  id: string;
  name: string;
  desc: string;
}

export interface AppConfig {
  taskTypes: ConfigOption[];
  layouts: ConfigOption[];
  robotTypes: ConfigOption[];
  dynamicVariations: ConfigOption[];
  defaultApiKeyConfigured: boolean;
}
