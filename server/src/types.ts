export enum RobotType {
  ALICE = 'Alice',
  BOB = 'Bob',
  DAVID = 'David',
  LUCY = 'Lucy',
}

export enum ActionType {
  NAVIGATE = 'navigate',
  OPEN = 'open',
  PICK = 'pick',
  PLACE = 'place',
  MOVE = 'move',
  COMMUNICATE = 'communicate',
  WAIT = 'wait',
}

export enum TaskType {
  PACK_OBJECTS = 'pack_objects',
  SORT_SOLIDS = 'sort_solids',
  MAKE_SANDWICH = 'make_sandwich',
}

export enum DynaHMRCStage {
  SELF_DESCRIPTION = 'self_description',
  TASK_ALLOCATION_BIDDING = 'task_allocation_bidding',
  LEADER_ELECTION = 'leader_election',
  EXECUTION_REFLECTION = 'execution_reflection',
  COMPLETED = 'completed',
}

export interface LLMMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface LLMResponse {
  thoughts: string;
  content: string;
  raw: string;
}

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
  actionType: ActionType;
  params: Record<string, any>;
  timestamp: number;
}

export interface RobotMessage {
  sender: string;
  receiver: string;
  content: string;
  timestamp: number;
}

export interface Feedback {
  actionType: ActionType;
  success: boolean;
  description: string;
  details: Record<string, any>;
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
  collaborationPlan: string;
  dialogues: RobotDialogue[];
  actions: RobotAction[];
  taskProgress: string;
  taskCompleted: boolean;
  roomWidth?: number;
  roomHeight?: number;
  restrictedZones?: Array<{ x: number; y: number; radius: number }>;
}

export interface WSMessage {
  type: string;
  data: Record<string, any>;
}

export const ROBOT_TYPE_NAMES: Record<string, string> = {
  'Alice': 'Alice',
  'Bob': 'Bob',
  'David': 'David',
  'Lucy': 'Lucy',
};
