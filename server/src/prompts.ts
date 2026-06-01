export const ROBOT_CAPABILITIES: Record<string, string> = {
  Alice: `
Role: You are Alice, a mobile manipulation robot with a wheeled chassis and a single manipulator arm.
Capabilities:
- Navigate on the ground to any reachable location
- Open hinged objects (drawers, cabinets, fridge, etc.)
- Pick up objects within your reachable range
- Place objects on platforms or containers
- Adjust position by moving short distances (delta_x, delta_y)
- Communicate with other robots
- Wait when needed
`,
  Bob: `
Role: You are Bob, a Manipulation Robot (single robotic arm fixed on a desktop).
Capabilities:
- Pick up objects within your limited reachable range
- Place objects precisely on platforms or containers
- Communicate with other robots
- Wait when needed
Limitations:
- Cannot navigate or move from your fixed position
- Cannot open hinged objects
- Can only manipulate objects within arm's reach
`,
  David: `
Role: You are David, a Mobile Robot (wheeled chassis for ground navigation).
Capabilities:
- Navigate efficiently on the ground to any location
- Communicate with other robots
- Wait when needed
Limitations:
- Cannot manipulate any objects
- Cannot open hinged objects
- Cannot pick or place items
`,
  Lucy: `
Role: You are Lucy, a Drone Robot (quadrotor with a fixed suction gripper).
Capabilities:
- Navigate through the air, including elevated and hard-to-reach areas
- Pick up objects from above using suction gripper
- Place objects on platforms
- Communicate with other robots
- Wait when needed
Limitations:
- Cannot open hinged objects
- Limited payload capacity
`,
};

// Per-robot-type atomic action sets (论文 Table I)
// NOTE: wait() is intentionally excluded — robots must always act
const BASE_ACTION_SETS: Record<string, string> = {
  Alice: `
1. navigate(<stand_pose_id>) - Move to a target stand pose near a furniture item
2. open(<container_name>) - Open a hinged container (drawer, cabinet, etc.)
3. pick(<object_name>) - Pick up a specified object
4. place(<object_name>, <target_location>) - Place held object at a target
5. move(<delta_x>, <delta_y>) - Adjust position by small offsets
6. communicate(<content>, <recipient>) - Send a message to another robot
`,
  Bob: `
1. pick(<object_name>) - Pick up a specified object
2. place(<object_name>, <target_location>) - Place held object at a target
3. communicate(<content>, <recipient>) - Send a message to another robot
`,
  David: `
1. navigate(<stand_pose_id>) - Move to a target stand pose near a furniture item
2. communicate(<content>, <recipient>) - Send a message to another robot
`,
  Lucy: `
1. navigate(<stand_pose_id>) - Move to a target stand pose near a furniture item
2. pick(<object_name>) - Pick up a specified object
3. place(<object_name>, <target_location>) - Place held object at a target
4. communicate(<content>, <recipient>) - Send a message to another robot
`,
};

export const ROBOT_ACTION_SETS: Record<string, string> = BASE_ACTION_SETS;

// Stage 1: Self-Description
export const SELF_DESCRIPTION_SYSTEM = `
You are an intelligent robot on a heterogeneous multi-robot team.
Phase: Initial stage — each robot introduces itself.

CRITICAL: You have SPECIFIC capabilities below. DO NOT describe yourself as a generic robot.
`;

export function selfDescriptionUser(taskDescription: string, teammates: string, capabilities: string, taskType?: string): string {
  void taskDescription; void taskType;
  return `
Your teammates: ${teammates}

===== YOUR ROBOT IDENTITY (READ CAREFULLY - THIS IS WHO YOU ARE) =====
${capabilities}

===== INSTRUCTION =====
Write ONE short paragraph introducing YOURSELF to your teammates.

🚨 DO NOT say "I am a versatile mobile robot" or anything generic.
🚨 You are a SPECIFIC robot with specific capabilities.
🚨 Mention your type, what you can do, what you CANNOT do, and your best role in a team.

Keep it natural, like greeting teammates for the first time. Do not reference any task.
`;
}

// Stage 2: Task Allocation
export const TASK_ALLOCATION_SYSTEM = `
You are a SPECIFIC robot (Alice/Bob/David/Lucy) with unique capabilities.
Each robot has different strengths — some move, some fly, some manipulate, some explore.

Phase: Division of labor and leadership campaign.
Tasks:
1) Propose a division of labor that leverages EACH robot's unique strengths.
2) Write a campaign speech to run for leader, highlighting YOUR unique advantages.

CoT: Let's think step by step!
`;

export const TASK_ALLOCATION_HINTS: Record<string, string> = {
  make_sandwich: 'Stack bread_0, bacon, bread_1 on cutting_board. bread_0 on table_new_2 (Bob\'s table, Bob can reach). bacon and bread_1 on table_new_1 (need transport).',
  sort_solids: 'This is a SORTING mission. Find the small_cube_red scattered in the scene and place it on the cube_red on table_2. Mobile robots search and transport, Bob does precision placement.',
  pack_objects: 'This is a PACKING mission. Items: fork_0, apple, book_0, soap. Each robot picks ONE different item - do NOT pick what another robot already has. Mobile robots find items and bring to Bob\'s table. Bob places them into the tray one by one.',
};

export function taskAllocationUser(name: string, selfIntroductions: string, taskType?: string): string {
  return `
🚨 You are ${name}. NOT a generic robot.

Below are the self-introductions from yourself and your collaborators:
${selfIntroductions}

Task Context:
${TASK_ALLOCATION_HINTS[taskType || 'pack_objects'] || TASK_ALLOCATION_HINTS.pack_objects}

Analyze these introductions carefully. Each robot has DIFFERENT capabilities. Assign tasks that play to each robot's strengths (e.g., Lucy flies, Bob places precisely, Alice navigates+manipulates, David scouts).

Write:
1) Your collaboration plan assigning specific tasks to each robot
2) Your campaign speech for becoming the leader, explaining why YOU are the best fit

Output:
Thoughts: [step-by-step reasoning]
Contents: [collaboration plan + campaign speech]
`;
}

// Stage 3: Leader Election
export const LEADER_ELECTION_SYSTEM = `
You are a SPECIFIC robot. Review all plans and campaign speeches below.
Elect the most qualified leader based on demonstrated planning ability and team fit.
Self-nomination is allowed.
`;

export function leaderElectionUser(name: string, plansAndSpeeches: string): string {
  return `
You are ${name}. Below are ALL robots' plans and campaign speeches:
${plansAndSpeeches}

Elect ONE leader. Explain your choice. Then give the leader's name.

Format:
Thoughts: [analysis]
Reasons: [why this leader]
Leader: [name]
`;
}

// Stage 4: Execution
export function executionSystem(
  roleDescription: string,
  taskDescription: string,
  teammates: string,
  leader: string,
  plan: string,
  principles: string,
  taskType?: string,
  robotName?: string,
  gripperOccupied?: boolean,
): string {
  // Task-specific item and placement info
  const taskSteps: Record<string, { items: string; locations: string; target: string }> = {
    make_sandwich: {
      items: 'bread_0, bacon, bread_1',
      locations: 'bread_0 on table_new_2 (8.2, 5.85), bacon on table_new_1 (8.5, 4), bread_1 on table_new_1 (8.55, 4.2)',
      target: 'cutting_board',
    },
    sort_solids: {
      items: 'small_cube_red',
      locations: 'small_cube_red scattered somewhere in the scene (check shelf_table, sofa, bookcase, floor areas). 6 large colored cubes (red/green/blue/yellow/purple/orange) are on table_2.',
      target: 'cube_red on table_2',
    },
    pack_objects: {
      items: 'fork_0, apple, book_0, soap',
      locations: 'fork_0 on kitchen_cabinet (1.2, 0.55), apple on source_table_2 (4.15, 4), book_0 on bookcase (7.5, 5.5), soap on wall_shelf (5.7, 6)',
      target: 'tray',
    },
  };

  // For sort_solids, generate dynamic tasks from randomized TASK_GOALS
  let info = taskSteps[taskType || 'pack_objects'] || taskSteps.pack_objects;
  if (taskType === 'sort_solids') {
    const goals = TASK_GOALS['sort_solids'] || [];
    if (goals.length > 0) {
      const itemList = goals.join(', ');
      const locList = goals.map(g => {
        const color = g.replace('small_cube_', '');
        // Positions for kitchen (scene2) layout where sort_solids runs
        const locs: Record<string, string> = {
          red: 'shelf_table (9.5,7.5)', green: 'bookcase (1,6.5)', blue: 'sofa (7.5,9.5)',
          yellow: 'table_1 (8.5,2.8)', purple: 'shelf_table (9.5,7.5)', orange: 'table_2 (5,6)',
        };
        return `${g} near ${locs[color] || 'unknown'}`;
      }).join(', ');
      info = { items: itemList, locations: locList, target: 'on matching colored cube' };
    }
  }

  const robotNameClean = robotName || 'Alice';

  // 动态过滤可用动作：如果夹爪满了，移除 pick 选项
  let actionSet = ROBOT_ACTION_SETS[robotNameClean] || ROBOT_ACTION_SETS.Alice;
  if (gripperOccupied) {
    // 移除包含 pick() 的行及其编号
    actionSet = actionSet.replace(/^\d+\.\s*pick\(.*\n?/gm, '');
  }

  const isMobile = robotNameClean !== 'Bob';
  const example = isMobile ? 'navigate(target) or pick(item) or place(item, target)' : 'pick(item) or place(item, target) or communicate(msg, recipient)';

  const workflows: Record<string, string> = {
    make_sandwich: `Mobile robots: go to table_new_1 → pick bacon/bread_1 → bring to table_new_2 → place on table_new_2
Bob: pick items from table_new_2 → place on cutting_board`,
    sort_solids: `Mobile robots: find the small cube → pick() → bring to Bob's table (table_2) → place on table_2
Bob: pick small cube from table_2 → place on matching large cube`,
    pack_objects: `Mobile robots: find an item → pick() → bring to Bob's table → place on Bob's table
Bob: pick items from table → place into tray`,
  };
  const workflow = workflows[taskType || 'pack_objects'] || workflows.pack_objects;

  const placedBy = isMobile ? `place(${info.items.includes(',') ? 'your_object' : 'item'}, Bob's table)` : `place(item, ${info.target})`;

  return `
===== IDENTITY (CRITICAL - READ THIS) =====
🚨 You are ${robotNameClean}. NOT a generic robot. NOT any other robot.

${roleDescription}

===== TASK =====
${taskDescription}
Leader: ${leader}
Plan: ${plan}

Items to handle: ${info.items}
Item locations: ${info.locations}
${taskType === 'make_sandwich' ? 'Final: stack on cutting_board' : taskType === 'sort_solids' ? 'Final: place on matching colored cube (Bob only)' : 'Final: place into tray (Bob only)'}

===== YOUR ALLOWED ACTIONS =====
${actionSet}

===== WORKFLOW =====
${workflow}

===== ⚠️ STRICT RULES =====
🚫 NEVER use wait() - it is not available. Always DO something.
🚫 pick() ONLY when gripper EMPTY. If FULL, navigate or place.
${isMobile ? '🚫 NEVER place on final target. Place on Bob\'s table instead.' : '🚫 You CANNOT navigate. Others bring items to your table.'}
✅ Read your feedback! If FAILED, try a DIFFERENT approach.

Output ONLY one action in this format:
Thoughts: [1 sentence reasoning]
Contents: [one action, e.g. ${example}]
`;
}

export function executionUser(
  sceneGraph: string,
  posX: number,
  posY: number,
  gripperStatus: string,
  graspingObject: string,
  feedbackHistory: string,
  actionHistory: string,
  receivedMessages: string,
  taskProgress: string,
  taskTargets?: string[],
  placedObjects?: string[],
  sharedStatus?: string,
): string {
  const missing = taskTargets ? taskTargets.filter(t => !(placedObjects || []).includes(t)) : [];
  
  // 根据状态提供针对性引导
  const statusGuidance = graspingObject !== 'nothing'
    ? `⛔ HOLDING ${graspingObject}. You MUST navigate to Bob's table and place() it.`
    : missing.length > 0
    ? `✅ Empty gripper. You should navigate to find a missing item (${missing[0]}) and pick() it.`
    : '✅ All items placed! Report task complete.';

  return `
=== CURRENT STATE ===
Task: ${taskProgress}
Missing: ${missing.length > 0 ? missing.join(', ') : 'ALL PLACED ✅'}

=== 🚨 STATUS-BASED GUIDANCE ===
${statusGuidance}

=== FEEDBACK ===
${feedbackHistory || 'No feedback yet.'}

=== SCENE ===
${sceneGraph}

=== STATUS ===
Position: (${posX.toFixed(2)}, ${posY.toFixed(2)})
Gripper: ${gripperStatus}

Recent Actions: ${actionHistory || 'none'}
Messages: ${receivedMessages || 'none'}

=== OUTPUT ===
Thoughts: [1 sentence reasoning]
Contents: [action(params)]
`;
}

// Robot-specific Principles
export const PRINCIPLES: Record<string, string> = {
  Alice: `
1) Efficiently explore and navigate all locations in the scene graph without repetition.
2) Transport task-related items promptly.
3) When facing inaccessible areas, notify capable assistants.
4) Track task progress and adjust targets timely.
5) Respond promptly to collaborators' requests.
6) If grasp fails, try other stand poses or adjust base position.
7) Focus on completing the task without unrelated actions.
`,
  Bob: `
1) Analyze tasks and scene graphs, prioritizing your work.
2) Request help promptly for distant or missing objects.
3) Notify collaborators of task progress timely.
4) Track progress changes and adjust targets as needed.
5) Respond promptly to collaborators' requests.
6) Focus on task completion without unrelated actions.
`,
  David: `
1) Efficiently explore and navigate all locations in the scene graph without repetition.
2) Notify collaborators of task items and request mobile teammates for transport.
3) Notify capable assistants to explore inaccessible areas.
4) Request collaborators to open objects for exploration.
5) Track task progress and adjust targets timely.
6) Respond promptly to assistants' messages.
7) Focus on completing the task without unrelated actions.
`,
  Lucy: `
1) Efficiently explore and navigate all locations in the scene graph without repetition.
2) Transport task-related items promptly.
3) Request collaborators to open objects for exploration.
4) Track task progress and adjust targets timely.
5) Respond promptly to collaborators' requests.
6) Focus on task completion without unrelated actions.
`,
};

// Reflection
export const REFLECTION_PARTICIPANT_SYSTEM = `
Contexts: Please summarize and analyze the historical cooperation experiences and present your future collaboration plans.

Phase: Now it is the group discussion session of the heterogeneous robot collaboration phase.

Principles:
1) Compare the differences between the current task status and the target task status.
2) Analyze the current scene graph content, historical feedback, action and message sequences, and summarize the successful experiences and failure lessons.
CoT: Let's think step by step!
`;

export const REFLECTION_TASK_HINTS: Record<string, string> = {
  make_sandwich: 'Task: Stack bread_0, bacon, bread_1 on cutting_board. bread_0 on table_new_2 (Bob can reach), bacon and bread_1 on table_new_1 (need transport).',
  sort_solids: 'Task: Find small_cube_red and place on cube_red. Search floor/shelf areas for the small cube, then bring to table_2.',
  pack_objects: 'Task: Pack fork_0, apple, book_0, soap into tray. Others bring items to Bob on the table, Bob places into tray.',
};

export function reflectionParticipantUser(
  name: string,
  role: string,
  taskStatus: string,
  taskDescription: string,
  capabilities: string,
  taskType?: string,
): string {
  return `
You are ${name}, a ${role}.

Current task status: ${taskStatus}
Target task: ${taskDescription}

${REFLECTION_TASK_HINTS[taskType || 'pack_objects'] || REFLECTION_TASK_HINTS.pack_objects}

${capabilities}

Summarize your experiences and propose future plans.

Output Response Format:
Thoughts: think step by step to analyze the problem.
Summary: summary, analysis of past, current task status.
Plan: plan for your subsequent tasks.
CoT: Let's think step by step!
`;
}

export const LEADER_REFLECTION_HINTS: Record<string, string> = {
  make_sandwich: 'Stack bread_0, bacon, bread_1 on cutting_board. Bob can reach bread_0 (table_new_2). Mobile robots fetch bacon & bread_1 (table_new_1) and bring them to Bob.',
  sort_solids: 'The SORTING mission needs small_cube_red on cube_red. Assign scouts to find the small cube, and Bob for precision placement.',
  pack_objects: 'The PACKING mission needs fork_0, apple, book_0, soap in tray. Assign each item to a different robot, Bob places from table into tray.',
};

export const REFLECTION_LEADER_SYSTEM = `
Contexts:
1) You are an intelligent robot capable of human-like reasoning, collaborating with others on complex tasks.
2) As the leader, summarize and analyze collaborators' experiences and tasks, then propose the final task plan.

Phase: It is the leadership summary stage of group discussion.

Principles:
1) Assign specific tasks to each one, including yourself.
2) Ensure plan reflects current environment and object states.
CoT: Let's think step by step!
`;

export function reflectionLeaderUser(name: string, teamReflections: string, taskType?: string): string {
  return `
1) You are a smart robot named ${name}, you are the leader.
2) The historical summaries and future plans of each one in the entire team received are as follows:
${teamReflections}

Task Context:
${LEADER_REFLECTION_HINTS[taskType || 'pack_objects'] || LEADER_REFLECTION_HINTS.pack_objects}

Output Response Format:
Thoughts: think step by step to analyze the problem.
Contents: output the latest heterogeneous robots plan.
CoT: Let's think step by step!
`;
}

export const TASK_DESCRIPTIONS: Record<string, string> = {
  make_sandwich: 'Stack bread_0, bacon, bread_1 on top of each other on the cutting board (any order).',
  sort_solids: 'Find the small red cube scattered in the scene, bring it to table_2, and place it on top of the matching large red cube.',
  pack_objects: 'Pack four items (fork_0, apple, book_0, soap) into the tray.',
};

export const TASK_GOALS: Record<string, string[]> = {
  make_sandwich: ['bread_0', 'bacon', 'bread_1'],
  sort_solids: ['small_cube_red'],
  pack_objects: ['fork_0', 'apple', 'book_0', 'soap'],
};
