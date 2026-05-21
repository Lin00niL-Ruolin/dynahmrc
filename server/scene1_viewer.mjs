import { SimEnvironment } from './src/simulation.js';

const sim = new SimEnvironment('scene1');
sim.reset('pack_objects');
const state = sim.getState();

const W = 10, H = 8;
const scale = 4; // chars per meter: 40x32 grid

const ch = H * scale, cw = W * scale;
const grid = Array.from({length: ch + 2}, () => Array(cw + 2).fill('.'));

// Border
for (let x = 0; x <= cw + 1; x++) { grid[0][x] = '#'; grid[ch+1][x] = '#'; }
for (let y = 0; y <= ch + 1; y++) { grid[y][0] = '#'; grid[y][cw+1] = '#'; }

// Zone labels (as text blocks)
function write(x, y, txt) {
  for (let i = 0; i < txt.length; i++) {
    if (x+i > 0 && x+i <= cw && y > 0 && y <= ch) grid[y][x+i] = txt[i];
  }
}

// Draw internal walls (thicker with #)
for (const [name, w] of Object.entries(state.scene.objects)) {
  if (!name.startsWith('wall_')) continue;
  if (w.width > w.height) {
    const cy = Math.round(w.posY * scale) + 1;
    const half = Math.round(w.width * scale / 2);
    for (let dx = -half; dx <= half; dx++) {
      const cx = Math.round(w.posX * scale) + 1 + dx;
      if (cx > 0 && cx <= cw) { grid[cy][cx] = '#'; grid[cy-1][cx] = '#'; grid[cy+1][cx] = '#'; }
    }
  } else {
    const cx = Math.round(w.posX * scale) + 1;
    const half = Math.round(w.height * scale / 2);
    for (let dy = -half; dy <= half; dy++) {
      const cy = Math.round(w.posY * scale) + 1 + dy;
      if (cy > 0 && cy <= ch) { grid[cy][cx] = '#'; grid[cy][cx-1] = '#'; grid[cy][cx+1] = '#'; }
    }
  }
}

// Write zone names
write(4, 2, 'DINING');
write(4, 16, 'BATHROOM');
write(21, 2, 'KITCHEN');
write(21, 16, 'BOB-LAB');

// Write furniture as text labels
write(11, 5, 'TBL');     // table_dining (3,2)
write(8, 9, 'CH');       // chair_bottom (3,1)
write(8, 17, 'CH');      // chair_top (3,3)
write(3, 5, 'BK1');      // bookshelf_1 (0.5,0.5)
write(3, 9, 'BK2');      // bookshelf_2 (0.5,1.5)
write(3, 13, 'BK3');     // bookshelf_3 (0.5,2.5)

write(23, 6, 'FRIDGE');  // fridge (9.7,0.5)
write(22, 4, 'MICRO');   // microwave (8.4,0.3)
write(20, 6, 'CTR-A');   // counter_elementA (7.7,0.5)
write(22, 8, 'DISH');    // dishwasher (8.9,0.5)
write(24, 4, 'FZ');      // frozen/freezer area

write(24, 12, 'CTR-B');  // counter_elementB (6.2,0.5)

write(22, 22, 'WTBL');   // table_bob (8.5,5.5)
write(23, 18, 'ETBL');   // table_extra (8.5,4)
write(24, 14, 'CH');     // chair_bob_1 (8.5,3)
write(24, 24, 'CH');     // chair_bob_2 (7.5,5)
write(22, 24, 'CUT');    // cutting_board (8.5,5.8)

write(6, 24, 'WC');      // toilet (1.5,7)
write(8, 28, 'BATH');    // bathtub (1.0,7)

// Robot spawn points
write(23, 23, '[B]');    // Bob (8.5,5.5) - at his desk
write(15, 26, '[A]');    // Alice (6,6) - center room
write(10, 25, '[D]');    // David (4,6) - dining area
write(11, 10, '[L]');    // Lucy (3,2) - dining area above

// Print
for (let y = 0; y <= ch + 1; y++) {
  let line = grid[y].join('');
  // Clean up: replace dots with spaces for readability
  line = line.replace(/\./g, ' ');
  console.log(line);
}

// Summary info
console.log('');
console.log('===================================================================');
console.log('  场景一 · 10m × 8m 布局');
console.log('===================================================================');
console.log('');

const zones = [
  ['DINING',   0,0,5,4, '餐桌 + 书架×3 + 椅子×2'],
  ['KITCHEN',  5,0,10,4, '冰箱 + 台面A/B + 洗碗机 + 微波炉'],
  ['BATHROOM', 0,4,5,8, '马桶 + 浴缸'],
  ['BOB-LAB',  5,4,10,8, '工作桌 + 副桌 + 砧板 + 椅子×2'],
];

for (const [name, x1, y1, x2, y2, desc] of zones) {
  console.log(`  ■ ${name.padEnd(10)} [${x1},${y1}] → [${x2},${y2}]  ${desc}`);
}
console.log('');
console.log('  内部墙壁:');
console.log('    ┃ 竖墙①  x=5, y=0→5  分隔厨房和左侧区');
console.log('    ━ 横墙    y=4, x=0→3   分隔Dining/Bathroom');
console.log('    ┃ 竖墙②  x=5, y=7→8  分隔Bathroom/Bob-Lab');
console.log('');
console.log('  机器人:');
console.log('    [A] Alice (6,6)        移动操作 (导航+开门+抓取+放置)');
console.log('    [B] Bob   (8.5,5.5)    固定臂 (精准操作，固定在Lab)');
console.log('    [D] David (4,6)        移动机器人 (导航+探索)');
console.log('    [L] Lucy  (3,2)        无人机 (空中导航+抓取)');
console.log('');
console.log('  任务物品位置 (pack_objects):');
console.log('    bowl  → 砧板(8.5,5.8)    fork → 冰箱(9.7,0.5)');
console.log('    soap  → 台面B(6.2,0.5)   apple → 餐桌(3,2)');
console.log('    tray  → 场景中央');
console.log('');
console.log('  干扰物: phone(洗碗机旁) | book(书架1) | toy_duck(马桶旁)');
console.log('===================================================================');
