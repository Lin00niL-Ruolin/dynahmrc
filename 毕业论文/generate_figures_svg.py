#!/usr/bin/env python3
"""生成详细的 SVG 论文插图"""
import os

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ============ 图2-1：DeepSeek API调用流程 ============
FIG2_1 = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 320" width="700" height="320">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#555"/>
    </marker>
    <filter id="shadow">
      <feDropShadow dx="1" dy="1" stdDeviation="1" flood-opacity="0.15"/>
    </filter>
  </defs>

  <!-- 背景 -->
  <rect width="700" height="320" fill="#fafafa" rx="8"/>

  <!-- 应用系统 -->
  <rect x="40" y="30" width="160" height="55" rx="8" fill="#e8f0fe" stroke="#4a86e8" stroke-width="2" filter="url(#shadow)"/>
  <text x="120" y="62" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">应用系统</text>

  <!-- DeepSeek SDK -->
  <rect x="40" y="130" width="160" height="55" rx="8" fill="#e8f0fe" stroke="#4a86e8" stroke-width="2" filter="url(#shadow)"/>
  <text x="120" y="162" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">DeepSeek SDK</text>

  <!-- DeepSeek API -->
  <rect x="280" y="130" width="170" height="55" rx="8" fill="#fce8e6" stroke="#d93025" stroke-width="2" filter="url(#shadow)"/>
  <text x="365" y="162" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">DeepSeek API</text>

  <!-- 大语言模型 -->
  <rect x="520" y="30" width="140" height="55" rx="8" fill="#e6f4ea" stroke="#34a853" stroke-width="2" filter="url(#shadow)"/>
  <text x="590" y="62" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">大语言模型</text>

  <!-- 箭头：应用系统→SDK -->
  <line x1="120" y1="85" x2="120" y2="120" stroke="#555" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="130" y="108" font-size="11" fill="#555">构造messages数组</text>

  <!-- 箭头：SDK→API -->
  <line x1="200" y1="155" x2="270" y2="155" stroke="#555" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="210" y="148" font-size="11" fill="#555">POST /v1/chat/completions</text>

  <!-- 箭头：API→模型 -->
  <line x1="365" y1="130" x2="580" y2="95" stroke="#555" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="400" y="108" font-size="11" fill="#555">模型推理</text>

  <!-- 箭头：模型→API -->
  <line x1="590" y1="85" x2="400" y2="120" stroke="#555" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="430" y="106" font-size="11" fill="#555">生成回复文本</text>

  <!-- 箭头：API→SDK -->
  <line x1="280" y1="170" x2="210" y2="170" stroke="#555" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="215" y="163" font-size="11" fill="#555">返回JSON响应</text>

  <!-- 箭头：SDK→应用 -->
  <line x1="120" y1="185" x2="120" y2="210" stroke="#555" stroke-width="2" marker-end="url(#arrow)"/>
  <text x="130" y="205" font-size="11" fill="#555">解析回复</text>

  <!-- 标注 -->
  <text x="350" y="280" text-anchor="middle" font-size="14" fill="#666" font-style="italic">图2-1 DeepSeek API调用流程图</text>
</svg>'''

# ============ 图4-1：系统总体架构 ============
FIG4_1 = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 750 480" width="750" height="480">
  <defs>
    <marker id="arrow2" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#555"/>
    </marker>
    <filter id="sh"><feDropShadow dx="1" dy="1" stdDeviation="1" flood-opacity="0.12"/></filter>
  </defs>

  <rect width="750" height="480" fill="#fafafa" rx="8"/>

  <!-- ===== 第1层：LLM推理层 ===== -->
  <rect x="15" y="10" width="720" height="90" rx="6" fill="#e6f4ea" stroke="#34a853" stroke-width="1.5"/>
  <text x="30" y="35" font-size="13" font-weight="bold" fill="#2e7d32">LLM推理层</text>

  <rect x="250" y="45" width="220" height="40" rx="6" fill="#fff" stroke="#34a853" stroke-width="1.5" filter="url(#sh)"/>
  <text x="360" y="70" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">DeepSeek API (deepseek-chat)</text>

  <!-- ===== 第2层：后端服务层 ===== -->
  <rect x="15" y="115" width="720" height="185" rx="6" fill="#e8f0fe" stroke="#4a86e8" stroke-width="1.5"/>
  <text x="30" y="140" font-size="13" font-weight="bold" fill="#1967d2">后端服务层 (Node.js + Express + WebSocket)</text>

  <!-- 引擎 -->
  <rect x="40" y="155" width="200" height="65" rx="6" fill="#fff" stroke="#4a86e8" stroke-width="1.5" filter="url(#sh)"/>
  <text x="140" y="180" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">DynaHMRC引擎</text>
  <text x="140" y="200" text-anchor="middle" font-size="10" fill="#666">四阶段流程编排</text>

  <!-- Agent管理器 -->
  <rect x="280" y="155" width="200" height="65" rx="6" fill="#fff" stroke="#4a86e8" stroke-width="1.5" filter="url(#sh)"/>
  <text x="380" y="180" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Agent管理器</text>
  <text x="380" y="200" text-anchor="middle" font-size="10" fill="#666">LLM调用·状态维护</text>

  <!-- 仿真环境 -->
  <rect x="520" y="155" width="190" height="65" rx="6" fill="#fff" stroke="#4a86e8" stroke-width="1.5" filter="url(#sh)"/>
  <text x="615" y="180" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">仿真环境</text>
  <text x="615" y="200" text-anchor="middle" font-size="10" fill="#666">SimEnvironment</text>

  <!-- WebSocket -->
  <rect x="280" y="240" width="200" height="40" rx="6" fill="#fff" stroke="#4a86e8" stroke-width="1.5" filter="url(#sh)"/>
  <text x="380" y="264" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">WebSocket广播 (ws库)</text>

  <!-- 箭头：引擎→Agent -->
  <line x1="240" y1="185" x2="275" y2="185" stroke="#555" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <text x="255" y="178" font-size="9" fill="#555">管理</text>

  <!-- 箭头：引擎→仿真 -->
  <line x1="240" y1="195" x2="515" y2="195" stroke="#555" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <text x="350" y="190" font-size="9" fill="#555">维护</text>

  <!-- 箭头：Agent→LLM -->
  <line x1="380" y1="155" x2="380" y2="115" stroke="#555" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#arrow2)"/>
  <text x="390" y="140" font-size="9" fill="#555">调用</text>

  <!-- 箭头：WebSocket→前端 -->
  <line x1="480" y1="260" x2="600" y2="300" stroke="#555" stroke-width="1.5" marker-end="url(#arrow2)" stroke-dasharray="5,3"/>

  <!-- ===== 第3层：前端展示层 ===== -->
  <rect x="15" y="320" width="720" height="80" rx="6" fill="#fef7e0" stroke="#f9ab00" stroke-width="1.5"/>
  <text x="30" y="345" font-size="13" font-weight="bold" fill="#e37400">前端展示层 (React + Vite + Canvas 2D)</text>

  <rect x="40" y="355" width="140" height="35" rx="5" fill="#fff" stroke="#f9ab00" stroke-width="1" filter="url(#sh)"/>
  <text x="110" y="377" text-anchor="middle" font-size="11" fill="#333">配置面板</text>

  <rect x="200" y="355" width="140" height="35" rx="5" fill="#fff" stroke="#f9ab00" stroke-width="1" filter="url(#sh)"/>
  <text x="270" y="377" text-anchor="middle" font-size="11" fill="#333">2D仿真画布</text>

  <rect x="360" y="355" width="140" height="35" rx="5" fill="#fff" stroke="#f9ab00" stroke-width="1" filter="url(#sh)"/>
  <text x="430" y="377" text-anchor="middle" font-size="11" fill="#333">对话日志</text>

  <rect x="520" y="355" width="140" height="35" rx="5" fill="#fff" stroke="#f9ab00" stroke-width="1" filter="url(#sh)"/>
  <text x="590" y="377" text-anchor="middle" font-size="11" fill="#333">状态面板</text>

  <!-- ===== 第4层：3D仿真层 ===== -->
  <rect x="15" y="415" width="720" height="50" rx="6" fill="#fce8e6" stroke="#d93025" stroke-width="1.5"/>
  <text x="30" y="440" font-size="13" font-weight="bold" fill="#c5221f">3D仿真层 (可选 — BestMan / PyBullet)</text>

  <rect x="280" y="425" width="200" height="30" rx="5" fill="#fff" stroke="#d93025" stroke-width="1"/>
  <text x="380" y="445" text-anchor="middle" font-size="11" fill="#333">机械臂IK·无人机·物理仿真</text>

  <!-- 标注 -->
  <text x="375" y="475" text-anchor="middle" font-size="13" fill="#666" font-style="italic">图4-1 系统总体架构图</text>
</svg>'''

# ============ 图4-2：四阶段协作流程 ============
FIG4_2 = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 750 480" width="750" height="480">
  <defs>
    <marker id="a3" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#555"/>
    </marker>
  </defs>

  <rect width="750" height="480" fill="#fafafa" rx="8"/>

  <!-- 左侧时间轴 -->
  <line x1="90" y1="30" x2="90" y2="440" stroke="#ccc" stroke-width="3" stroke-dasharray="6,4"/>

  <!-- === 阶段一 === -->
  <rect x="140" y="20" width="560" height="90" rx="8" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
  <text x="160" y="48" font-size="16" font-weight="bold" fill="#1565c0">阶段一：自我描述 (Self-Description)</text>
  <text x="160" y="70" font-size="12" fill="#333">所有机器人同时进行自我介绍，输出内容包含类型、能力、局限性</text>
  <text x="160" y="90" font-size="12" fill="#333">和团队角色定位。建立团队共识，了解成员能力边界。</text>
  <circle cx="90" cy="65" r="12" fill="#1976d2"/>
  <text x="90" y="70" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">1</text>
  <line x1="102" y1="65" x2="135" y2="65" stroke="#555" stroke-width="2" marker-end="url(#a3)"/>

  <!-- 箭头 -->
  <line x1="420" y1="110" x2="420" y2="125" stroke="#555" stroke-width="2" marker-end="url(#a3)"/>

  <!-- === 阶段二 === -->
  <rect x="140" y="130" width="560" height="90" rx="8" fill="#fff3e0" stroke="#f57c00" stroke-width="2"/>
  <text x="160" y="158" font-size="16" font-weight="bold" fill="#e65100">阶段二：任务分配与竞选</text>
  <text x="160" y="180" font-size="12" fill="#333">各机器人阅读他人自我介绍后，提出分工方案和竞选演讲。</text>
  <text x="160" y="200" font-size="12" fill="#333">方案需匹配任务需求与各成员能力，实现并行效率最大化。</text>
  <circle cx="90" cy="175" r="12" fill="#f57c00"/>
  <text x="90" y="180" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">2</text>
  <line x1="102" y1="175" x2="135" y2="175" stroke="#555" stroke-width="2" marker-end="url(#a3)"/>

  <line x1="420" y1="220" x2="420" y2="235" stroke="#555" stroke-width="2" marker-end="url(#a3)"/>

  <!-- === 阶段三 === -->
  <rect x="140" y="240" width="560" height="80" rx="8" fill="#e8f5e9" stroke="#388e3c" stroke-width="2"/>
  <text x="160" y="268" font-size="16" font-weight="bold" fill="#2e7d32">阶段三：领导选举 (Leader Election)</text>
  <text x="160" y="290" font-size="12" fill="#333">所有机器人投票选举领导者。系统收集投票并统计，</text>
  <text x="160" y="308" font-size="12" fill="#333">得票最高者当选为团队领导。</text>
  <circle cx="90" cy="280" r="12" fill="#388e3c"/>
  <text x="90" y="285" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">3</text>
  <line x1="102" y1="280" x2="135" y2="280" stroke="#555" stroke-width="2" marker-end="url(#a3)"/>

  <line x1="420" y1="320" x2="420" y2="335" stroke="#555" stroke-width="2" marker-end="url(#a3)"/>

  <!-- === 阶段四 === -->
  <rect x="140" y="340" width="560" height="100" rx="8" fill="#fce4ec" stroke="#c62828" stroke-width="2"/>
  <text x="160" y="368" font-size="16" font-weight="bold" fill="#b71c1c">阶段四：执行与反思</text>
  <text x="160" y="390" font-size="12" fill="#333">循环执行：Agent决策动作→仿真执行→反馈→下一决策</text>
  <text x="160" y="410" font-size="12" fill="#333">每N步反思：Agent总结失败经验→提出改进计划→</text>
  <text x="160" y="428" font-size="12" fill="#333">领导者汇总→更新协作计划。形成闭环控制。</text>
  <circle cx="90" cy="385" r="12" fill="#c62828"/>
  <text x="90" y="390" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">4</text>
  <line x1="102" y1="385" x2="135" y2="385" stroke="#555" stroke-width="2" marker-end="url(#a3)"/>

  <text x="375" y="475" text-anchor="middle" font-size="13" fill="#666" font-style="italic">图4-2 四阶段协作流程图</text>
</svg>'''

# ============ 图4-3：数据实体关系 ============
FIG4_3 = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 280" width="700" height="280">
  <defs>
    <marker id="a4" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#555"/>
    </marker>
  </defs>

  <rect width="700" height="280" fill="#fafafa" rx="8"/>

  <!-- 引擎 -->
  <rect x="30" y="80" width="150" height="70" rx="8" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
  <text x="105" y="115" text-anchor="middle" font-size="14" font-weight="bold" fill="#1565c0">DynaHMRC引擎</text>
  <text x="105" y="135" text-anchor="middle" font-size="10" fill="#555">流程编排·调度</text>

  <!-- Agent -->
  <rect x="250" y="30" width="150" height="70" rx="8" fill="#fff3e0" stroke="#f57c00" stroke-width="2"/>
  <text x="325" y="65" text-anchor="middle" font-size="14" font-weight="bold" fill="#e65100">机器人Agent</text>
  <text x="325" y="85" text-anchor="middle" font-size="10" fill="#555">×4（每机器人1个）</text>

  <!-- 仿真环境 -->
  <rect x="250" y="140" width="150" height="70" rx="8" fill="#e8f5e9" stroke="#388e3c" stroke-width="2"/>
  <text x="325" y="175" text-anchor="middle" font-size="14" font-weight="bold" fill="#2e7d32">Simulation环境</text>
  <text x="325" y="195" text-anchor="middle" font-size="10" fill="#555">场景图·状态管理</text>

  <!-- 机器人状态 -->
  <rect x="470" y="30" width="180" height="70" rx="8" fill="#fce4ec" stroke="#c62828" stroke-width="2"/>
  <text x="560" y="60" text-anchor="middle" font-size="13" font-weight="bold" fill="#b71c1c">RobotStatus</text>
  <text x="560" y="78" text-anchor="middle" font-size="10" fill="#555">位置·夹爪·抓取物</text>

  <!-- 场景物体 -->
  <rect x="470" y="140" width="180" height="70" rx="8" fill="#fff" stroke="#7b1fa2" stroke-width="2"/>
  <text x="560" y="170" text-anchor="middle" font-size="13" font-weight="bold" fill="#6a1b9a">SceneObject</text>
  <text x="560" y="188" text-anchor="middle" font-size="10" fill="#555">家具·物品·容器</text>

  <!-- LLM -->
  <rect x="470" y="200" width="180" height="50" rx="8" fill="#f3e5f5" stroke="#7b1fa2" stroke-width="2"/>
  <text x="560" y="230" text-anchor="middle" font-size="13" font-weight="bold" fill="#6a1b9a">DeepSeek API</text>

  <!-- 连线 -->
  <line x1="180" y1="110" x2="245" y2="75" stroke="#555" stroke-width="1.5" marker-end="url(#a4)"/>
  <text x="200" y="88" font-size="10" fill="#555">管理 1:N</text>

  <line x1="180" y1="120" x2="245" y2="165" stroke="#555" stroke-width="1.5" marker-end="url(#a4)"/>
  <text x="195" y="145" font-size="10" fill="#555">维护 1:1</text>

  <line x1="400" y1="70" x2="465" y2="60" stroke="#555" stroke-width="1.5" marker-end="url(#a4)"/>
  <text x="425" y="62" font-size="10" fill="#555">包含</text>

  <line x1="400" y1="170" x2="465" y2="170" stroke="#555" stroke-width="1.5" marker-end="url(#a4)"/>
  <text x="425" y="162" font-size="10" fill="#555">包含</text>

  <line x1="325" y1="100" x2="325" y2="135" stroke="#555" stroke-width="1.5" marker-end="url(#a4)" stroke-dasharray="4,3"/>
  <text x="335" y="120" font-size="10" fill="#555">调用</text>

  <line x1="560" y1="200" x2="560" y2="215" stroke="#555" stroke-width="1" marker-end="url(#a4)"/>

  <text x="350" y="270" text-anchor="middle" font-size="13" fill="#666" font-style="italic">图4-3 数据实体关系图</text>
</svg>'''

# ============ 图3-1：用例图 ============
FIG3_1 = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 350" width="600" height="350">
  <rect width="600" height="350" fill="#fafafa" rx="8"/>
  <text x="300" y="340" text-anchor="middle" font-size="13" fill="#666" font-style="italic">图3-1 系统用例图</text>
  <!-- 用户 -->
  <ellipse cx="80" cy="175" rx="50" ry="30" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
  <text x="80" y="180" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">用户</text>
  <!-- 用例 -->
  <ellipse cx="280" cy="60" rx="75" ry="25" fill="#fff" stroke="#333" stroke-width="1.5"/>
  <text x="280" y="65" text-anchor="middle" font-size="12" fill="#333">配置任务场景</text>
  <ellipse cx="280" cy="140" rx="75" ry="25" fill="#fff" stroke="#333" stroke-width="1.5"/>
  <text x="280" y="145" text-anchor="middle" font-size="12" fill="#333">启动协作流程</text>
  <ellipse cx="280" cy="220" rx="75" ry="25" fill="#fff" stroke="#333" stroke-width="1.5"/>
  <text x="280" y="225" text-anchor="middle" font-size="12" fill="#333">监控运行状态</text>
  <ellipse cx="280" cy="300" rx="75" ry="25" fill="#fff" stroke="#333" stroke-width="1.5"/>
  <text x="280" y="305" text-anchor="middle" font-size="12" fill="#333">查看执行结果</text>
  <!-- 连线 -->
  <line x1="130" y1="165" x2="200" y2="60" stroke="#333" stroke-width="1"/>
  <line x1="130" y1="172" x2="200" y2="140" stroke="#333" stroke-width="1"/>
  <line x1="130" y1="180" x2="200" y2="220" stroke="#333" stroke-width="1"/>
  <line x1="130" y1="188" x2="200" y2="300" stroke="#333" stroke-width="1"/>
  <!-- 系统边界 -->
  <rect x="190" y="30" width="200" height="310" rx="8" fill="none" stroke="#666" stroke-width="1.5" stroke-dasharray="5,3"/>
  <text x="290" y="50" text-anchor="middle" font-size="10" fill="#666">DynaHMRC系统</text>
</svg>'''

images = {
    'fig2-1': FIG2_1,
    'fig3-1': FIG3_1,
    'fig4-1': FIG4_1,
    'fig4-2': FIG4_2,
    'fig4-3': FIG4_3,
}

if __name__ == '__main__':
    print('生成SVG插图...')
    for name, svg in images.items():
        path = os.path.join(FIG_DIR, f'{name}.svg')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(svg.strip())
        print(f'  ✅ {name}.svg')
    print('完成! 可在浏览器中打开查看。')
