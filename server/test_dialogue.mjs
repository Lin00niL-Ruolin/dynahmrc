import WebSocket from 'ws';
import http from 'http';

// Create run via API
async function main() {
  const resp = await fetch('http://localhost:3001/api/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      taskType: 'make_sandwich',
      layout: 'scene1',
      robots: [{name:'Alice'},{name:'Bob'},{name:'David'},{name:'Lucy'}],
      useBestMan: false,
    }),
  });
  const { runId } = await resp.json();
  console.log(`Run: ${runId}\n`);

  let dialogueCount = 0;
  let lastState = null;
  const ws = new WebSocket(`ws://localhost:3001/ws/${runId}`);

  ws.on('open', () => {
    console.log('WS connected, starting...\n');
    ws.send(JSON.stringify({ command: 'start' }));
  });

  ws.on('message', (data) => {
    const msg = JSON.parse(data.toString());

    if (msg.type === 'dialogue') {
      const d = msg.data;
      dialogueCount++;
      const stageMap = {
        self_description: '📝 Self-Description',
        task_allocation_bidding: '📋 Task Allocation',
        leader_election: '🗳️ Leader Election',
        execution_reflection: '⚡ Execution',
        completed: '✅ Complete'
      };
      console.log(`[${stageMap[d.stage] || d.stage}] ${d.robotName}`);

      if (d.thoughts && !d.robotName.startsWith('[SYS')) {
        console.log(`  💭 ${(d.thoughts || '').slice(0, 150).replace(/\n/g, ' ')}`);
      }
      
      // Show content without Thoughts header
      const content = (d.content || '');
      const cleanContent = content.replace(/^\*{0,2}Thoughts?\*{0,2}\s*:.*?\n/i, '').trim();
      const lines = cleanContent.split('\n').filter(l => l.trim()).slice(0, 4);
      if (lines.length > 0) {
        lines.forEach(l => {
          const display = l.length > 120 ? l.slice(0, 120) + '...' : l;
          console.log(`  ${display}`);
        });
      }

      if (d.vote) console.log(`  🗳️ Votes: ${d.vote}`);
      console.log('');
    }

    if (msg.type === 'state') {
      lastState = msg.data;
      if (lastState.stage === 'completed' || lastState.stage === 'stopped') {
        console.log('='.repeat(55));
        console.log(`🏁 DONE! ${dialogueCount} dialogues`);
        console.log(`   Leader: ${lastState.leader}`);
        const actions = lastState.actions || [];
        console.log(`   Total actions: ${actions.length}`);

        // Show action summary
        const actionCounts = {};
        actions.forEach(a => {
          const key = a.robotName + ':' + a.actionType;
          actionCounts[key] = (actionCounts[key] || 0) + 1;
        });
        console.log('\n📊 Actions by robot:');
        Object.entries(actionCounts).forEach(([k, v]) => console.log(`   ${k} ×${v}`));

        // Check feasibility
        console.log('\n🔍 Feasibility check:');
        let issues = 0;
        actions.forEach(a => {
          if (a.actionType === 'pick' || a.actionType === 'place') {
            if (a.robotName === 'David') {
              console.log(`   ❌ David不能 pick/place! action: ${a.actionType}`);
              issues++;
            }
          }
        });
        if (issues === 0) console.log('   ✅ All actions feasible');

        console.log(`\n📦 Placed: ${JSON.stringify(lastState.placedObjects)}`);
        console.log(`   Targets: ${JSON.stringify(lastState.taskTargets)}`);

        ws.close();
        process.exit(0);
      }
    }
  });

  ws.on('error', (e) => { console.error('WS Error:', e.message); });
  setTimeout(() => { console.log('⏰ Timeout'); process.exit(1); }, 120000);
}

main().catch(e => { console.error(e); process.exit(1); });
