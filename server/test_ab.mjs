import WebSocket from 'ws';
const resp = await fetch('http://localhost:3001/api/run', {
  method: 'POST', headers: {'Content-Type':'application/json'},
  body: JSON.stringify({
    taskType:'make_sandwich', layout:'scene1',
    robots:[{name:'Alice'},{name:'Bob'}], useBestMan:false, maxSteps:30
  })
});
const { runId } = await resp.json();
console.log('Run:', runId);

let step = 0;
const ws = new WebSocket(`ws://localhost:3001/ws/${runId}`);
ws.on('open', () => ws.send(JSON.stringify({command:'start'})));
ws.on('message', (data) => {
  const msg = JSON.parse(data.toString());
  if (msg.type === 'dialogue' && msg.data.stage === 'execution_reflection') {
    step++;
    const d = msg.data;
    const actionMatch = d.content?.match(/Action: (\w+)\(([^)]*)\)/) || [];
    const fbMatch = d.content?.match(/Feedback: (.+)/) || [];
    const action = actionMatch[1] || '';
    const fb = (fbMatch[1] || '').slice(0,60);
    console.log(`Step ${step}: ${d.robotName} → ${action}(${actionMatch[2]||''}) → ${fb.includes('FAIL')||fb.includes('fail')?'❌':'✅'}`);
    if (step >= 40 || fb.includes('TASK COMPLETE')) {
      console.log('=== DONE ===');
      ws.close(); process.exit(0);
    }
  }
  if (msg.type === 'state' && (msg.data.stage==='completed'||msg.data.stage==='stopped')) {
    const placed = msg.data.placedObjects;
    console.log('Placed:', JSON.stringify(placed));
    ws.close(); process.exit(0);
  }
});
setTimeout(() => process.exit(1), 90000);
