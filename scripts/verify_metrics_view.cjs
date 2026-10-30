// Static JavaScript / selector-output verification; no browser automation.
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const root = path.resolve(__dirname, '../qa/basic_metrics_20261030');
const html = fs.readFileSync(path.join(root, 'basic_metrics_report.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = new Map();
const document = { getElementById(id) {
  if (!elements.has(id)) elements.set(id, { innerHTML:'', textContent:'', value:'pooled', callbacks:{},
    addEventListener(event, callback) { this.callbacks[event] = callback; } });
  return elements.get(id);
}};
new vm.Script(script).runInNewContext({ document });
if (!elements.get('stats').innerHTML.includes('46.80%')) throw new Error('headline mismatch');
if (!elements.get('classification').innerHTML.includes('43.40%')) throw new Error('source metric mismatch');
if (!elements.get('evidence').innerHTML.includes('52.83%')) throw new Error('evidence denominator mismatch');
const data = JSON.parse(fs.readFileSync(path.join(root, 'metrics.json'), 'utf8'));
for (const source of ['phishingpot','enron_spam','ephishllm','kaggle','pooled']) {
  const select = elements.get('source'); select.value = source; select.callbacks.change();
  const expected = (data.agents[source].action.accuracy_all * 100).toFixed(2) + '%';
  if (!elements.get('agents').innerHTML.includes(expected)) throw new Error('Agent selector mismatch: ' + source);
  if ((elements.get('pairs').innerHTML.match(/<tr>/g) || []).length !== 7) throw new Error('pair rows mismatch');
}
const check = { status:'passed', scope:'HTML JavaScript parses and five cohort selections produce expected metric cells; not a rendered-browser visual test' };
fs.writeFileSync(path.join(root, 'view_verification.json'), JSON.stringify(check, null, 2));
console.log(JSON.stringify(check));
