"""Portable HTML and one-file Canvas views of precomputed aggregate metrics."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'qa/basic_metrics_20261030'
data=json.loads((OUT/'metrics.json').read_text(encoding='utf8'))
payload=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')

canvas=r'''import {Stack,Row,Grid,H1,H2,Text,Table,Select,Stat,Divider,Callout,useState,useHostTheme} from "cursor/canvas";
const data:any=__DATA__;
const sources=["phishingpot","enron_spam","ephishllm","kaggle","pooled"];
const names:Record<string,string>={phishingpot:"PhishingPot",enron_spam:"Enron",ephishllm:"ePhishLLM",kaggle:"Kaggle",pooled:"合并统计"};
const roleNames:Record<string,string>={action:"Action",context:"Context",persuasion:"Persuasion",purpose:"Purpose"};
const roles=["action","context","persuasion","purpose"];
const pct=(n:number|null)=>n===null?"未定义":(n*100).toFixed(2)+"%";
const dec=(n:number|null)=>n===null?"未定义":n.toFixed(4);
export default function Metrics(){
 const theme=useHostTheme();const [source,setSource]=useState("pooled");
 const c=data.complementarity[source],ev=data.evidence.overall;
 return <Stack gap={18} style={{padding:24,maxWidth:1500,margin:"0 auto",color:theme.text.primary}}>
  <H1>V5 基础指标 · 2026-10-30</H1>
  <Text tone="secondary">冻结四库各500封；未校准四角色等权均值；固定阈值0.5；无PPO。正类依照各库的垃圾邮件／钓鱼任务定义。</Text>
  <Grid columns={4} gap={16}>
   <Stat label="全样本 Accuracy" value={pct(data.classification.pooled.accuracy_all)}/>
   <Stat label="已评分样本 F1" value={pct(data.classification.pooled.f1)}/>
   <Stat label="评分覆盖率" value="1999 / 2000 · 99.95%"/>
   <Stat label="证据有效字段覆盖" value={pct(ev.valid_feature_coverage)}/>
  </Grid>
  <H2>分类、防御与正常邮件保护</H2>
  <Text size="small" tone="secondary">Accuracy以全部样本为分母；Precision、Recall、F1、FNR、HAM Recall、FPR以有评分的样本为基础。PhishingPot有1封正类超预算；其全正类召回为{pct(data.classification.phishingpot.recall_all_positive)}。</Text>
  <Table headers={["数据集","N / 有分数","Accuracy 全样本","Precision","正类 Recall","F1","FNR","HAM Recall","FPR"]}
   rows={sources.map(s=>{const m=data.classification[s];return[names[s],`${m.n} / ${m.n_scored}`,pct(m.accuracy_all),pct(m.precision),pct(m.recall),pct(m.f1),pct(m.fnr),pct(m.ham_recall),pct(m.fpr)]})}/>
  <Table headers={["数据集","TP","TN","FP","FN","未评分","正类总数","负类总数"]}
   rows={sources.map(s=>{const m=data.classification[s];return[names[s],m.tp,m.tn,m.fp,m.fn,m.n_unknown,m.positive_support,m.negative_support]})}/>
  <H2>原始相对分的概率质量与排序能力</H2>
  <Text size="small" tone="secondary">Brier、ECE在有分数的样本上计算，越低越好。ECE为正类分数的10个等宽分箱；AUC为补充排序诊断，越高越好。不同任务和分数分布混合时，合并AUC/ECE不能代替分库结果。</Text>
  <Table headers={["数据集","Brier ↓","ECE10 ↓","ROC-AUC ↑","正类预测 / 有分数"]}
   rows={sources.map(s=>{const m=data.classification[s];return[names[s],dec(m.brier),dec(m.ece10),dec(m.roc_auc),`${m.positive_predictions} / ${m.n_scored}`]})}/>
  <Callout>Enron虽然在0.5阈值下全判spam，AUC仍为0.8646：有排序信号，但固定阈值表现差。校准与阈值需使用独立校准／验证集检验，本轮未用测试标签调整。</Callout>
  <Divider/>
  <Row justify="space-between" wrap><H2>Agent质量与互补性</H2><Select value={source} onChange={setSource} options={sources.map(s=>({value:s,label:names[s]}))}/></Row>
  <Table headers={["Agent","Accuracy 全样本","Precision","Recall","F1","Brier","ECE10","AUC"]}
   rows={roles.map(role=>{const m=data.agents[source][role];return[roleNames[role],pct(m.accuracy_all),pct(m.precision),pct(m.recall),pct(m.f1),dec(m.brier),dec(m.ece10),dec(m.roc_auc)]})}/>
  <Text>当前选择：{names[source]}。四角色共同误判 {c.all_four_wrong_count} / {c.complete_n}，共同误判率 {pct(c.all_four_wrong_rate)}。</Text>
  <Table headers={["Agent 对","错误相关性","共同误判数 / N","共同误判率","预测分歧率"]}
   rows={c.pairs.map((p:any)=>[`${roleNames[p.left]} × ${roleNames[p.right]}`,dec(p.error_correlation),`${p.joint_error_count} / ${p.n}`,pct(p.joint_error_rate),pct(p.disagreement_rate)])}/>
  <Text size="small" tone="secondary">错误相关性是二元错误指示量的Pearson相关；共同误判率为“两角色都错的邮件数 / 共同有分数的邮件数”。这些结果基于当前0.5阈值。</Text>
  <Divider/>
  <H2>8封辅助证据测试：输出质量</H2>
  <Table headers={["指标","计数","比例","含义"]} rows={[
   ["JSON Validity（含重试）",`${ev.json_syntax_valid} / ${ev.generation_attempts}`,pct(ev.json_validity),"完整严格JSON语法合法"],
   ["首轮 JSON Validity",`19 / 32`,pct(ev.json_validity_initial),"每封每角色第一次生成"],
   ["输出字段覆盖率",`${ev.emitted_feature_slots} / ${ev.assigned_feature_slots}`,pct(ev.emitted_feature_coverage),"可解析输出中出现的已分配字段ID，去重后计数"],
   ["有效字段覆盖率",`${ev.valid_feature_slots} / ${ev.assigned_feature_slots}`,pct(ev.valid_feature_coverage),"通过结构、字段关系和原文引用校验；含合法found=false"],
   ["有原文证据字段占比",`${ev.found_grounded_feature_slots} / ${ev.assigned_feature_slots}`,pct(ev.found_grounded_feature_fraction),"found=true且引用通过，不代表语义正确"],
   ["完整角色报告通过率",`${ev.final_full_contract_valid} / ${ev.role_reports}`,pct(ev.final_full_contract_rate),"有限重试后全部字段通过校验"],
   ["四角色全通过邮件",`${data.evidence.all_four_contract_valid_mails} / ${data.evidence.mails}`,pct(data.evidence.all_four_contract_valid_mails/data.evidence.mails),"仅结构和引用，不作语义认证"]]}/>
  <Text>共32次首轮生成和21次重试。32条角色报告中，12条通过、4条部分通过、16条无法可靠提取；22条没有有效非空证据。</Text>
  <Table headers={["角色","完整报告通过","有效字段 / 分配字段","有效字段覆盖率","非空原文证据字段"]}
   rows={roles.map(role=>{const e=data.evidence.by_role[role];return[roleNames[role],`${e.final_full_contract_valid} / ${e.role_reports}`,`${e.valid_feature_slots} / ${e.assigned_feature_slots}`,pct(e.valid_feature_coverage),e.found_grounded_feature_slots]})}/>
  <H2>本轮尚不能给出数值的指标</H2>
  <Text>Evidence Accuracy与Role Consistency需要完整语义标注，现有人工抽查不足以估计准确率。PPO相对等权平均的提升需要本版PPO结果，本轮未运行，因此不填0。</Text>
  <H2>复算与分母</H2>
  <Text size="small" tone="secondary">Precision=TP/(TP+FP)；Recall=TP/(TP+FN)；F1=2TP/(2TP+FP+FN)；HAM Recall=TN/(TN+FP)。无正类预测时Precision未定义。合并结果为混合任务的描述性统计。来源：2026-10-30冻结运行原始记录；独立sklearn/NumPy核对209项一致。指标数据与脚本随报告保存。</Text>
 </Stack>
}
'''.replace('__DATA__',payload)
(OUT/'v5-basic-metrics-20261030.canvas.tsx').write_text(canvas,encoding='utf8')

html=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V5 基础指标 · 2026-10-30</title>
<style>:root{font:14px/1.6 system-ui,"Microsoft YaHei",sans-serif;color:#222;background:#fafafa}body{max-width:1480px;margin:28px auto;padding:0 22px 50px}h1{font-size:24px;margin:0}h2{font-size:18px;margin:26px 0 10px}p{margin:8px 0}.muted,small{color:#666}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin:24px 0}.stat{border-top:2px solid #36687a;padding-top:9px}.stat b{font-size:22px;display:block}.wrap{overflow:auto;border:1px solid #ddd;border-radius:5px;margin:10px 0}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;white-space:nowrap}th,td{padding:9px 12px;border-bottom:1px solid #e4e4e4;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#f0f2f3;font-weight:600}tr:last-child td{border:0}tr:hover td{background:#f3f6f7}.note{padding:12px 16px;border-left:3px solid #36687a;background:#edf3f5}.selectrow{display:flex;justify-content:space-between;align-items:center;gap:20px}select{padding:7px 12px;border:1px solid #bbb;background:inherit;color:inherit;border-radius:4px}footer{margin-top:30px;border-top:1px solid #ddd;padding-top:15px}code{font-size:12px}@media(max-width:800px){.stats{grid-template-columns:repeat(2,1fr)}}@media(prefers-color-scheme:dark){:root{background:#191b1d;color:#e6e6e6}.muted,small{color:#aaa}th{background:#25292d}th,td{border-color:#373d41}.wrap{border-color:#41464b}tr:hover td{background:#252b2e}.note{background:#232e33}select{border-color:#69747b}}</style>
<h1>V5 基础指标 · 2026-10-30</h1><p class="muted">冻结四库各500封；未校准四角色等权均值；固定阈值0.5；无PPO。正类依照各库的垃圾邮件／钓鱼任务定义。</p>
<div class="stats" id="stats"></div>
<h2>分类、防御与正常邮件保护</h2><p class="muted">Accuracy以全部样本为分母；Precision、Recall、F1、FNR、HAM Recall、FPR以有评分的样本计算。PhishingPot的1封超预算正类另外保留，其全正类召回为1.0490%。</p><div id="classification"></div><div id="confusion"></div>
<h2>原始相对分的概率质量与排序能力</h2><p class="muted">Brier、ECE在有分数的样本上计算，越低越好；ECE使用正类分数的10个等宽分箱。AUC是补充排序诊断，越高越好。混合任务的合并AUC/ECE不能代替分库结果。</p><div id="probability"></div><p class="note">Enron在0.5阈值下全判spam，但AUC为0.8646，说明仍有排序信号。校准与阈值需使用独立校准／验证集检验，本轮未用测试标签调整。</p>
<div class="selectrow"><h2>Agent质量与互补性</h2><label>查看数据集 <select id="source"></select></label></div><div id="agents"></div><p id="allwrong"></p><div id="pairs"></div><small>错误相关性：二元错误指示量的Pearson相关。共同误判率：两角色都错的邮件数 / 共同有分数的邮件数。均基于当前0.5阈值。</small>
<h2>8封辅助证据测试：输出质量</h2><div id="evidence"></div><p>共32次首轮生成和21次重试。32条角色报告中12条通过、4条部分通过、16条无法可靠提取；22条没有有效非空证据。</p><div id="evidenceRoles"></div>
<h2>尚不能给出数值的指标</h2><p>Evidence Accuracy与Role Consistency需要完整语义标注；人工抽查不足以估计准确率。PPO相对等权平均的提升需要本版PPO结果，本轮未运行，因此不填0。</p>
<footer><b>计算口径</b><p>Precision=TP/(TP+FP)，Recall=TP/(TP+FN)，F1=2TP/(2TP+FP+FN)，FNR=FN/(TP+FN)，HAM Recall=TN/(TN+FP)，FPR=FP/(TN+FP)。无正类预测时Precision未定义。Brier=mean((p-y)²)；ECE=Σ(n_bin/n × |mean(p)-mean(y)|)。引用来自原文不保证字段语义正确。</p><small>来源：2026-10-30冻结运行原始记录；1999封有分数，1封超预算；完整指标已与sklearn/NumPy核对209项一致。完整精度和分箱计数见同目录metrics.json，复算入口scripts/calculate_basic_metrics.py。</small></footer>
<script>
const D=__DATA__,sources=['phishingpot','enron_spam','ephishllm','kaggle','pooled'],names={phishingpot:'PhishingPot',enron_spam:'Enron',ephishllm:'ePhishLLM',kaggle:'Kaggle',pooled:'合并统计'},roles=['action','context','persuasion','purpose'],roleNames={action:'Action',context:'Context',persuasion:'Persuasion',purpose:'Purpose'};
const pct=n=>n===null?'未定义':(n*100).toFixed(2)+'%',dec=n=>n===null?'未定义':n.toFixed(4),esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
function table(id,heads,rows){document.getElementById(id).innerHTML='<div class="wrap"><table><thead><tr>'+heads.map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(x=>'<td>'+esc(x)+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>'}
document.getElementById('stats').innerHTML=[['全样本 Accuracy',pct(D.classification.pooled.accuracy_all)],['已评分样本 F1',pct(D.classification.pooled.f1)],['评分覆盖率','1999 / 2000 · 99.95%'],['证据有效字段覆盖',pct(D.evidence.overall.valid_feature_coverage)]].map(([k,v])=>'<div class="stat">'+k+'<b>'+v+'</b></div>').join('');
table('classification',['数据集','N / 有分数','Accuracy 全样本','Precision','正类 Recall','F1','FNR','HAM Recall','FPR'],sources.map(s=>{let m=D.classification[s];return[names[s],m.n+' / '+m.n_scored,...['accuracy_all','precision','recall','f1','fnr','ham_recall','fpr'].map(k=>pct(m[k]))]}));
table('confusion',['数据集','TP','TN','FP','FN','未评分','正类总数','负类总数'],sources.map(s=>{let m=D.classification[s];return[names[s],...['tp','tn','fp','fn','n_unknown','positive_support','negative_support'].map(k=>m[k])]}));
table('probability',['数据集','Brier ↓','ECE10 ↓','ROC-AUC ↑','正类预测 / 有分数'],sources.map(s=>{let m=D.classification[s];return[names[s],dec(m.brier),dec(m.ece10),dec(m.roc_auc),m.positive_predictions+' / '+m.n_scored]}));
document.getElementById('source').innerHTML=sources.map(s=>'<option value="'+s+'" '+(s==='pooled'?'selected':'')+'>'+names[s]+'</option>').join('');
function renderAgents(){let s=document.getElementById('source').value,c=D.complementarity[s];table('agents',['Agent','Accuracy 全样本','Precision','Recall','F1','Brier','ECE10','AUC'],roles.map(r=>{let m=D.agents[s][r];return[roleNames[r],...['accuracy_all','precision','recall','f1'].map(k=>pct(m[k])),dec(m.brier),dec(m.ece10),dec(m.roc_auc)]}));document.getElementById('allwrong').textContent=names[s]+'：四角色共同误判 '+c.all_four_wrong_count+' / '+c.complete_n+'，共同误判率 '+pct(c.all_four_wrong_rate);table('pairs',['Agent 对','错误相关性','共同误判数 / N','共同误判率','预测分歧率'],c.pairs.map(p=>[roleNames[p.left]+' × '+roleNames[p.right],dec(p.error_correlation),p.joint_error_count+' / '+p.n,pct(p.joint_error_rate),pct(p.disagreement_rate)]))}
document.getElementById('source').addEventListener('change',renderAgents);renderAgents();
let e=D.evidence.overall;table('evidence',['指标','计数','比例','含义'],[['JSON Validity（含重试）',e.json_syntax_valid+' / '+e.generation_attempts,pct(e.json_validity),'完整严格JSON语法合法'],['首轮 JSON Validity','19 / 32',pct(e.json_validity_initial),'每封每角色第一次生成'],['输出字段覆盖率',e.emitted_feature_slots+' / '+e.assigned_feature_slots,pct(e.emitted_feature_coverage),'可解析输出中出现的已分配字段ID，去重后计数'],['有效字段覆盖率',e.valid_feature_slots+' / '+e.assigned_feature_slots,pct(e.valid_feature_coverage),'通过结构、字段关系与原文引用校验；含合法found=false'],['有原文证据字段占比',e.found_grounded_feature_slots+' / '+e.assigned_feature_slots,pct(e.found_grounded_feature_fraction),'found=true且引用通过，不代表语义正确'],['完整角色报告通过率',e.final_full_contract_valid+' / '+e.role_reports,pct(e.final_full_contract_rate),'有限重试后全部字段通过校验'],['四角色全通过邮件',D.evidence.all_four_contract_valid_mails+' / '+D.evidence.mails,pct(D.evidence.all_four_contract_valid_mails/D.evidence.mails),'仅结构和引用，不作语义认证']]);table('evidenceRoles',['角色','完整报告通过','有效字段 / 分配字段','有效字段覆盖率','非空原文证据字段'],roles.map(r=>{let e=D.evidence.by_role[r];return[roleNames[r],e.final_full_contract_valid+' / '+e.role_reports,e.valid_feature_slots+' / '+e.assigned_feature_slots,pct(e.valid_feature_coverage),e.found_grounded_feature_slots]}));
</script></html>'''.replace('__DATA__',payload)
(OUT/'basic_metrics_report.html').write_text(html,encoding='utf8')
print(json.dumps(dict(canvas_bytes=len(canvas.encode()),html_bytes=len(html.encode()))))
