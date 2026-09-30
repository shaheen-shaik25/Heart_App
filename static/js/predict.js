const form = document.getElementById('form');
const $ = id => document.getElementById(id);
const FEATURES = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'];
const COLORS = {success:'#198754', warning:'#e0a100', danger:'#dc3545'};
const SAMPLES = window.SAMPLES;
let lastInputs = null, whatifTimer = null;

function fill(v){ FEATURES.forEach(f => form.elements[f].value = v[f]); }
$('sampleLow').onclick = () => fill(SAMPLES.low);
$('sampleHigh').onclick = () => fill(SAMPLES.high);
$('reset').onclick = () => setTimeout(() => { $('output').classList.add('hidden'); $('placeholder').classList.remove('hidden'); clearErrors(); });

function clearErrors(){ FEATURES.forEach(f => { $('err_'+f).textContent=''; form.elements[f].classList.remove('err'); }); }
function payload(){ const o={}; new FormData(form).forEach((v,k)=>o[k]=v); return o; }

form.addEventListener('submit', async e => {
  e.preventDefault(); clearErrors();
  const btn = form.querySelector('button[type=submit]'); btn.disabled = true; btn.textContent = 'Analysing…';
  try {
    const r = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload())});
    const d = await r.json();
    if (!d.ok && d.errors && d.errors.server) { alert('Server error: '+d.errors.server); return; }
    if (!d.ok) { Object.entries(d.errors).forEach(([f,m]) => { $('err_'+f).textContent = m; form.elements[f].classList.add('err'); }); return; }
    render(d);
  } catch(err){ alert('Could not reach the server. Is `python app.py` still running? Check the terminal for errors.\n\n'+err); }
  finally { btn.disabled = false; btn.textContent = '🔍 Analyse risk'; }
});

function render(d){
  lastInputs = d.inputs;
  $('placeholder').classList.add('hidden'); $('output').classList.remove('hidden');
  setGauge(d.percent, d.color);
  const b = $('riskBadge'); b.textContent = d.risk + ' risk'; b.className = 'badge ' + d.color;
  $('riskLabel').textContent = d.label + ' · ' + d.model;
  $('pdf').href = `/report/${d.id}.pdf`;
  $('consensus').innerHTML = Object.entries(d.consensus).map(([n,p]) => `<span class="chip">${n}: <b>${(p*100).toFixed(0)}%</b></span>`).join('');
  // explanation bars
  const max = Math.max(...d.explanation.map(x => Math.abs(x.impact)), 0.01);
  $('explain').innerHTML = d.explanation.slice(0,8).map(x => {
    const w = Math.abs(x.impact)/max*50, up = x.impact > 0;
    return `<div class="bar-row"><span title="${x.value}">${x.label}<br><small style="color:var(--muted)">${x.value}</small></span>
      <div class="bar-track"><div class="bar ${up?'up':'down'}" style="width:${w}%"></div></div>
      <span style="color:${up?'#dc3545':'#198754'};font-weight:600">${x.impact>0?'+':''}${(x.impact*100).toFixed(1)}</span></div>`;
  }).join('');
  $('advice').innerHTML = d.advice.map(([t,m]) => `<div class="tip"><b>${t}:</b> ${m}</div>`).join('');
  buildWhatIf(d.inputs);
  $('output').scrollIntoView({behavior:'smooth', block:'start'});
}

function setGauge(pct, color){
  const arc = $('arc'); arc.style.stroke = COLORS[color];
  arc.style.strokeDashoffset = 251 * (1 - pct/100);
  $('pct').textContent = pct.toFixed(0) + '%';
}

const WHATIF = [
  {f:'chol', label:'Cholesterol', min:120, max:400, step:5},
  {f:'trestbps', label:'Blood pressure', min:90, max:200, step:2},
  {f:'thalach', label:'Max heart rate', min:70, max:210, step:2},
  {f:'oldpeak', label:'ST depression', min:0, max:6, step:0.1},
];
function buildWhatIf(inp){
  $('whatif').innerHTML = WHATIF.map(w => `<div class="slider-row"><span>${w.label}</span>
    <input type="range" data-f="${w.f}" min="${w.min}" max="${w.max}" step="${w.step}" value="${Math.min(Math.max(inp[w.f],w.min),w.max)}">
    <b id="wv_${w.f}">${inp[w.f]}</b></div>`).join('') +
    `<div class="row" style="margin-top:10px">New estimated risk: <b id="wiRisk" style="font-size:1.2rem">–</b> <span class="badge" id="wiBadge"></span></div>`;
  $('whatif').querySelectorAll('input').forEach(s => s.oninput = () => {
    $('wv_'+s.dataset.f).textContent = s.value; clearTimeout(whatifTimer); whatifTimer = setTimeout(runWhatIf, 120);
  });
}
async function runWhatIf(){
  const o = {...lastInputs, model: form.elements.model.value};
  $('whatif').querySelectorAll('input').forEach(s => o[s.dataset.f] = s.value);
  const r = await fetch('/whatif', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(o)});
  const d = await r.json(); if (!d.ok) return;
  $('wiRisk').textContent = d.percent + '%'; const b = $('wiBadge'); b.textContent = d.risk; b.className = 'badge ' + d.color;
  setGauge(d.percent, d.color);
}
