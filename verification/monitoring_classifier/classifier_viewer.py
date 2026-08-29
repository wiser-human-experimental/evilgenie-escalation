"""Local viewer for the two-pass monitoring classifier: for each sample,
shows the tagged corpus Pass 1 actually saw, Pass 1's extraction, and
both Pass 2 assessments (real match against the sample's own problem,
and the decoy match against a different-category problem) side by side
-- built to make the arc191_c/arc185_c decoy-contamination pattern
(the paper discussion, 2026-08-05) directly inspectable
rather than inferred from aggregate stats.

Matches viewer.py's http.server pattern (no Flask dependency) rather
than duplicating a different stack. Runs on PORT (separate from
viewer.py's 8765) since both may run at once.

Run: python3 verification/monitoring_classifier/classifier_viewer.py
"""
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'verification' / 'disclosure_analysis'))

import zipfile_zstd as zz
from viewer import build_steps, relabel_epochs_by_chronology, PROBLEM_DEFECTS
from build_corpus import tagged_corpus

PORT = 8766

PROBLEM_ORDER = ['arc185_c', 'arc188_c', 'abc385_f', 'abc392_f', 'abc396_e',
                  'abc397_d', 'arc190_a', 'arc191_c', 'arc195_c']


# ── Data loading / merging ──────────────────────────────────────────────────

def load_rows():
    pass1 = json.load(open(HERE / 'pass1_results.json'))
    pass2_manifest = json.load(open(HERE / 'pass2_manifest.json'))
    pass2 = {}
    for line in open(HERE / 'pass2_output.jsonl'):
        rec = json.loads(line)
        pass2[rec['custom_id']] = json.loads(rec['response']['body']['choices'][0]['message']['content'])

    # index pass2 real/decoy results by their source pass1 custom_id
    real_by_source, decoy_by_source = {}, {}
    for cid, m in pass2_manifest.items():
        target = real_by_source if m['kind'] == 'real' else decoy_by_source
        target[m['source_cid']] = {**pass2[cid], 'decoy_problem_id': m.get('decoy_problem_id')}

    rows = []
    for cid, r in pass1.items():
        rows.append({
            'cid': cid,
            'key': r['key'],
            'model': r['model'],
            'cond': r['cond'],
            'problem_id': r['problem_id'],
            'population': r['population'],
            'deterministic_skip': r.get('deterministic_skip', False),
            'pass1': {
                'infrastructure_issue_noted': r['infrastructure_issue_noted'],
                'source': r['source'],
                'evidence_quotes': r['evidence_quotes'],
                'flaw_description': r['flaw_description'],
                'behavioral_consequence': r['behavioral_consequence'],
                'behavioral_detail': r['behavioral_detail'],
                'temporal_position': r['temporal_position'],
                'confidence': r['confidence'],
            },
            'pass2_real': real_by_source.get(cid),
            'pass2_decoy': decoy_by_source.get(cid),
            'own_defect_text': PROBLEM_DEFECTS.get(f"lcb_{r['problem_id']}", ''),
            'decoy_defect_text': (
                PROBLEM_DEFECTS.get(f"lcb_{decoy_by_source[cid]['decoy_problem_id']}", '')
                if cid in decoy_by_source else None
            ),
        })
    rows.sort(key=lambda r: (
        PROBLEM_ORDER.index(r['problem_id']) if r['problem_id'] in PROBLEM_ORDER else 99,
        r['population'], r['model'], r['cond'],
    ))
    return rows


ROWS = load_rows()
ROWS_BY_CID = {r['cid']: r for r in ROWS}
print(f"Loaded {len(ROWS)} Pass 1 rows "
      f"({sum(1 for r in ROWS if r['pass1']['infrastructure_issue_noted'])} noted, "
      f"{sum(1 for r in ROWS if r['pass2_real'])} with Pass 2 real, "
      f"{sum(1 for r in ROWS if r['pass2_decoy'])} with Pass 2 decoy)")

_corpus_cache = {}


def get_corpus(key):
    if key in _corpus_cache:
        return _corpus_cache[key]
    fname, sample_path = key.split('::', 1)
    import glob
    matches = glob.glob(str(REPO / 'results' / '**' / fname), recursive=True)
    if not matches:
        return f"[could not locate eval file {fname}]"
    with zz.ZipFile(matches[0]) as z:
        s = json.loads(z.read(sample_path))
    relabel_epochs_by_chronology([s])
    steps = build_steps(s.get('messages', []))
    corpus = tagged_corpus(steps)
    _corpus_cache[key] = corpus
    return corpus


# ── HTTP server ──────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        if parsed.path == "/api/rows":
            # omit corpus (fetched lazily per-row) to keep this payload light
            self._send(json.dumps(ROWS).encode(), "application/json")
        elif parsed.path == "/api/corpus":
            cid = (qs.get("cid") or [""])[0]
            row = ROWS_BY_CID.get(cid)
            if not row:
                self.send_response(404); self.end_headers(); return
            corpus = get_corpus(row['key'])
            self._send(json.dumps({'corpus': corpus}).encode(), "application/json")
        elif parsed.path in ("/", "/index.html"):
            self._send(HTML.encode(), "text/html; charset=utf-8")
        else:
            self.send_response(404)
            self.end_headers()


# ── Frontend ─────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Monitoring Classifier Viewer</title>
<style>
  :root {
    --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --bg:#f8fafc; --panel:#fff;
    --accent:#4f46e5; --good:#16a34a; --bad:#dc2626; --warn:#ea580c;
    --code-bg:#0f172a; --code-fg:#e2e8f0; --reason-bg:#1e293b; --comment-bg:#312e1e;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         font-size:13px; color:var(--ink); background:var(--bg); }
  header { background:#0f172a; color:#fff; padding:14px 18px; display:flex; align-items:center; gap:16px; }
  header .title { font-weight:600; font-size:15px; }
  header .stats { color:#94a3b8; font-size:12px; }
  .layout { display:grid; grid-template-columns:1fr; }
  .toolbar { display:flex; gap:10px; padding:10px 18px; background:var(--panel); border-bottom:1px solid var(--line);
             flex-wrap:wrap; align-items:center; }
  .toolbar select, .toolbar input[type=text] { padding:5px 8px; border:1px solid var(--line); border-radius:6px; font-size:12.5px; }
  .toolbar label { display:flex; align-items:center; gap:5px; font-size:12.5px; color:var(--muted); }
  .count { margin-left:auto; color:var(--muted); font-size:12px; }
  table { width:100%; border-collapse:collapse; background:var(--panel); }
  th, td { padding:7px 10px; text-align:left; border-bottom:1px solid var(--line); font-size:12.5px; white-space:nowrap; }
  th { background:#f1f5f9; position:sticky; top:0; font-weight:600; color:var(--muted); font-size:11.5px;
       text-transform:uppercase; letter-spacing:.03em; }
  tr:hover { background:#eef2ff; cursor:pointer; }
  .pill { display:inline-block; padding:1px 8px; border-radius:10px; font-size:11px; font-weight:600; }
  .yes { background:#dcfce7; color:var(--good); }
  .no { background:#f1f5f9; color:var(--muted); }
  .decoy-hit { background:#fee2e2; color:var(--bad); }
  .na { color:#cbd5e1; }
  tr.flagged { background:#fff7ed; }
  tr.flagged:hover { background:#fed7aa; }

  /* modal detail view */
  .overlay { display:none; position:fixed; inset:0; background:rgba(15,23,42,.6); z-index:50; }
  .overlay.open { display:flex; align-items:flex-start; justify-content:center; padding:24px; overflow:auto; }
  .modal { background:var(--panel); border-radius:10px; max-width:1100px; width:100%; max-height:calc(100vh - 48px);
           overflow:auto; box-shadow:0 20px 60px rgba(0,0,0,.3); }
  .modal-head { display:flex; justify-content:space-between; align-items:center; padding:14px 20px;
                border-bottom:1px solid var(--line); position:sticky; top:0; background:var(--panel); z-index:2; }
  .modal-head h2 { font-size:15px; }
  .modal-head .sub { color:var(--muted); font-size:12px; margin-top:2px; }
  .close-btn { background:none; border:none; font-size:22px; cursor:pointer; color:var(--muted); line-height:1; }
  .modal-body { padding:18px 20px; display:grid; gap:16px; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  .card { border:1px solid var(--line); border-radius:8px; overflow:hidden; }
  .card-head { padding:8px 12px; background:#f8fafc; font-weight:600; font-size:12px; border-bottom:1px solid var(--line);
               display:flex; justify-content:space-between; align-items:center; }
  .card-body { padding:12px; font-size:12.5px; line-height:1.5; }
  .card-body .field { margin-bottom:8px; }
  .card-body .field b { display:block; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.03em; margin-bottom:2px; }
  .corpus-box { background:var(--code-bg); color:var(--code-fg); border-radius:8px; padding:12px; font-family:ui-monospace,Menlo,monospace;
                font-size:11.5px; line-height:1.6; max-height:420px; overflow:auto; white-space:pre-wrap; }
  .corpus-box .tag-reasoning { display:block; background:var(--reason-bg); padding:6px 8px; border-radius:5px; margin:8px 0 2px;
                border-left:3px solid #60a5fa; }
  .corpus-box .tag-comment { display:block; background:var(--comment-bg); padding:6px 8px; border-radius:5px; margin:8px 0 2px;
                border-left:3px solid #fbbf24; }
  .corpus-box mark { background:#facc15; color:#000; border-radius:2px; padding:0 2px; }
  .loading { color:var(--muted); padding:20px; text-align:center; }
  .card.real .card-head { color:var(--good); }
  .card.decoy .card-head { color:var(--warn); }
  .badge { padding:2px 7px; border-radius:5px; font-size:10.5px; font-weight:700; }
  .badge.hit { background:#fee2e2; color:var(--bad); }
  .badge.miss { background:#dcfce7; color:var(--good); }
  .defect-text { background:#fffbeb; border:1px solid #fde68a; border-radius:6px; padding:8px 10px; font-size:12px; margin-bottom:8px; }
</style>
</head>
<body>
<header>
  <div class="title">Monitoring Classifier Viewer</div>
  <div class="stats" id="headerStats"></div>
</header>
<div class="toolbar">
  <select id="fProblem"><option value="">All problems</option></select>
  <select id="fPopulation">
    <option value="">All populations</option>
    <option value="main">main</option>
    <option value="ecological_fp">ecological_fp</option>
  </select>
  <select id="fPass1"><option value="">Pass 1: any</option><option value="true">noted=true</option><option value="false">noted=false</option></select>
  <select id="fPass2Real"><option value="">Pass 2 real: any</option><option value="true">detected=true</option><option value="false">detected=false</option><option value="none">no Pass 2</option></select>
  <select id="fPass2Decoy"><option value="">Pass 2 decoy: any</option><option value="true">HIT (contamination)</option><option value="false">correctly rejected</option></select>
  <select id="fModel"><option value="">All models</option></select>
  <input type="text" id="fSearch" placeholder="search evidence / rationale…">
  <span class="count" id="rowCount"></span>
</div>
<table>
  <thead>
    <tr>
      <th>Problem</th><th>Model</th><th>Cond</th><th>Pop</th>
      <th>Pass1 noted</th><th>Source</th>
      <th>Pass2 real</th><th>Pass2 decoy (vs)</th><th>Decoy hit?</th>
    </tr>
  </thead>
  <tbody id="tbody"></tbody>
</table>

<div class="overlay" id="overlay">
  <div class="modal">
    <div class="modal-head">
      <div>
        <h2 id="mTitle"></h2>
        <div class="sub" id="mSub"></div>
      </div>
      <button class="close-btn" onclick="closeModal()">&times;</button>
    </div>
    <div class="modal-body" id="mBody"></div>
  </div>
</div>

<script>
let ROWS = [];

function esc(s) { return (s===null||s===undefined) ? '' : String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

function pill(val, extraClass) {
  if (val === null || val === undefined) return '<span class="na">—</span>';
  const cls = val ? ('yes ' + (extraClass||'')) : 'no';
  return `<span class="pill ${cls}">${val ? 'true' : 'false'}</span>`;
}

async function load() {
  const res = await fetch('/api/rows');
  ROWS = await res.json();

  const problems = [...new Set(ROWS.map(r => r.problem_id))];
  const models = [...new Set(ROWS.map(r => r.model))].sort();
  document.getElementById('fProblem').innerHTML += problems.map(p => `<option value="${p}">${p}</option>`).join('');
  document.getElementById('fModel').innerHTML += models.map(m => `<option value="${esc(m)}">${esc(m)}</option>`).join('');

  const noted = ROWS.filter(r => r.pass1.infrastructure_issue_noted).length;
  const decoyHits = ROWS.filter(r => r.pass2_decoy && r.pass2_decoy.flaw_detected).length;
  const decoyTotal = ROWS.filter(r => r.pass2_decoy).length;
  document.getElementById('headerStats').textContent =
    `${ROWS.length} samples · ${noted} Pass1 hits · decoy contamination ${decoyHits}/${decoyTotal} (${(100*decoyHits/decoyTotal).toFixed(1)}%)`;

  ['fProblem','fPopulation','fPass1','fPass2Real','fPass2Decoy','fModel','fSearch'].forEach(id =>
    document.getElementById(id).addEventListener('input', render));
  render();
}

function render() {
  const fProblem = document.getElementById('fProblem').value;
  const fPopulation = document.getElementById('fPopulation').value;
  const fPass1 = document.getElementById('fPass1').value;
  const fPass2Real = document.getElementById('fPass2Real').value;
  const fPass2Decoy = document.getElementById('fPass2Decoy').value;
  const fModel = document.getElementById('fModel').value;
  const fSearch = document.getElementById('fSearch').value.toLowerCase();

  let rows = ROWS.filter(r => {
    if (fProblem && r.problem_id !== fProblem) return false;
    if (fPopulation && r.population !== fPopulation) return false;
    if (fPass1 && String(r.pass1.infrastructure_issue_noted) !== fPass1) return false;
    if (fPass2Real === 'none' && r.pass2_real) return false;
    if (fPass2Real === 'true' && !(r.pass2_real && r.pass2_real.flaw_detected)) return false;
    if (fPass2Real === 'false' && !(r.pass2_real && !r.pass2_real.flaw_detected)) return false;
    if (fPass2Decoy === 'true' && !(r.pass2_decoy && r.pass2_decoy.flaw_detected)) return false;
    if (fPass2Decoy === 'false' && !(r.pass2_decoy && !r.pass2_decoy.flaw_detected)) return false;
    if (fModel && r.model !== fModel) return false;
    if (fSearch) {
      const hay = JSON.stringify([r.pass1, r.pass2_real, r.pass2_decoy]).toLowerCase();
      if (!hay.includes(fSearch)) return false;
    }
    return true;
  });

  document.getElementById('rowCount').textContent = `${rows.length} rows`;
  document.getElementById('tbody').innerHTML = rows.map(r => {
    const decoyHit = r.pass2_decoy && r.pass2_decoy.flaw_detected;
    return `<tr class="${decoyHit ? 'flagged' : ''}" onclick="openModal('${r.cid.replace(/'/g,"\\'")}')">
      <td>${esc(r.problem_id)}</td>
      <td>${esc(r.model)}</td>
      <td>${esc(r.cond)}</td>
      <td>${esc(r.population)}</td>
      <td>${pill(r.pass1.infrastructure_issue_noted)}</td>
      <td>${esc(r.pass1.source)}</td>
      <td>${r.pass2_real ? pill(r.pass2_real.flaw_detected) : '<span class="na">—</span>'}</td>
      <td>${r.pass2_decoy ? esc(r.pass2_decoy.decoy_problem_id) : '<span class="na">—</span>'}</td>
      <td>${r.pass2_decoy ? pill(r.pass2_decoy.flaw_detected, 'decoy-hit') : '<span class="na">—</span>'}</td>
    </tr>`;
  }).join('');
}

function fieldRow(label, value) {
  return `<div class="field"><b>${esc(label)}</b>${value === null || value === undefined || value === '' ? '<span class="na">null</span>' : esc(value)}</div>`;
}

function pass2Card(title, cls, data, defectText) {
  if (!data) return `<div class="card ${cls}"><div class="card-head">${title}</div><div class="card-body"><span class="na">No Pass 2 request (Pass 1 found nothing here)</span></div></div>`;
  const badge = data.flaw_detected ? '<span class="badge hit">flaw_detected: true</span>' : '<span class="badge miss">flaw_detected: false</span>';
  return `<div class="card ${cls}">
    <div class="card-head"><span>${title}${data.decoy_problem_id ? ' vs ' + esc(data.decoy_problem_id) : ''}</span>${badge}</div>
    <div class="card-body">
      <div class="defect-text"><b style="display:block;font-size:10.5px;text-transform:uppercase;color:#92400e;margin-bottom:3px;">Confirmed defect being checked against</b>${esc(defectText)}</div>
      ${fieldRow('understanding_depth', data.understanding_depth)}
      ${fieldRow('mapping_confidence', data.mapping_confidence)}
      ${fieldRow('mapping_rationale', data.mapping_rationale)}
    </div>
  </div>`;
}

function highlightCorpus(corpus, quotes) {
  let html = esc(corpus);
  (quotes || []).forEach(q => {
    if (!q) return;
    const e = esc(q);
    if (html.includes(e)) html = html.split(e).join('<mark>' + e + '</mark>');
  });
  // tag lines
  html = html.replace(/\[STEP (\d+) \| REASONING\]/g, '<span class="tag-reasoning">STEP $1 · REASONING</span>');
  html = html.replace(/\[STEP (\d+) \| CODE COMMENT\]/g, '<span class="tag-comment">STEP $1 · CODE COMMENT</span>');
  return html;
}

async function openModal(cid) {
  const r = ROWS.find(x => x.cid === cid);
  if (!r) return;
  document.getElementById('mTitle').textContent = `${r.problem_id} — ${r.model} — Condition ${r.cond}`;
  document.getElementById('mSub').textContent = r.key;

  const fd = r.pass1.flaw_description || {};
  const pass1Card = `<div class="card">
    <div class="card-head"><span>Pass 1 extraction</span>${r.pass1.infrastructure_issue_noted ? '<span class="badge hit">infrastructure_issue_noted: true</span>' : '<span class="badge miss">infrastructure_issue_noted: false</span>'}</div>
    <div class="card-body">
      ${fieldRow('source', r.pass1.source)}
      ${fieldRow('what_is_wrong', fd.what_is_wrong)}
      ${fieldRow('why_it_is_wrong', fd.why_it_is_wrong)}
      ${fieldRow('what_should_happen', fd.what_should_happen)}
      ${fieldRow('scope', fd.scope)}
      ${fieldRow('specific_values', fd.specific_values)}
      ${fieldRow('behavioral_consequence', r.pass1.behavioral_consequence)}
      ${fieldRow('behavioral_detail', r.pass1.behavioral_detail)}
      ${fieldRow('temporal_position', r.pass1.temporal_position)}
      ${fieldRow('confidence', r.pass1.confidence)}
      ${fieldRow('evidence_quotes', (r.pass1.evidence_quotes||[]).join(' | '))}
    </div>
  </div>`;

  const pass2Grid = `<div class="grid2">
    ${pass2Card('Pass 2 — REAL match', 'real', r.pass2_real, r.own_defect_text)}
    ${pass2Card('Pass 2 — DECOY', 'decoy', r.pass2_decoy, r.decoy_defect_text)}
  </div>`;

  document.getElementById('mBody').innerHTML = `
    <div class="card">
      <div class="card-head">Tagged monitoring corpus (what Pass 1 saw)</div>
      <div class="card-body"><div class="corpus-box loading">Loading…</div></div>
    </div>
    ${pass1Card}
    ${pass2Grid}
  `;
  document.getElementById('overlay').classList.add('open');

  const res = await fetch('/api/corpus?cid=' + encodeURIComponent(cid));
  const { corpus } = await res.json();
  document.querySelector('.corpus-box').outerHTML =
    `<div class="corpus-box">${corpus ? highlightCorpus(corpus, r.pass1.evidence_quotes) : '<span class="na">empty corpus</span>'}</div>`;
}

function closeModal() { document.getElementById('overlay').classList.remove('open'); }
document.getElementById('overlay').addEventListener('click', e => { if (e.target.id === 'overlay') closeModal(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

load();
</script>
</body>
</html>"""

if __name__ == "__main__":
    url = f"http://localhost:{PORT}"
    print(f"Serving at {url}  (fully local)")
    print("Press Ctrl+C to stop.\n")
    threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        HTTPServer(("", PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
