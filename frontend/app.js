const API = window.CETSS_API || "";           // same-origin by default
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x) => (x * 100).toFixed(0) + "%";

// ---------- upload & analyze ----------
const drop = $("#drop"), fileInput = $("#file"), results = $("#results"), statusEl = $("#status");

["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) analyze(e.dataTransfer.files[0]); });
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); } });
fileInput.addEventListener("change", () => fileInput.files[0] && analyze(fileInput.files[0]));

async function analyze(file) {
  if (!/^image\/(png|jpe?g)$/.test(file.type)) {
    results.innerHTML = `<p class="muted">Please choose a PNG or JPEG image.</p>`;
    return;
  }
  const prev = $("#preview");
  prev.src = URL.createObjectURL(file);
  prev.hidden = false;
  results.innerHTML = `<p class="muted">Analyzing…</p>`;
  statusEl.textContent = "";

  const body = new FormData();
  body.append("file", file);
  try {
    const res = await fetch(`${API}/api/analyze`, { method: "POST", body });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Request failed (${res.status})`);
    }
    render(await res.json());
  } catch (e) {
    results.innerHTML = `<p class="muted">Could not analyze image: ${esc(e.message)}</p>`;
  }
}

function render(r) {
  const cls = r.label === "Positive" ? "pos" : r.label === "Negative" ? "neg" : "unc";
  const c = r.confidence;
  const conf = c
    ? `<dt>Confidence</dt><dd>p = ${c.p.toFixed(2)} · ${pct(c.level)} interval [${c.interval[0].toFixed(2)}, ${c.interval[1].toFixed(2)}]
         <div class="meter"><i style="left:${c.interval[0] * 100}%;width:${(c.interval[1] - c.interval[0]) * 100}%"></i><b style="left:${c.p * 100}%"></b></div></dd>`
    : `<dt>Confidence</dt><dd>—</dd>`;
  results.innerHTML = `
    ${r.demo_mode ? `<div class="banner">Demo mode — outputs come from a placeholder heuristic, not a trained clinical model.</div>` : ""}
    <div class="row"><span class="tag ${cls}">${esc(r.label)}</span><strong>${esc(r.finding ?? "")}</strong></div>
    <dl class="kv">
      ${conf}
      <dt>Image quality</dt><dd>${r.quality.score.toFixed(2)} · ${esc(r.quality.status)}</dd>
    </dl>
    <strong>Evidence</strong>
    <ul class="ev">${r.evidence.map((e) => `<li>${esc(e)}</li>`).join("")}</ul>
    ${r.heatmap ? `<img class="heat" src="${r.heatmap}" alt="Attention overlay">` : ""}
    <div class="triage"><span class="tag ${r.triage.action}">${esc(r.triage.action)}</span> ${esc(r.triage.label.replace(/^\w+ - /, ""))}</div>
    <p class="muted" style="font-size:13px;margin-top:12px">${esc(r.disclaimer)}</p>`;
}

// ---------- case walkthrough ----------
async function loadExamples() {
  try {
    const cases = await (await fetch(`${API}/api/examples`)).json();
    const tabs = $("#tabs");
    tabs.innerHTML = cases.map((c, i) => `<button role="tab" data-i="${i}" aria-selected="${i === 0}">${esc(c.title)}</button>`).join("");
    const show = (i) => {
      const c = cases[i];
      tabs.querySelectorAll("button").forEach((b, j) => b.setAttribute("aria-selected", j === i));
      const action = c.triage.split(" ")[0];
      $("#case").innerHTML = `
        <dl class="kv">
          <dt>Finding</dt><dd>${esc(c.finding)}</dd>
          <dt>Confidence</dt><dd>${c.p == null ? "— (abstained)" : `p = ${c.p.toFixed(2)} · 90% interval [${c.interval[0].toFixed(2)}, ${c.interval[1].toFixed(2)}]`}</dd>
          <dt>Image quality</dt><dd>${c.quality.toFixed(2)} · ${esc(c.quality_status)}</dd>
        </dl>
        <strong>Evidence</strong>
        <ul class="ev">${c.evidence.map((e) => `<li>${esc(e)}</li>`).join("")}</ul>
        <div class="triage"><span class="tag ${esc(action)}">${esc(action)}</span> ${esc(c.triage.replace(/^\w+ - /, ""))}</div>`;
    };
    tabs.addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) show(+b.dataset.i); });
    show(0);
  } catch { $("#case").innerHTML = `<p class="muted">Examples unavailable (is the backend running?).</p>`; }
}

// ---------- validation ----------
async function loadValidation() {
  try {
    const v = await (await fetch(`${API}/api/validation`)).json();
    $("#val-note").textContent = v.note;
    $("#metrics").innerHTML = v.metrics.map((m) => `<div class="metric"><strong>${esc(m.value)}</strong><span>${esc(m.key)}</span></div>`).join("");
  } catch { $("#val-note").textContent = "Validation data unavailable."; }
}

loadExamples();
loadValidation();
