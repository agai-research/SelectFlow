/* ============================================================================
   SelectFlow portal — service layers (no UI code here).
   Store / persistence · Simulation utilities · Preparation (tool cards, model
   training, certification) · Interpretation · Derivation · Governance ·
   Composition · Repair · History.  Each service is a plain object so that a
   real backend (MCP servers, LLM API, policy engine) can replace it later.
   ========================================================================== */
const esc = s => String(s == null ? "" : s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const clone = o => JSON.parse(JSON.stringify(o));
const uid = p => p + "-" + Math.random().toString(36).slice(2, 8);
const fmt = (x, d = 2) => (x == null || isNaN(x)) ? "–" : Number(x).toFixed(d);
const pct = x => (x == null || isNaN(x)) ? "–" : Math.round(x * 100) + "%";
const sum = a => a.reduce((s, x) => s + x, 0);
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

/* ------------------------------------------------------------------ store */
const ENT = ["organizations", "servers", "tools", "descriptors", "hosts", "sharingBases"];
const Store = {
  state: null, base: null, listeners: [], mode: "local", dbRef: null, saveTimer: null, status: "Saved in this browser",
  KEY: "selectflow-portal-v10",
  diff() {
    const st = this.state; const compact = r => { if (!r.composition) return r; const c = clone(r); Object.values(c.composition.perTask || {}).forEach(list => list.forEach(x => { x.nFeasible = (x.feasible || []).length; delete x.feasible; delete x.flowIssues; })); c.composition.violatingPlans = []; return c; };
    const out = { version: SEED_VERSION, updatedAt: st.updatedAt, config: st.config, runs: st.runs.slice(-25).map(compact), events: st.events.slice(-400), prepOps: st.prepOps.slice(-60), counters: st.counters, traces: {}, connectors: st.connectors, ent: {} };
    for (const k of ENT) { const bm = new Map(this.base[k].map(e => [e.id, JSON.stringify(e)])); const changed = st[k].filter(e => bm.get(e.id) !== JSON.stringify(e)); const ids = new Set(st[k].map(e => e.id)); out.ent[k] = { changed, deleted: this.base[k].filter(e => !ids.has(e.id)).map(e => e.id) }; }
    return out;
  },
  apply(d) {
    const st = seedState();
    if (!d || d.version !== SEED_VERSION) return st;
    for (const k of ENT) { const e = d.ent && d.ent[k]; if (!e) continue; const del = new Set(e.deleted); st[k] = st[k].filter(x => !del.has(x.id)); for (const c of e.changed) { const i = st[k].findIndex(x => x.id === c.id); if (i >= 0) st[k][i] = c; else st[k].push(c); } }
    Object.assign(st, { config: d.config || st.config, runs: d.runs || [], events: d.events || [], prepOps: d.prepOps || [], counters: d.counters || st.counters, connectors: d.connectors || [], updatedAt: d.updatedAt || st.updatedAt });
    return st;
  },
  init() {
    this.base = seedState(); let d = null;
    try { const raw = localStorage.getItem(this.KEY); if (raw) d = JSON.parse(raw); } catch (e) { d = null; }
    this.state = d ? this.apply(d) : seedState();
    if (!d) this.log("seed", "Registry federation loaded (Greece, Saudi Arabia)");
    this.connectDb();
  },
  async connectDb() {
    try {
      if (!window.claude || typeof window.claude.use !== "function") return;
      const [db, user] = await Promise.all([window.claude.use("db"), window.claude.use("user")]);
      if (!db || !user) return;
      const id = await user.id(); if (!id) return;
      this.dbRef = db.doc("data/users/" + id + "/portal-v10");
      const snap = await Promise.race([this.dbRef.get(), new Promise(r => setTimeout(() => r(null), 10000))]);
      if (snap && snap.exists) { const d = JSON.parse(snap.data().payload || "null"); if (d && d.version === SEED_VERSION && (!this.state.updatedAt || d.updatedAt > this.state.updatedAt)) { this.state = this.apply(d); this.persistLocal(); } }
      this.mode = "db"; this.status = "Saved to your account"; this.emit();
    } catch (e) { this.status = "Saved in this browser"; this.emit(); }
  },
  persistLocal() { try { localStorage.setItem(this.KEY, JSON.stringify(this.diff())); } catch (e) { } },
  scheduleSave() {
    clearTimeout(this.saveTimer);
    this.saveTimer = setTimeout(async () => {
      this.persistLocal(); if (!this.dbRef) return;
      try { let d = this.diff(); let payload = JSON.stringify(d); if (payload.length > 240000) { d.events = d.events.slice(-120); d.runs = d.runs.slice(-8); payload = JSON.stringify(d); } await this.dbRef.set({ payload, updatedAt: this.state.updatedAt }); }
      catch (e) { this.status = "Saved in this browser (account save failed)"; this.emit(); }
    }, 900);
  },
  commit(mutator, quiet) { mutator(this.state); this.state.updatedAt = new Date().toISOString(); Governance.reset(); this.scheduleSave(); if (!quiet) this.emit(); },
  on(fn) { this.listeners.push(fn); },
  emit() { this.listeners.forEach(f => f()); },
  log(type, summary, runId, details) { this.state.events.push({ id: uid("ev"), t: new Date().toISOString(), type, summary, runId: runId || null, details: details || null }); if (this.state.events.length > 1500) this.state.events = this.state.events.slice(-1500); },
  reset() { this.state = seedState(); Governance.reset(); this.log("seed", "Registry federation reloaded"); this.state.updatedAt = new Date().toISOString(); this.scheduleSave(); this.emit(); },
};
const S = () => Store.state;
const byId = (arr, id) => arr.find(x => x.id === id);
const orgName = id => id === "REQ" ? "Requester" : id === "EXT" ? "Outside the federation" : (byId(S().organizations, id) || {}).name || id;
const artName = id => (byId(S().config.artefacts, id) || {}).name || id.replace(/_/g, " ");
const capById = id => byId(S().config.capabilities, id) || { id, name: id, statement: id, category: "" };
const artCats = id => (byId(S().config.artefacts, id) || { categories: ["BUS"] }).categories;
const serverOf = t => byId(S().servers, t.serverId);

/* ------------------------------------------------------------ text model */
const STOP = new Set("a an the of for to and or in on at by with from is are be as that this its it into your their any once after before then".split(" "));
function toks(s) { return (s || "").toLowerCase().replace(/[^a-z0-9 ]+/g, " ").split(/\s+/).filter(w => w.length > 2 && !STOP.has(w)).map(w => w.replace(/(ions|ion|ing|ed|es|s)$/, "")); }
function textSim(a, b) { const A = new Set(toks(a)), B = new Set(toks(b)); if (!A.size || !B.size) return 0; let i = 0; A.forEach(x => { if (B.has(x)) i++; }); return i / Math.sqrt(A.size * B.size); }
const ASSERTIVE = /\b(best|fastest|always|official|recommended|any)\b/i;

/* ----------------------------------------------------- preparation: cards */
const Preparation = {
  async generateCardsLive(toolIds) {
    const st = S(); const out = [];
    for (const id of toolIds) {
      const t = byId(st.tools, id); const vocab = this.vocabularyFor(t).slice(0, 60);
      const res = await LLM.json(`Map this tool to the capabilities it provides (usually one). Return {"capabilities": [ids]} using only listed ids.\nCapabilities:\n${vocab.map(c => c.id + ": " + c.statement).join("\n")}\nTool: ${t.name}. ${t.description}`);
      const ok = new Set(vocab.map(c => c.id)); const caps = {}; (res.capabilities || []).filter(x => ok.has(x)).forEach(x => caps[x] = 1);
      out.push({ id, caps });
    }
    Store.commit(s => { out.forEach(({ id, caps }) => { const t = byId(s.tools, id); t.card = { caps, inputs: t.inputArtefacts.slice(), outputs: t.outputArtefacts.slice(), tokens: t.tokens, qos: clone(t.qos), behaviour: clone(t.declared), generatedAt: new Date().toISOString(), status: "generated", provenance: "live", passes: 1, model: LLM.label() }; });
      s.prepOps.push({ id: uid("op"), t: new Date().toISOString(), kind: "Tool cards", detail: `${out.length} tool card(s) mapped by ${LLM.label()}` }); Store.log("prep-cards", `Mapped ${out.length} tool card(s) with ${LLM.label()}`); });
    return out;
  },
  vocabularyFor(tool) {
    const st = S(); const own = st.descriptors.filter(d => d.orgId === tool.orgId).flatMap(d => d.capabilities);
    const caps = tool.orgId === "org-portal" || !own.length ? st.config.capabilities.map(c => c.id) : [...new Set(own)];
    return caps.map(capById);
  },
  generateCards(toolIds) {
    const st = S(); const rng = mulberry32(st.config.simulation.seed + st.counters.op * 13);
    const results = [];
    for (const id of toolIds) {
      const t = byId(st.tools, id); if (!t) continue;
      const vocab = this.vocabularyFor(t); const passes = 5; const hits = {};
      for (let p = 0; p < passes; p++) {
        const scored = vocab.map(c => ({ c, s: textSim(t.name + " " + t.description, c.name + " " + c.statement) + (rng() - 0.5) * 0.18 })).sort((a, b) => b.s - a.s);
        const top = scored[0];
        if (top && top.s > 0.16) hits[top.c.id] = (hits[top.c.id] || 0) + 1;
        if (scored[1] && scored[1].s > 0.32) hits[scored[1].c.id] = (hits[scored[1].c.id] || 0) + 1;
      }
      const caps = Object.fromEntries(Object.entries(hits).map(([k, v]) => [k, +(v / passes).toFixed(2)]));
      t.card = { caps, inputs: t.inputArtefacts.slice(), outputs: t.outputArtefacts.slice(), tokens: t.tokens, qos: clone(t.qos), behaviour: clone(t.declared),
        generatedAt: new Date().toISOString(), status: "generated", provenance: "simulated", passes };
      results.push({ id, caps });
    }
    st.counters.op++;
    st.prepOps.push({ id: uid("op"), t: new Date().toISOString(), kind: "Tool cards", detail: `${results.length} tool card(s) generated (simulated mapping, 5 passes)`, items: results.map(r => r.id) });
    Store.log("prep-cards", `Generated ${results.length} tool card(s)`, null, { tools: toolIds });
    return results;
  },
};

/* --------------------------------------------- tool-selection model (MNL) */
const SelectionModel = {
  features(t, capId, desc, host) {
    const cap = capById(capId);
    return [1, textSim(t.name + " " + t.description, cap.name + " " + cap.statement + " " + (desc ? desc.name : "")), textSim(t.name, cap.name),
      ASSERTIVE.test(t.description) ? 1 : 0, desc && t.orgId === desc.orgId ? 1 : 0];
  },
  load(K) { const st = S(); const servers = new Set(); let L = 0; K.forEach(id => { const t = byId(st.tools, id); if (!t) return; L += t.tokens; servers.add(t.serverId); }); servers.forEach(s => { L += (byId(st.servers, s) || { preambleTokens: 0 }).preambleTokens; }); return L; },
  probs(K, capId, desc, host, beta, gamma) {
    const st = S(); const b = beta || host.model.beta; const g = gamma == null ? host.model.gamma : gamma;
    const a = {}; K.forEach(id => { const t = byId(st.tools, id); const f = this.features(t, capId, desc, host); a[id] = Math.exp(sum(f.map((x, i) => x * b[i]))); });
    const base = Math.max(0.05, 1 + g * this.load(K) / 1000); const Z = base + sum(Object.values(a));
    const p = {}; Object.keys(a).forEach(k => p[k] = a[k] / Z); return { p, none: base / Z };
  },
  /** Fit on the recorded calls of Claude Sonnet 5.5 (citizen-phrased steps, 120 capabilities), in this browser. */
  trainRecorded(hostId) {
    const st = S(); const host = byId(st.hosts, hostId); const tl = DATA.tools.map(t => t.id); const dl = DATA.descriptors.map(d => d.id);
    const rows = DATA.traces.map(([v, r, K, c]) => ({ d: byId(st.descriptors, dl[v]), r, K: K.map(i => tl[i]).filter(id => byId(st.tools, id)), c: c >= 0 ? tl[c] : null })).filter(x => x.d && x.K.length);
    rows.forEach(x => { x.F = x.K.map(id => this.features(byId(st.tools, id), x.r, x.d, host)); x.L = this.load(x.K) / 1000; x.ci = x.c ? x.K.indexOf(x.c) : -1; });
    const ll = (b, g) => { let l = 0; for (const x of rows) { const a = x.F.map(f => Math.exp(f.reduce((s, v, i) => s + v * b[i], 0))); const base = Math.max(0.05, 1 + g * x.L); const Z = base + a.reduce((s, v) => s + v, 0); l += Math.log(Math.max(1e-12, (x.ci >= 0 ? a[x.ci] : base) / Z)); } return l / rows.length; };
    let b = [0, 0, 0, 0, 0], g = 0;
    for (let it = 0; it < 90; it++) { const grad = [0, 0, 0, 0, 0]; for (const x of rows) { const a = x.F.map(f => Math.exp(f.reduce((s, v, i) => s + v * b[i], 0))); const base = Math.max(0.05, 1 + g * x.L); const Z = base + a.reduce((s, v) => s + v, 0); const fb = [0, 0, 0, 0, 0]; x.F.forEach((f, k) => f.forEach((v, i) => fb[i] += a[k] / Z * v)); const fc = x.ci >= 0 ? x.F[x.ci] : [0, 0, 0, 0, 0]; fc.forEach((v, i) => grad[i] += v - fb[i]); } b = b.map((v, i) => v + 0.15 * grad[i] / rows.length); }
    const gs = [-0.15, -0.1, -0.05, 0, 0.05].map(gg => ({ gg, l: ll(b, gg) })).sort((p, q) => q.l - p.l)[0];
    host.model = { ...host.model, beta: b.map(v => +v.toFixed(3)), gamma: gs.gg, mode: "fitted-portal", trainedAt: new Date().toISOString(), traces: rows.length, logLik: +gs.l.toFixed(3), source: `Fitted in this portal on ${rows.length} recorded calls of Claude Sonnet 5.5` };
    st.counters.op++; st.prepOps.push({ id: uid("op"), t: new Date().toISOString(), kind: "Model training", detail: `${host.name}: fitted on ${rows.length} recorded calls` });
    Store.log("prep-train", `Fitted the tool-selection model of ${host.name} on ${rows.length} recorded calls`, null, { host: hostId });
    return host.model;
  },
  /* Simulated training: synthetic traces drawn from a hidden parameter, then maximum likelihood by gradient ascent. */
  train(hostId, nTraces) {
    const st = S(); const host = byId(st.hosts, hostId); const rng = mulberry32(st.config.simulation.seed * 31 + st.counters.op);
    const hidden = host.model.beta.map((b, i) => b + (rng() - 0.5) * 0.6 * (i === 0 ? 1 : 1)); const hiddenG = host.model.gamma + (rng() - 0.5) * 0.04;
    const tasks = st.descriptors; const traces = [];
    for (let n = 0; n < nTraces; n++) {
      const d = tasks[Math.floor(rng() * tasks.length)]; const r = d.capabilities[Math.floor(rng() * d.capabilities.length)];
      const pool = st.tools.slice().sort(() => rng() - 0.5); const k = 2 + Math.floor(rng() * 4);
      const own = st.tools.filter(t => t.card && (t.card.caps[r] || 0) >= 0.4).map(t => t.id);
      const K = [...new Set(own.slice(0, 1).concat(pool.slice(0, k).map(t => t.id)))];
      const pr = this.probs(K, r, d, host, hidden, hiddenG); let u = rng(), choice = null;
      for (const [id, p] of Object.entries(pr.p)) { u -= p; if (u <= 0) { choice = id; break; } }
      traces.push({ K, cap: r, desc: d.id, choice });
    }
    let beta = host.model.beta.map(() => 0), gamma = 0; const lr = 0.08; let ll = 0;
    for (let it = 0; it < 120; it++) {
      const grad = beta.map(() => 0); ll = 0;
      for (const tr of traces) {
        const d = byId(tasks, tr.desc); const pr = this.probs(tr.K, tr.cap, d, host, beta, gamma);
        ll += Math.log(Math.max(1e-9, tr.choice ? pr.p[tr.choice] : pr.none));
        const fbar = beta.map(() => 0);
        tr.K.forEach(id => { const f = this.features(byId(st.tools, id), tr.cap, d, host); f.forEach((x, i) => fbar[i] += pr.p[id] * x); });
        const fc = tr.choice ? this.features(byId(st.tools, tr.choice), tr.cap, d, host) : beta.map(() => 0);
        fc.forEach((x, i) => grad[i] += x - fbar[i]);
      }
      beta = beta.map((b, i) => b + lr * grad[i] / traces.length);
    }
    const best = [-0.08, -0.04, 0, 0.04].map(g => { let l = 0; traces.forEach(tr => { const d = byId(tasks, tr.desc); const pr = this.probs(tr.K, tr.cap, d, host, beta, g); l += Math.log(Math.max(1e-9, tr.choice ? pr.p[tr.choice] : pr.none)); }); return { g, l }; }).sort((a, b) => b.l - a.l)[0];
    host.model = { ...host.model, beta: beta.map(x => +x.toFixed(3)), gamma: best.g, mode: "simulated-trained", trainedAt: new Date().toISOString(), traces: traces.length, logLik: +(best.l / traces.length).toFixed(3), hidden: hidden.map(x => +x.toFixed(3)) };
    st.traces[hostId] = traces.slice(-200);
    st.counters.op++;
    st.prepOps.push({ id: uid("op"), t: new Date().toISOString(), kind: "Model training", detail: `${host.name}: fitted on ${traces.length} synthetic traces (simulation)` });
    Store.log("prep-train", `Trained the tool-selection model of ${host.name} on ${traces.length} synthetic traces`, null, { host: hostId });
    return host.model;
  },
};

/* --------------------------------------------- behaviour certification */
const Certification = {
  score(t, rng) { let s = 0.55 + (t.declared.attested ? 0.25 : -0.15) + ((byId(S().organizations, t.orgId) || {}).category === "Third-party provider" ? -0.15 : 0.05) + (ASSERTIVE.test(t.description) ? -0.1 : 0) + (rng() - 0.5) * 0.2; return Math.min(0.99, Math.max(0.01, s)); },
  run(eps, auditFraction) {
    const st = S(); const rng = mulberry32(st.config.simulation.seed * 7 + st.counters.op);
    const tools = st.tools; tools.forEach(t => t.certification.score = +this.score(t, rng).toFixed(3));
    const shuffled = tools.slice().sort(() => rng() - 0.5); const nAudit = Math.max(2, Math.round(auditFraction * tools.length));
    const audited = new Set(shuffled.slice(0, nAudit).map(t => t.id));
    const unsafe = t => t.truth.forwards || t.truth.retains.some(c => !t.declared.retains.includes(c));
    const calib = tools.filter(t => audited.has(t.id) && unsafe(t)).map(t => t.certification.score);
    const test = tools.filter(t => !audited.has(t.id));
    const pv = test.map(t => ({ t, p: (1 + calib.filter(s => s >= t.certification.score).length) / (calib.length + 1) })).sort((a, b) => a.p - b.p);
    let k = 0; pv.forEach((x, i) => { if (x.p <= eps * (i + 1) / pv.length) k = i + 1; });
    const sel = new Set(pv.slice(0, k).map(x => x.t.id));
    const stamp = new Date().toISOString(); let falseCert = 0, nCert = 0;
    tools.forEach(t => {
      let status, basis;
      if (audited.has(t.id)) { status = unsafe(t) ? "rejected" : "certified"; basis = "audit"; }
      else if (sel.has(t.id)) { status = "certified"; basis = "conformal selection"; }
      else { status = "uncertified"; basis = "not selected"; }
      if (status === "certified") { nCert++; if (unsafe(t)) falseCert++; }
      const prev = t.certification;
      t.certification = { status, basis, score: prev.score, audited: audited.has(t.id), attested: !!t.declared.attested, eps, at: stamp,
        pvalue: (pv.find(x => x.t.id === t.id) || {}).p ?? null, history: (prev.history || []).concat([{ at: stamp, status, basis, eps }]).slice(-12) };
    });
    const out = { eps, auditFraction, audited: nAudit, calibrationUnsafe: calib.length, certified: nCert, simulatedFalseCertifications: falseCert, at: stamp };
    st.counters.op++;
    st.prepOps.push({ id: uid("op"), t: stamp, kind: "Certification", detail: `ε = ${eps}: ${nCert} certified, ${nAudit} audited (simulation)`, result: out });
    Store.log("prep-cert", `Certification run at ε = ${eps}: ${nCert} tools certified`, null, out);
    return out;
  },
};

/* ------------------------------------------------------- governance */
const Governance = {
  _idx: null, _wild: null, _memo: null, _comp: null,
  reset() { this._idx = null; this._memo = null; },
  build() {
    const st = S(); this._idx = new Map(); this._wild = []; this._memo = new Map(); this._comp = new Set();
    st.descriptors.forEach(d => d.elicited.forEach(c => this._comp.add(d.orgId + "|" + d.purpose + "|" + c)));
    for (const sb of st.sharingBases) for (const r of sb.rules) {
      if (r.purpose === "*" || r.origin === "*") { this._wild.push({ sb, r }); continue; }
      const k = r.purpose + "|" + r.category + "|" + r.origin + "|" + r.receiver; const cur = this._idx.get(k);
      if (!cur || r.effect === "deny") this._idx.set(k, { sb, r });
    }
  },
  /** Is a flow of `category` from `origin` to `receiver` for `purpose` allowed? Returns {allowed, basis, reason}. */
  check(purpose, category, origin, receiver) {
    if (!this._idx) this.build();
    const mk = purpose + "|" + category + "|" + origin + "|" + receiver; if (this._memo.has(mk)) return this._memo.get(mk);
    const res = this._check(purpose, category, origin, receiver); this._memo.set(mk, res); return res;
  },
  _check(purpose, category, origin, receiver) {
    if (origin === receiver && origin !== "REQ") return { allowed: true, basis: "Same organization", reason: "Data stay within the organization." };
    if (receiver === "EXT") return { allowed: false, basis: null, reason: "No sharing base allows data to leave the federation." };
    const own = this._comp.has(receiver + "|" + purpose + "|" + category);
    let deny = null, allow = null;
    const hit = this._idx.get(purpose + "|" + category + "|" + origin + "|" + receiver);
    if (hit) { if (hit.r.effect === "deny") deny = hit; else allow = hit; }
    for (const w of this._wild) { const r = w.r; if ((r.purpose === "*" || r.purpose === purpose) && r.category === category && (r.origin === "*" || r.origin === origin) && r.receiver === receiver) { if (r.effect === "deny") deny = deny || w; else allow = allow || w; } }
    if (deny) return { allowed: false, basis: deny.sb.name, reason: `Denied by “${deny.sb.name}”.` };
    if (origin === "REQ" && own) return { allowed: true, basis: "Competence", reason: `${orgName(receiver)} is competent for ${purpose} and collects ${category} data itself.` };
    if (allow) return { allowed: true, basis: allow.sb.name, reason: `Allowed by “${allow.sb.name}”${allow.r.retention ? " (retention allowed)" : ""}.` };
    return { allowed: false, basis: null, reason: `No sharing base allows ${category} data from ${orgName(origin)} to ${orgName(receiver)} for ${purpose}.` };
  },
  /** Categories a tool may return and whether it may forward data, as used by the composer. */
  behaviour(t) {
    const certified = t.certification.status === "certified";
    if (certified) return { retains: t.declared.retains, forwards: !!t.declared.forwards, basis: "certified" };
    const org = byId(S().organizations, t.orgId) || {};
    return { retains: S().config.dataCategories.map(c => c.id), forwards: org.category === "Third-party provider" || !!t.declared.forwards, basis: "uncertified: pessimistic" };
  },
  hostAdmissible(host, desc) {
    if (!host.supports.includes(desc.purpose)) return { ok: false, reason: `${host.name} is not configured for ${desc.purpose} tasks (model compatibility).` };
    for (const c of desc.elicited) { const g = this.check(desc.purpose, c, "REQ", host.operatorOrgId); if (!g.allowed) return { ok: false, reason: `Requester ${c} data may not reach ${orgName(host.operatorOrgId)}: ${g.reason}` }; }
    return { ok: true, reason: "Model compatible; requester data may reach the host operator." };
  },
  toolkitFlows(K, desc, host) {
    const st = S(); const issues = [];
    for (const id of K) {
      const t = byId(st.tools, id); const beh = this.behaviour(t);
      for (const c of desc.elicited.filter(c => t.inputCategories.includes(c))) {
        const g = this.check(desc.purpose, c, "REQ", t.orgId); if (!g.allowed) issues.push({ tool: id, flow: `${c}: requester → ${orgName(t.orgId)}`, reason: g.reason });
      }
      for (const c of beh.retains) {
        if (t.orgId === host.operatorOrgId) continue;
        const g = this.check(desc.purpose, c, t.orgId, host.operatorOrgId); if (!g.allowed) issues.push({ tool: id, flow: `${c}: ${orgName(t.orgId)} → ${orgName(host.operatorOrgId)} (${beh.basis})`, reason: g.reason });
      }
      if (beh.forwards) issues.push({ tool: id, flow: `forwarding: ${orgName(t.orgId)} → outside the federation (${beh.basis})`, reason: "Forwarding outside the federation is never allowed." });
    }
    return issues;
  },
};

/* ------------------------------------------------------- interpretation */
const Interpretation = {
  simulate(text) {
    const st = S(); const producible = new Set(st.descriptors.flatMap(d => d.delivers));
    const score = st.config.artefacts.filter(a => producible.has(a.id)).map(a => {
      const prods = st.descriptors.filter(d => d.delivers.includes(a.id));
      return { id: a.id, s: Math.max(textSim(text, a.name), ...prods.map(d => textSim(text, d.name + " " + d.description))) };
    }).sort((a, b) => b.s - a.s);
    const targets = score.slice(0, 1).map(x => x.id);
    return { targets, held: [], confidence: +(Math.min(0.95, 0.3 + (score[0] ? score[0].s : 0))).toFixed(2), source: "lexical" };
  },
  async live(text) {
    const st = S(); const producible = new Set(st.descriptors.flatMap(d => d.delivers));
    const ranked = st.config.artefacts.map(a => ({ a, s: textSim(text, a.name) })).sort((x, y) => y.s - x.s).slice(0, 40).map(x => x.a);
    const arts = ranked.map(a => `${a.id}: ${a.name}${producible.has(a.id) ? "" : " (held by applicants)"}`).join("\n");
    const out = await LLM.json(`A citizen writes to an administrative assistant. From the documents below, return {"targets": [ids the citizen wants to obtain], "held": [ids the citizen already holds]}. Use only listed ids.\nDocuments:\n${arts}\nRequest: ${text}`);
    const ok = new Set(st.config.artefacts.map(a => a.id));
    return { targets: (out.targets || []).filter(x => ok.has(x) && producible.has(x)), held: (out.held || []).filter(x => ok.has(x)), confidence: null, source: "live", model: LLM.label() };
  },
};

/* --------------------------------------------------------- derivation */
const Derivation = {
  /** Producers of an artefact; an alternative procedure is used only when its primary procedure is excluded. */
  producers(a, excluded) { const ex = excluded || []; return S().descriptors.filter(d => d.delivers.includes(a) && !ex.includes(d.id) && !(d.alternativeOf && !ex.includes(d.alternativeOf) && byId(S().descriptors, d.alternativeOf))); },
  retrieve(text, targets, excluded, n) {
    const st = S(); const pool = st.descriptors.filter(d => !excluded.includes(d.id));
    const scored = pool.map(d => ({ d, s: textSim(text, d.name + " " + d.description) + (d.delivers.some(a => targets.includes(a)) ? 1 : 0) })).sort((a, b) => b.s - a.s);
    const out = scored.slice(0, n).map(x => x.d.id);
    return { ids: out, scores: Object.fromEntries(scored.map(x => [x.d.id, +x.s.toFixed(2)])) };
  },
  alignment(produced, required) { if (produced === required) return 1; const A = (S().config.alignments || {})[produced + ">" + required]; return A == null ? 0 : A; },
  /** Deterministic critics: closure, alignment, companions, minimality. */
  critics(order, targets, held) {
    const st = S(); const avail = new Set(held); const supplier = {}; const rep = { closure: [], alignment: [], companions: [], minimality: [], handoffs: [] };
    for (const id of order) {
      const d = byId(st.descriptors, id);
      for (const a of d.requires) {
        if (avail.has(a)) { if (supplier[a]) { const c = this.alignment(a, a); rep.handoffs.push({ from: supplier[a], to: id, artefact: a, confidence: c }); if (c < st.config.params.tauChi) rep.alignment.push(`${artName(a)} from ${supplier[a]} to ${id}: confidence ${c}`); } }
        else rep.closure.push(`${id} needs ${artName(a)}, which is neither held nor produced earlier.`);
      }
      for (const a of d.delivers) { if (!supplier[a]) supplier[a] = id; avail.add(a); }   // first producer kept (forward-chaining rule)
      for (const c of d.companions) if (!order.includes(c)) rep.companions.push(`${id} requires its companion ${c}.`);
    }
    for (const a of targets) if (!avail.has(a)) rep.closure.push(`Target ${artName(a)} is not delivered.`);
    const needed = new Set(targets); for (let i = order.length - 1; i >= 0; i--) { const d = byId(st.descriptors, order[i]); const useful = d.delivers.some(a => needed.has(a)) || order.some(o => (byId(st.descriptors, o).companions || []).includes(d.id)); if (useful) d.requires.forEach(a => needed.add(a)); else rep.minimality.push(`${order[i]} delivers nothing that is needed.`); }
    rep.ok = !rep.closure.length && !rep.alignment.length && !rep.companions.length && !rep.minimality.length;
    return rep;
  },
  toposort(ids) {
    const st = S(); const out = []; const set = new Set(ids); const done = new Set(); const visiting = new Set();
    const visit = id => { if (done.has(id) || visiting.has(id)) return; visiting.add(id); const d = byId(st.descriptors, id); d.requires.forEach(a => ids.filter(o => o !== id && byId(st.descriptors, o).delivers.includes(a)).forEach(visit)); visiting.delete(id); done.add(id); out.push(id); };
    ids.forEach(visit); return out.filter(x => set.has(x));
  },
  /** Simulated constrained generator: proposes k candidates over the retrieved descriptors, with typical errors. */
  generate(retrieved, targets, held, k, rng) {
    const st = S(); const R = retrieved.map(id => byId(st.descriptors, id));
    const direct = R.filter(d => d.delivers.some(a => targets.includes(a))).map(d => d.id);
    const oneLevel = new Set(direct); direct.forEach(id => byId(st.descriptors, id).requires.forEach(a => { if (!held.includes(a)) R.filter(d => d.delivers.includes(a)).slice(0, 1).forEach(d => oneLevel.add(d.id)); }));
    const extra = R.filter(d => !oneLevel.has(d.id)).map(d => d.id);
    const cands = [direct, [...oneLevel], [...oneLevel].concat(extra.slice(0, 1))].slice(0, k);
    return cands.map(c => this.toposort([...new Set(c)]));
  },
  revise(order, rep, retrieved, held, targets) {
    const st = S(); let o = order.slice(); const notes = [];
    rep.closure.forEach(msg => { const m = msg.match(/needs (.+), which/) || msg.match(/Target (.+) is not/); if (!m) return; const a = st.config.artefacts.find(x => x.name === m[1]); const id = a ? a.id : null; if (!id) return; const p = retrieved.map(x => byId(st.descriptors, x)).find(d => d.delivers.includes(id)); if (p && !o.includes(p.id)) { o.push(p.id); notes.push(`added ${p.id} to supply ${artName(id)}`); } else if (!p) notes.push(`no retrieved descriptor supplies ${artName(id)}`); });
    rep.companions.forEach(msg => { const m = msg.match(/companion (.+)\.$/); if (m && retrieved.includes(m[1]) && !o.includes(m[1])) { o.push(m[1]); notes.push(`added companion ${m[1]}`); } else if (m) notes.push(`companion ${m[1]} was not retrieved`); });
    rep.minimality.forEach(msg => { const id = msg.split(" ")[0]; o = o.filter(x => x !== id); notes.push(`removed ${id}`); });
    return { order: this.toposort([...new Set(o)]), notes };
  },
  /** Forward chaining over all descriptors: complete whenever a closed workflow exists. */
  forwardChain(targets, held, excluded) {
    const st = S(); const steps = []; const need = new Set(); const chosen = new Set(); const unmet = [];
    const want = a => { if (held.includes(a) || need.has(a)) return; need.add(a); const p = this.producers(a, excluded); if (!p.length) { unmet.push(a); steps.push(`No descriptor delivers ${artName(a)}.`); return; } const d = p.sort((x, y) => x.requires.length - y.requires.length)[0]; if (!chosen.has(d.id)) { chosen.add(d.id); steps.push(`${artName(a)} ← ${d.id}`); d.requires.forEach(want); d.companions.forEach(c => { if (!chosen.has(c) && !excluded.includes(c)) { chosen.add(c); steps.push(`companion ${c} of ${d.id}`); byId(st.descriptors, c).requires.forEach(want); } }); } };
    targets.forEach(want);
    return { order: unmet.length ? null : this.toposort([...chosen]), steps, unmet };
  },
  retrieveFor(text, targets, excluded) { return this.retrieve(text, targets, excluded.concat(S().descriptors.filter(d => d.alternativeOf && !excluded.includes(d.alternativeOf)).map(d => d.id)), S().config.params.nRetrieve); },
  /** Live generation: the language model proposes k orders over the retrieved descriptors only (grammar = their ids). */
  async generateLive(retrieved, targets, held, k) {
    const st = S(); const list = retrieved.map(id => { const d = byId(st.descriptors, id); return `${id}: ${d.name} | requires: ${d.requires.join(", ") || "-"} | delivers: ${d.delivers.join(", ")}${d.companions.length ? " | companions: " + d.companions.join(", ") : ""}`; }).join("\n");
    const out = await LLM.json(`You compose administrative workflows. Using ONLY the procedures listed, propose ${k} different ordered workflows that deliver the target documents, given the documents already held. Each workflow is a list of procedure ids in execution order.\nTargets: ${targets.join(", ")}\nHeld: ${held.join(", ") || "none"}\nProcedures:\n${list}\nReturn {"workflows": [["id", ...], ...]}.`);
    const ok = new Set(retrieved);
    return (out.workflows || []).slice(0, k).map(w => (Array.isArray(w) ? w : []).filter(x => ok.has(x))).filter(w => w.length);
  },
  run(text, targets, held, excluded, liveCandidates) {
    const st = S(); const p = st.config.params; const rng = mulberry32(st.config.simulation.seed + text.length);
    const ret = this.retrieveFor(text, targets, excluded);
    const cands = (liveCandidates || this.generate(ret.ids, targets, held, p.kCandidates, rng)).map(order => {
      const history = []; let o = order, rep = this.critics(o, targets, held); history.push({ order: o, report: rep });
      for (let i = 0; i < p.nRevisions && !rep.ok; i++) { const rv = this.revise(o, rep, ret.ids, held, targets); o = rv.order; rep = this.critics(o, targets, held); history.push({ order: o, report: rep, notes: rv.notes }); }
      return { initial: order, order: o, accepted: rep.ok, history };
    });
    const fc = this.forwardChain(targets, held, excluded);
    const fcRep = fc.order ? this.critics(fc.order, targets, held) : null;
    const acc = cands.find(c => c.accepted);
    let final = null, source = null;
    if (acc) { final = acc.order; source = "generated candidate (accepted by the critics)"; }
    else if (fc.order && fcRep && fcRep.ok) { final = fc.order; source = "forward chaining"; }
    const finalRep = final ? this.critics(final, targets, held) : null;
    return { generator: liveCandidates ? "live" : "simulated", retrieved: ret.ids, retrievalScores: ret.scores, candidates: cands, forwardChaining: { ...fc, report: fcRep }, final, source, report: finalRep, unmet: fc.unmet, excluded: excluded.slice() };
  },
};

/* --------------------------------------------------------- composition */
const Composition = {
  subsets(arr, maxK) { const out = []; const rec = (i, cur) => { if (cur.length) out.push(cur.slice()); if (cur.length === maxK) return; for (let j = i; j < arr.length; j++) { cur.push(arr[j]); rec(j + 1, cur); cur.pop(); } }; rec(0, []); return out; },
  quality(K, desc, host) {
    const st = S(); const p = st.config.params; let Q = 0; const perCap = {};
    for (const r of desc.capabilities) {
      const pr = SelectionModel.probs(K, r, desc, host); let q = 0;
      K.forEach(id => { const t = byId(st.tools, id); const mu = t.card ? (t.card.caps[r] || 0) : 0; q += pr.p[id] * (mu >= p.tauMu ? mu : -p.wrongCost); });
      perCap[r] = { value: q, probs: pr.p, none: pr.none }; Q += q;
    }
    return { Q: Q / desc.capabilities.length, perCap };
  },
  candidates(desc) {
    const st = S(); const p = st.config.params;
    return st.tools.filter(t => t.card && desc.capabilities.some(r => (t.card.caps[r] || 0) >= 0.2)).map(t => t.id)
      .sort((a, b) => { const v = id => Math.max(...desc.capabilities.map(r => (byId(st.tools, id).card.caps[r] || 0))); return v(b) - v(a); }).slice(0, p.N);
  },
  /** Exact enumeration of admissible toolkits for one task and host. */
  task(desc, host) {
    const st = S(); const p = st.config.params; const N = this.candidates(desc); const counts = { C1: 0, C2: 0, C3: 0, C4: 0 }; const examples = {}; const ok = [];
    const flowIssues = {}; N.forEach(id => { flowIssues[id] = Governance.toolkitFlows([id], desc, host); });
    for (const K of this.subsets(N, Math.min(p.kMax + 1, N.length))) {
      if (K.length > p.kMax) { counts.C4++; examples.C4 = examples.C4 || { K, why: `${K.length} tools > maximum ${p.kMax}` }; continue; }
      const uncovered = desc.capabilities.filter(r => K.filter(id => (byId(st.tools, id).card.caps[r] || 0) >= p.tauMu).length < p.nMin);
      if (uncovered.length) { counts.C1++; examples.C1 = examples.C1 || { K, why: `no tool with confidence ≥ ${p.tauMu} for ${uncovered.map(r => capById(r).name).join(", ")}` }; continue; }
      const L = SelectionModel.load(K); if (L > host.budget) { counts.C2++; examples.C2 = examples.C2 || { K, why: `load ${L} tokens > budget ${host.budget}` }; continue; }
      const iss = K.flatMap(id => flowIssues[id]); if (iss.length) { counts.C3++; examples.C3 = examples.C3 || { K, why: iss[0].flow + ": " + iss[0].reason }; continue; }
      const q = this.quality(K, desc, host); ok.push({ K, load: L, Q: q.Q, perCap: q.perCap });
    }
    ok.sort((a, b) => b.Q - a.Q || a.load - b.load);
    const uncoverable = desc.capabilities.filter(r => N.filter(id => (byId(st.tools, id).card.caps[r] || 0) >= p.tauMu).length < p.nMin);
    return { host: host.id, candidates: N, feasible: ok, best: ok[0] || null, rejected: counts, examples, uncoverable, flowIssues };
  },
  handoffs(order, held) { const st = S(); const sup = {}; const out = []; order.forEach(id => { const d = byId(st.descriptors, id); d.requires.forEach(a => { if (sup[a] && !held.includes(a)) out.push({ from: sup[a], to: id, artefact: a }); }); d.delivers.forEach(a => { if (!sup[a]) sup[a] = id; }); }); return out; },
  qos(order, choice, gap) {
    const st = S(); const finish = {}; let price = 0; let dep = 1;
    for (const id of order) {
      const d = byId(st.descriptors, id); const preds = this.handoffs(order, []).filter(h => h.to === id).map(h => finish[h.from] || 0);
      let dur = d.durationDays, pr = d.fee;
      const c = choice[id];
      if (c && !gap.includes(id)) { const tools = c.K.map(x => byId(st.tools, x)); dur += Math.max(...tools.map(t => t.qos.latencyS)) / 86400; pr += Math.max(...tools.map(t => t.qos.price)); dep *= Math.min(...tools.map(t => t.qos.dependability)); }
      finish[id] = Math.max(0, ...preds) + dur; price += pr;
    }
    return { duration: Math.max(0, ...Object.values(finish)), price, dependability: dep };
  },
  /** Joint composition: exact per task and host, enumeration of the best options across tasks, constraints C5–C6, non-dominated plans. */
  run(order, held, gapTasks) {
    const st = S(); const p = st.config.params; const q = st.config.qos;
    const perTask = {}; const hostRejections = {};
    for (const id of order) {
      const d = byId(st.descriptors, id); perTask[id] = []; hostRejections[id] = [];
      if (gapTasks.includes(id)) continue;
      for (const h of st.hosts) { const adm = Governance.hostAdmissible(h, d); if (!adm.ok) { hostRejections[id].push({ host: h.id, reason: adm.reason }); continue; } perTask[id].push(this.task(d, h)); }
    }
    const infeasible = order.filter(id => !gapTasks.includes(id) && !perTask[id].some(x => x.best));
    const options = {}; order.forEach(id => { options[id] = gapTasks.includes(id) ? [null] : perTask[id].filter(x => x.best).sort((a, b) => b.best.Q - a.best.Q).slice(0, 3).map(x => ({ host: x.host, ...x.best })); });
    const plans = []; const violations = [];
    if (!infeasible.length) {
      let combos = [{}]; for (const id of order) { const nxt = []; for (const c of combos) for (const o of options[id]) { nxt.push({ ...c, [id]: o }); if (nxt.length > 400) break; } combos = nxt; }
      const hos = this.handoffs(order, held);
      combos.forEach((choice, i) => {
        const broken = hos.filter(h => !gapTasks.includes(h.from) && !gapTasks.includes(h.to) && (!choice[h.to].K.some(t => byId(st.tools, t).inputArtefacts.includes(h.artefact)) || !choice[h.from].K.some(t => byId(st.tools, t).outputArtefacts.includes(h.artefact))));
        const qq = this.qos(order, choice, gapTasks);
        const c6 = []; if (qq.duration > q.maxDurationDays) c6.push(`duration ${fmt(qq.duration, 1)} d > ${q.maxDurationDays}`); if (qq.price > q.maxPrice) c6.push(`price ${fmt(qq.price, 0)} € > ${q.maxPrice}`); if (qq.dependability < q.minDependability) c6.push(`dependability ${fmt(qq.dependability, 3)} < ${q.minDependability}`);
        const equipped = order.filter(id => !gapTasks.includes(id));
        const plan = { id: "P" + (i + 1), choice, gap: gapTasks.slice(), Qbar: equipped.length ? sum(equipped.map(id => choice[id].Q)) / equipped.length : 0, ...qq, left: gapTasks.length, c5: broken, c6 };
        if (broken.length || c6.length) violations.push(plan); else plans.push(plan);
      });
    }
    const dominated = (a, b) => (b.Qbar >= a.Qbar && b.left <= a.left && b.price <= a.price && b.duration <= a.duration) && (b.Qbar > a.Qbar || b.left < a.left || b.price < a.price || b.duration < a.duration);
    const front = plans.filter(a => !plans.some(b => b !== a && dominated(a, b)));
    front.sort((a, b) => a.left - b.left || b.Qbar - a.Qbar);
    return { at: new Date().toISOString(), perTask, hostRejections, infeasible, plans: front, allFeasible: plans.length, violatingPlans: violations.slice(0, 5), defaultPlan: front[0] ? front[0].id : null, params: { kMax: p.kMax, N: p.N, nMin: p.nMin, tauMu: p.tauMu } };
  },
};

/* --------------------------------------------------------------- repair */
const Repair = {
  /** Corrections per blocked task; the minimum-weight correction picks the lightest applicable one for each. */
  propose(run) {
    const st = S(); const w = st.config.params.repairWeights; const comp = run.composition; const out = [];
    const blocked = (comp.infeasible || []).slice();
    for (const id of blocked) {
      const d = byId(st.descriptors, id); const rows = comp.perTask[id] || []; const cands = [];
      const alt = st.descriptors.filter(x => x.id !== id && x.delivers.every(a => d.delivers.includes(a)) && !(run.excluded || []).includes(x.id));
      alt.forEach(a => cands.push({ kind: "alternative", weight: w.alternative, task: id, target: a.id, label: `Replace ${id} by ${a.id} (${a.name})`, impact: `Re-derives the workflow without ${id}; ${a.name} delivers the same document.` }));
      // Which capabilities lack a provider that passes the flow check, and what would unblock them.
      const tau = st.config.params.tauMu; const seen = new Set();
      for (const r of rows.filter(x => !x.best)) {
        const h = byId(st.hosts, r.host);
        const blockedCaps = d.capabilities.filter(cap => !r.candidates.some(tid => (byId(st.tools, tid).card.caps[cap] || 0) >= tau && !(r.flowIssues[tid] || []).length));
        if (!blockedCaps.length && r.rejected.C2 > 0) { const v = Math.ceil((h.budget * 1.5) / 100) * 100; if (!seen.has("b" + h.id)) { seen.add("b" + h.id); cands.push({ kind: "budget", weight: w.budget, task: id, target: h.id, value: v, label: `Raise the context budget of ${h.name} to ${v} tokens`, impact: "Configuration change of the model host." }); } }
        for (const cap of blockedCaps) {
          const providers = r.candidates.filter(tid => (byId(st.tools, tid).card.caps[cap] || 0) >= tau);
          for (const tid of providers) {
            const iss = r.flowIssues[tid] || []; const t = byId(st.tools, tid);
            if (iss.length && iss.every(x => x.flow.includes("uncertified")) && t.certification.status !== "rejected") { if (!seen.has("c" + tid)) { seen.add("c" + tid); cands.push({ kind: "certify", weight: w.certify, task: id, target: tid, label: `Audit and certify ${tid}`, impact: `Would lift the pessimistic labels of the only tool mapped to “${capById(cap).name}” on ${h.name}, if the audit confirms its declarations.` }); } }
            else if (iss.length && !iss.some(x => x.flow.startsWith("forwarding"))) { const key = "s" + iss[0].flow; if (!seen.has(key)) { seen.add(key); cands.push({ kind: "sharing", weight: w.sharing, task: id, target: h.id, label: `Agree a sharing base for ${iss[0].flow.split(" (")[0]}`, impact: `Would let ${tid} be offered on ${h.name}; requires an agreement between the organizations concerned.` }); } }
          }
        }
      }
      const missing = rows.length ? rows[0].uncoverable : [];
      cands.push({ kind: "requester", weight: w.requester, task: id, label: `Leave ${id} to the requester`, impact: `The requester completes “${d.name}” through the usual channel; downstream tasks use its document.` });
      const viol = []; rows.forEach(r => Object.entries(r.rejected).forEach(([c, n]) => { if (n && r.examples[c]) viol.push(`${c} on ${byId(st.hosts, r.host).name}: ${r.examples[c].why}`); }));
      (comp.hostRejections[id] || []).forEach(h => viol.push(`Host ${byId(st.hosts, h.host).name}: ${h.reason}`));
      if (!rows.length && !(comp.hostRejections[id] || []).length) viol.push("No model host is configured.");
      cands.sort((a, b) => a.weight - b.weight);
      out.push({ task: id, violations: viol, missingCapabilities: missing.map(r => capById(r).name), corrections: cands, recommended: cands[0] });
    }
    return out;
  },
  ledgerEntries(run, proposal) {
    return proposal.map(p => ({ id: uid("gap"), task: p.task, at: new Date().toISOString(), missingArtefacts: [], missingCapabilities: p.missingCapabilities,
      incompatibleTools: [], violated: p.violations, organizationalChange: p.corrections.filter(c => c.kind === "sharing" || c.kind === "alternative").map(c => c.label),
      suggested: p.recommended ? p.recommended.label : "", status: "open" }));
  },
};
