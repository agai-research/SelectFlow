/* ============================================================================
   Services that reach real systems: language models and MCP connectors.
   ========================================================================== */
const LLM = {
  session: { key: "", base: "", model: "", ollama: "http://localhost:11434" },   // keys are kept in memory and this tab only
  init() { try { const s = JSON.parse(sessionStorage.getItem("sf-llm-session") || "null"); if (s) this.session = s; } catch (e) { } },
  remember() { try { sessionStorage.setItem("sf-llm-session", JSON.stringify(this.session)); } catch (e) { } },
  cfg() { return S().config.llm; },
  label() { const c = this.cfg(); return c.mode === "claude" ? "Claude · " + ({ "claude-haiku-5-5": "Haiku 5.5", "claude-sonnet-5-5": "Sonnet 5.5", "claude-opus-5-5": "Opus 5.5" }[c.model] || c.model) : c.mode === "anthropic" ? "Anthropic API · " + (this.session.model || "model") : c.mode === "openai" ? "OpenAI-compatible · " + (this.session.model || "model") : c.mode === "ollama" ? "Ollama · " + (this.session.model || "model") : "No model (recorded examples)"; },
  readySync() { const c = this.cfg(); if (c.mode === "none") return false; if (c.mode === "claude") return this.hosted(); return !this.hosted() && !!this.session.model && (c.mode === "ollama" || !!this.session.key); },
  hosted() { return !!(window.claude && typeof window.claude.use === "function"); },
  async sample() { return this.hosted() ? await window.claude.use("sample") : null; },
  async status() {
    const c = this.cfg();
    if (c.mode === "none") return { ok: false, why: "No model selected." };
    if (c.mode === "claude") { const s = await this.sample(); return s ? { ok: true } : { ok: false, why: "Claude is available when this page is opened in claude.ai." }; }
    if (this.hosted()) return { ok: false, why: "Pages published on claude.ai cannot send requests to other providers. Download the page and open it locally to use this provider." };
    if (c.mode !== "ollama" && !this.session.key) return { ok: false, why: "Enter an API key." };
    if (!this.session.model) return { ok: false, why: "Enter the name of the model." };
    if (c.mode === "openai" && !/^https?:\/\//.test(this.session.base)) return { ok: false, why: "Enter the endpoint, for example https://api.openai.com/v1." };
    return { ok: true };
  },
  last: null,
  tier(m) { return { "claude-haiku-5-5": "quick", "claude-opus-5-5": "complex" }[m] || "default"; },
  async ensure() { const st = await this.status(); if (!st.ok) throw new Error(st.why); },
  async text(prompt, opts = {}) {
    try { return await this._text(prompt, opts); }
    catch (e) { if (e instanceof TypeError && /fetch/i.test(e.message)) throw new Error("The endpoint could not be reached (address, network or cross-origin access refused)."); throw e; }
  },
  async _text(prompt, opts = {}) {
    await this.ensure(); const c = this.cfg(); const t0 = performance.now();
    if (c.mode === "claude") {
      const s = await this.sample();
      const r = await s(prompt, { model: c.model, modelTier: this.tier(c.model), ...(opts.tools ? { tools: opts.tools } : {}) });
      this.last = { provider: "Claude in claude.ai", requested: c.model, applied: r.modelApplied || r.modelTierApplied || c.model, ms: Math.round(performance.now() - t0) };
      return r.text;
    }
    if (this.hosted()) throw new Error("Other providers cannot be reached from a page published on claude.ai.");
    if (c.mode === "anthropic") {
      const r = await fetch("https://api.anthropic.com/v1/messages", { method: "POST", headers: { "content-type": "application/json", "x-api-key": this.session.key, "anthropic-version": "2023-06-01", "anthropic-dangerous-direct-browser-access": "true" }, body: JSON.stringify({ model: this.session.model || "claude-sonnet-5-5", max_tokens: 1200, messages: [{ role: "user", content: prompt }] }) });
      if (!r.ok) throw new Error(r.status === 401 ? "The API key was refused (401)." : "Provider error " + r.status); const j = await r.json(); this.last = { provider: "Anthropic API", requested: this.session.model, applied: j.model, ms: Math.round(performance.now() - t0) }; return (j.content || []).map(b => b.text || "").join("");
    }
    const base = c.mode === "ollama" ? this.session.ollama + "/v1" : this.session.base.replace(/\/$/, "");
    const r = await fetch(base + "/chat/completions", { method: "POST", headers: { "content-type": "application/json", ...(this.session.key ? { authorization: "Bearer " + this.session.key } : {}) }, body: JSON.stringify({ model: this.session.model, messages: [{ role: "user", content: prompt }], temperature: 0 }) });
    if (!r.ok) throw new Error(r.status === 401 ? "The API key was refused (401)." : r.status === 404 ? "Unknown model or endpoint (404)." : "Provider error " + r.status); const j = await r.json(); this.last = { provider: c.mode === "ollama" ? "Ollama" : "OpenAI-compatible", requested: this.session.model, applied: j.model, ms: Math.round(performance.now() - t0) }; return j.choices[0].message.content;
  },
  async json(prompt) {
    await this.ensure(); const c = this.cfg();
    if (c.mode === "claude") { const s = await this.sample(); return await s.json(prompt, { model: c.model, modelTier: this.tier(c.model) }); }
    const t = await this.text(prompt + "\nAnswer with JSON only."); const m = t.match(/[\[{][\s\S]*[\]}]/); if (!m) throw new Error("The model did not return JSON."); return JSON.parse(m[0]);
  },
  /** Live agent call: the toolkit is offered as callable tools; returns the tool the model calls (or null). */
  async selectTool(step, toolkit) {
    await this.ensure(); const st = S(); const c = this.cfg(); let chosen = null;
    if (c.mode === "claude") {
      const s = await this.sample();
      const tools = toolkit.map(id => { const t = byId(st.tools, id); return { name: id.replace(/[^A-Za-z0-9_-]/g, "_").slice(0, 60), description: t.description, inputSchema: { type: "object", properties: { applicant_id: { type: "string" }, details: { type: "string" } } }, execute: () => { if (!chosen) chosen = id; return { status: "accepted (simulated server)" }; } }; });
      await s("You carry out one step of an administrative procedure for a citizen. Call the single tool that performs it, once. If none fits, answer briefly.\nStep: " + step, { model: c.model, tools });
      return chosen;
    }
    const list = toolkit.map(id => `- ${id}: ${byId(st.tools, id).description}`).join("\n");
    const out = await this.json(`You are an agent with these tools:\n${list}\nStep to perform: ${step}\nReturn {"tool": "<tool id>" or null}.`);
    return toolkit.includes(out.tool) ? out.tool : null;
  },
  async ollamaModels() { const r = await fetch(this.session.ollama + "/api/tags"); if (!r.ok) throw new Error("Ollama is not reachable at " + this.session.ollama); return (await r.json()).models.map(m => m.name); },
  async ollamaPull(name, onProgress) {
    const r = await fetch(this.session.ollama + "/api/pull", { method: "POST", body: JSON.stringify({ name, stream: true }) }); if (!r.ok) throw new Error("Download refused: " + r.status);
    const rd = r.body.getReader(); const dec = new TextDecoder(); let buf = "";
    for (;;) { const { done, value } = await rd.read(); if (done) break; buf += dec.decode(value, { stream: true }); const lines = buf.split("\n"); buf = lines.pop(); lines.forEach(l => { try { const j = JSON.parse(l); onProgress(j); } catch (e) { } }); }
  },
};

const Connectors = {
  OBSERVED: { "Google Drive": ["list_recent_files"] },   // tools whose real request/response shape was checked
  async mcp() { return LLM.hosted() ? await window.claude.use("mcp") : null; },
  async list() { const m = await this.mcp(); if (!m) throw new Error("Connectors are available when this page is opened in claude.ai."); return (await m.listTools()).servers; },
  async allow(server) { try { const p = await window.claude.use("permissions"); if (p) await p.request(["mcp:" + server]); } catch (e) { } },
  async describe(server, tool) { const m = await this.mcp(); return await m.describeTool(server, tool); },
  async testCall(server) {
    const m = await this.mcp(); if (server !== "Google Drive") throw new Error("A test call is offered only for tools whose responses were checked.");
    const r = await m.callTool("Google Drive", "list_recent_files", { pageSize: 5, excludeContentSnippets: true });
    const p = r.payload || r; return (p.files || []).map(f => ({ title: String(f.title || ""), type: String(f.mimeType || "").split(".").pop(), modified: String(f.modifiedTime || "") }));
  },
  importServer(info) {
    Store.commit(s => {
      const oid = "o-connector-" + info.server.toLowerCase().replace(/[^a-z0-9]+/g, "-");
      if (!byId(s.organizations, oid)) s.organizations.push({ id: oid, name: info.server + " (connector)", category: "Third-party provider", country: "Connector", description: "", status: "active", base: false });
      const sid = "s_conn_" + info.server.toLowerCase().replace(/[^a-z0-9]+/g, "_");
      if (!byId(s.servers, sid)) s.servers.push({ id: sid, orgId: oid, name: info.server, endpoint: "mcp://" + info.server, status: "connected", preambleTokens: 120, base: false });
      info.tools.forEach(t => { const id = "conn." + sid.slice(7) + "." + t.name; if (byId(s.tools, id)) return;
        s.tools.push({ id, serverId: sid, orgId: oid, kind: "connector", name: t.name, description: t.description || t.name, inputArtefacts: [], outputArtefacts: [], inputCategories: ["ID"], tokens: 60 + Math.round((t.description || "").length / 4), inputSchema: { type: "object" }, outputSchema: { type: "object" }, qos: { latencyS: 1, price: 0, dependability: 0.99 }, declared: { forwards: false, retains: [], attested: false }, truth: { forwards: false, retains: [] }, card: null, certification: { status: "uncertified", score: null, audited: false, attested: false, history: [] }, readOnly: !!(t.annotations && t.annotations.readOnlyHint), base: false }); });
      Store.log("connector", `Imported ${info.tools.length} tool(s) from the ${info.server} connector`);
    });
  },
};
LLM.init();
