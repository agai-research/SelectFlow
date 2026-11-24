/* Base state built from the real registries (DATA, data.js). Persistence stores only what differs from it. */
const SEED_VERSION = 12;
const slugId = s => "o-" + String(s).toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40);
function seedState() {
  const now = "2026-11-24T00:00:00.000Z";   // fixed, so that the base can be compared with saved changes
  const SMALL = new Set(["of", "and", "the", "for", "in", "on", "to", "a", "an"]);
  const cleanName = n => { let x = String(n).split(/\s+(?:Service\s*\/|Method of Implementation|Department of|Directorate)/)[0].trim();
    const letters = x.replace(/[^A-Za-z]/g, ""); if (letters && x.replace(/[^A-Z]/g, "").length / letters.length > 0.6) x = x.toLowerCase();
    return x.split(/\s+/).map((w, i) => (i && SMALL.has(w.toLowerCase())) ? w.toLowerCase() : w.charAt(0).toUpperCase() + w.slice(1)).join(" "); };
  const orgId = {}; DATA.orgs.forEach(o => orgId[o.id] = slugId(cleanName(o.name)));
  const seenOrg = new Map(); DATA.orgs.forEach(o => { const id = orgId[o.id]; if (!seenOrg.has(id)) seenOrg.set(id, { id, name: ["ThirdParty", "PortalContractor", "GovCloud"].includes(o.id) ? o.name : cleanName(o.name), category: o.category, country: o.country, description: "", status: "active", base: true }); });
  const organizations = [...seenOrg.values()];
  const servers = DATA.servers.map(s => ({ id: s.id, orgId: orgId[s.org], name: s.id.replace(/^s_/, "").replace(/_/g, "-"), endpoint: "registry://" + s.id.replace(/^s_/, ""), status: "registry", preambleTokens: s.preamble, country: s.country, base: true }));
  const descTitle = {}; DATA.descriptors.forEach(d => descTitle[d.id] = d.title);
  const kindName = { submit: "Apply", pay: "Pay", get: "Obtain", upload: "Fill and upload forms" };
  const tools = DATA.tools.map(t => ({
    id: t.id, serverId: t.server, orgId: orgId[t.org], country: t.country, kind: t.kind,
    name: (kindName[t.kind] || t.kind) + ": " + (descTitle[t.family] || t.family || t.id), description: t.desc,
    inputArtefacts: t.cons, outputArtefacts: t.prod, inputCategories: t.inp, tokens: t.cost,
    inputSchema: { type: "object", properties: { applicant_id: { type: "string" }, details: { type: "string" } }, required: ["applicant_id"] },
    outputSchema: { type: "object", properties: Object.fromEntries((t.prod.length ? t.prod : ["status"]).map(a => [a, { type: "string" }])) },
    qos: { latencyS: t.lat, price: t.pr, dependability: t.dep },
    declared: { forwards: false, retains: t.ret, attested: t.org !== "ThirdParty" }, truth: { forwards: !!t.tfwd, retains: t.tret },
    card: { caps: t.caps, adequacy: t.ad, generatedAt: now, status: "registry", provenance: "registry", passes: 0 },
    certification: { status: "uncertified", score: null, audited: false, attested: t.org !== "ThirdParty", history: [] }, family: t.family, base: true }));
  const descriptors = DATA.descriptors.map(d => ({ id: d.id, orgId: orgId[d.org], country: d.country, name: d.title, description: d.text, purpose: d.purpose,
    requires: d.In, delivers: d.Out, capabilities: d.R, companions: d.comp, elicited: d.A, fee: d.fee, durationDays: d.dur, preconditions: "", record: d.src, base: true }));
  const groups = { portal: [], cloud: [], competence: [], exchange: [] };
  DATA.sharing.forEach(([p, c, o, r]) => { const rule = { purpose: p, category: c, origin: o === "REQ" ? "REQ" : orgId[o] || o, receiver: orgId[r] || r, effect: "allow", retention: false };
    (r === "PortalContractor" ? groups.portal : r === "GovCloud" ? groups.cloud : o === "REQ" ? groups.competence : groups.exchange).push(rule); });
  const sharingBases = [
    { id: "sb-portal", name: "Citizen portal contract", kind: "Contract", orgIds: [orgId.PortalContractor], description: "Identity, business and location data may reach the portal contractor's assistant.", rules: groups.portal, base: true },
    { id: "sb-cloud", name: "Government cloud agreement", kind: "Agreement", orgIds: [orgId.GovCloud], description: "All categories may reach the government cloud's assistant.", rules: groups.cloud, base: true },
    { id: "sb-competence", name: "Competence", kind: "Competence", orgIds: [], description: "Each body may collect the documents its own procedures require.", rules: groups.competence, base: true },
    { id: "sb-exchange", name: "Registry-sanctioned exchanges", kind: "Agreement", orgIds: [], description: "A body may send a document to the body whose procedure requires it, as the registry states.", rules: groups.exchange, base: true } ];
  const beta = DATA.fitted.beta, gamma = DATA.fitted.gamma;
  const hosts = DATA.hosts.map(h => ({ id: h.id, name: h.name, operatorOrgId: orgId[h.org], budget: h.budget, supports: ["registration", "certification", "notification", "modification", "authorization", "information"], status: "configured", base: true,
    model: { beta: beta.slice(), gamma, features: ["intercept", "text similarity", "name similarity", "assertive wording", "own organization"], mode: "fitted-recorded", trainedAt: null, traces: DATA.traces.length, logLik: null, source: DATA.fitted.source } }));
  const artefacts = DATA.artefacts.map(a => ({ id: a.id, name: a.name, categories: a.cats, usuallyHeld: !DATA.descriptors.some(d => d.Out.includes(a.id)) }));
  const capabilities = DATA.capabilities.map(c => ({ id: c.id, name: c.name, category: c.kind, statement: c.statement }));
  const config = {
    domain: "Business and civic procedures (Greece, Saudi Arabia)", description: "",
    dataCategories: [["ID", "Identity"], ["BUS", "Business"], ["LOC", "Premises and location"], ["FIN", "Financial and tax"], ["HLT", "Health"]].map(([id, name]) => ({ id, name })),
    purposes: ["registration", "certification", "notification", "modification", "authorization", "information"], taskCategories: ["Apply", "Pay", "Obtain"],
    artefacts, capabilities, qos: { maxDurationDays: 400, maxPrice: 2000, minDependability: 0.5 },
    params: { kMax: 5, N: 8, nMin: 1, tauMu: 0.5, tauChi: 0.6, lambdaQ: 0.3, wrongCost: 0.2, nRetrieve: 12, kCandidates: 3, nRevisions: 2, eps: 0.1, auditFraction: 0.3, repairWeights: { requester: 3, alternative: 1, budget: 2, sharing: 4, certify: 2 } },
    simulation: { seed: 7 }, llm: { mode: "claude", model: "claude-sonnet-5-5" } };
  return { version: SEED_VERSION, createdAt: now, updatedAt: now, config, organizations, servers, tools, descriptors, hosts, sharingBases,
    traces: {}, prepOps: [], runs: [], events: [], counters: { run: 0, op: 0 }, connectors: [] };
}
