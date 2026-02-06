const API_BASE = "";

const state = {
  profiles: [],
  activeProfileTag: null,
  jobs: [],
  jobsOffset: 0,
  jobsHasMore: true,
  jobsPageSize: 40,
  jobsLoadingMore: false,
  jobsLoadingInitial: false,
  lastSearchQueries: [],
  lastSearchUrls: [],
  autoPullInProgress: false,
  autoPullEnabled: true,
  matchesByJobId: {},
  jobStatuses: {},
  currentTab: "all",
  autoApplyResults: {},
  scoreState: "idle", // idle|running|ok|partial|failed
  editors: {}, // jobId -> editor payload
};

const storedActiveProfileTag = localStorage.getItem("activeProfileTag");

const tailoredByJob = {};
const syncTimers = {};
let applyRunPollToken = 0;
let activeApplyRunId = null;
let autoPullTimer = null;
let scoreRetryTimer = null;

const el = {
  statusPill: document.getElementById("status-pill"),
  profileSummary: document.getElementById("profile-summary"),
  activeProfileSelect: document.getElementById("active-profile-select"),
  targetsList: document.getElementById("targets-list"),
  baseTag: document.getElementById("base-tag"),
  uploadForm: document.getElementById("upload-form"),
  uploadResult: document.getElementById("upload-result"),
  jobsGrid: document.getElementById("jobs-grid"),
  searchForm: document.getElementById("search-form"),
  searchInput: document.getElementById("search-query"),
  apiKeyInput: document.getElementById("api-key-input"),
  saveApiKeyBtn: document.getElementById("save-api-key"),
  loadingOverlay: document.getElementById("loading-overlay"),
  loadingText: document.getElementById("loading-text"),
  jobsCount: document.getElementById("jobs-count"),
  tailorModal: document.getElementById("tailor-modal"),
  tailorModalBody: document.getElementById("tailor-modal-body"),
  tailorModalClose: document.getElementById("tailor-modal-close"),
  tailorDownloadTxt: document.getElementById("tailor-download-txt"),
  tailorDownloadHtml: document.getElementById("tailor-download-html"),
  tailorModalMeta: document.getElementById("tailor-modal-meta"),
  applyRunModal: document.getElementById("apply-run-modal"),
  applyRunMeta: document.getElementById("apply-run-meta"),
  applyRunStatus: document.getElementById("apply-run-status"),
  applyRunId: document.getElementById("apply-run-id"),
  applyRunSteps: document.getElementById("apply-run-steps"),
  applyRunClose: document.getElementById("apply-run-close"),
  applyRunActions: document.getElementById("apply-run-actions"),
  applyRunPrompt: document.getElementById("apply-run-prompt"),
  applyRunCloseOnly: document.getElementById("apply-run-close-only"),
  applyRunCloseSubmitted: document.getElementById("apply-run-close-submitted"),
};

let apiKey = localStorage.getItem("apiKey") || "";
if (apiKey && el.apiKeyInput) el.apiKeyInput.value = apiKey;

function arr(v) {
  return Array.isArray(v) ? v : [];
}

function esc(v) {
  return String(v || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function host(url) {
  try {
    return new URL(url).host.replace(/^www\./, "");
  } catch {
    return "";
  }
}

function postedLabel(job) {
  const raw = job?.posted_at || job?.ingested_at;
  if (!raw) return "";
  const dt = new Date(raw);
  if (Number.isNaN(dt.getTime())) return "";
  const diffMs = Date.now() - dt.getTime();
  const days = Math.floor(diffMs / (1000 * 60 * 60 * 24));
  if (days <= 0) return "posted today";
  if (days === 1) return "posted 1 day ago";
  if (days < 31) return `posted ${days} days ago`;
  return `posted ${dt.toLocaleDateString()}`;
}

function isUnknownTitle(title) {
  return !title || /^unknown$/i.test(String(title).trim());
}

function looksLikeTitleText(text) {
  const low = String(text || "").toLowerCase();
  return /(engineer|scientist|developer|analyst|manager|architect|researcher|director|intern)/.test(low);
}

function looksLikeLocationText(text) {
  const v = String(text || "").trim();
  if (!v) return false;
  if (looksLikeTitleText(v) && !/remote|hybrid|onsite|on-site/i.test(v)) return false;
  if (/remote|hybrid|onsite|on-site/i.test(v)) return true;
  if (/,\s*[A-Z]{2}(?:\s|$|,)/.test(v)) return true;
  if (/(United States|USA|India|Canada|United Kingdom|Germany|France|Australia)/i.test(v)) return true;
  return /^[A-Z][A-Za-z .'-]+,\s*[A-Z][A-Za-z .'-]+(?:,\s*[A-Z][A-Za-z .'-]+)?$/.test(v);
}

function titleFromRawText(rawText) {
  const lines = String(rawText || "")
    .split(/\r?\n/)
    .map((x) => x.trim())
    .filter(Boolean)
    .slice(0, 180);

  for (const prefix of ["JobPosting Title:", "Page Title:", "OG Title:", "Twitter Title:", "Title:"]) {
    const line = lines.find((ln) => ln.startsWith(prefix));
    if (!line) continue;
    const cand = line.slice(prefix.length).trim().replace(/\s+\|\s+.*$/, "").replace(/\s+-\s+.*$/, "");
    if (cand && looksLikeTitleText(cand)) return cand;
  }
  for (const line of lines) {
    if (line.length < 8 || line.length > 100) continue;
    if (!looksLikeTitleText(line)) continue;
    if (looksLikeLocationText(line)) continue;
    return line;
  }
  return "";
}

function displayTitleForJob(job) {
  const explicit = String(job?.title || "").trim();
  if (!isUnknownTitle(explicit)) return explicit;
  const fromLocation = String(job?.location || "").trim();
  if (fromLocation && looksLikeTitleText(fromLocation) && !looksLikeLocationText(fromLocation)) return fromLocation;
  const fromRaw = titleFromRawText(job?.raw_text);
  return fromRaw || explicit || "Unknown";
}

function displayLocationForJob(job, displayTitle) {
  const loc = String(job?.location || "").trim();
  if (!loc) return "";
  if (displayTitle && loc === displayTitle) return "";
  if (looksLikeTitleText(loc) && !looksLikeLocationText(loc)) return "";
  return loc;
}

function loading(text = "Processing...") {
  if (!el.loadingOverlay) return;
  el.loadingOverlay.classList.remove("hidden");
  el.loadingOverlay.classList.add("flex");
  if (el.loadingText) el.loadingText.textContent = text;
}

function doneLoading() {
  if (!el.loadingOverlay) return;
  el.loadingOverlay.classList.add("hidden");
  el.loadingOverlay.classList.remove("flex");
}

function status(text, tone = "neutral") {
  const map = {
    neutral: ["bg-white/10", "bg-slate-400"],
    ok: ["bg-emerald-500/20", "bg-emerald-400"],
    warn: ["bg-amber-500/20", "bg-amber-400"],
    loading: ["bg-blue-500/20", "bg-blue-400"],
  };
  const [bg, dot] = map[tone] || map.neutral;
  el.statusPill.innerHTML = `<span class="w-2 h-2 rounded-full ${dot} ${tone === "loading" ? "animate-pulse" : ""}"></span>${text}`;
  el.statusPill.className = `inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold ${bg} border border-white/20`;
}

async function jfetch(path, options = {}) {
  const headers = options.headers ? { ...options.headers } : {};
  if (apiKey) headers["x-api-key"] = apiKey;
  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });
  if (!res.ok) {
    const bodyText = await res.text();
    const err = new Error(bodyText || res.statusText || `HTTP ${res.status}`);
    err.status = res.status;
    err.retryAfter = Number(res.headers.get("Retry-After") || 0);
    throw err;
  }
  return res.json();
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRateLimited(err) {
  return Number(err?.status) === 429;
}

function mergeJobs(existing, incoming) {
  const out = [];
  const seen = new Set();
  arr(existing)
    .concat(arr(incoming))
    .forEach((job) => {
      const id = String(job?.id || "");
      if (!id || seen.has(id)) return;
      seen.add(id);
      out.push(job);
    });
  return out;
}

function resetPaging() {
  state.jobsOffset = 0;
  state.jobsHasMore = true;
  state.jobsLoadingMore = false;
}

function saveApiKey() {
  apiKey = (el.apiKeyInput?.value || "").trim();
  if (apiKey) {
    localStorage.setItem("apiKey", apiKey);
    status("Saved", "ok");
  } else {
    localStorage.removeItem("apiKey");
    status("Cleared", "ok");
  }
}

function activeProfile() {
  return state.profiles.find((p) => p.tag === state.activeProfileTag) || null;
}

function persistActiveProfileTag(tag) {
  if (tag) localStorage.setItem("activeProfileTag", tag);
  else localStorage.removeItem("activeProfileTag");
}

function clearProfileScopedCaches() {
  stopAutoPull();
  state.jobs = [];
  resetPaging();
  state.matchesByJobId = {};
  state.jobStatuses = {};
  state.editors = {};
  state.autoApplyResults = {};
  state.lastSearchQueries = [];
  state.lastSearchUrls = [];
  Object.keys(tailoredByJob).forEach((k) => delete tailoredByJob[k]);
  if (scoreRetryTimer) {
    clearTimeout(scoreRetryTimer);
    scoreRetryTimer = null;
  }
}

function renderActiveProfileSelect() {
  if (!el.activeProfileSelect) return;
  if (!state.profiles.length) {
    el.activeProfileSelect.innerHTML = `<option value="">No profiles found</option>`;
    el.activeProfileSelect.value = "";
    return;
  }
  el.activeProfileSelect.innerHTML = state.profiles
    .map((p) => `<option value="${esc(p.tag)}">${esc(p.name || "Unknown")} (${esc(p.tag)})</option>`)
    .join("");
  if (state.activeProfileTag) {
    el.activeProfileSelect.value = state.activeProfileTag;
  }
}

async function switchActiveProfile(tag) {
  if (!tag || tag === state.activeProfileTag) return;
  stopAutoPull();
  state.activeProfileTag = tag;
  persistActiveProfileTag(tag);
  clearProfileScopedCaches();
  renderActiveProfileSelect();
  renderProfile();
  populateBaseTag();

  status("Switching profile...", "loading");
  loading("Loading profile context...");
  try {
    await loadStatuses();
    await loadJobs(state.currentTab, false);
    await refreshMatches();
    if (!state.lastSearchQueries.length) state.lastSearchQueries = defaultQueries();
    startAutoPull();
    status("Profile switched", "ok");
  } finally {
    doneLoading();
  }
}

function renderTargets(selected) {
  const targets = [
    ["data_scientist", "DS"],
    ["machine_learning_engineer", "MLE"],
    ["ai_engineer", "AI"],
    ["software_engineer_ml_ai", "SWE"],
  ];
  el.targetsList.innerHTML = "";
  targets.forEach(([id, label]) => {
    const on = selected.includes(id);
    const b = document.createElement("button");
    b.className = `px-2 py-1 rounded-full text-xs font-medium ${on ? "bg-ink text-white" : "bg-slate-100 text-slate-600 hover:bg-slate-200"}`;
    b.textContent = label;
    b.onclick = async () => {
      const p = activeProfile();
      if (!p) return;
      const next = new Set(p.targets || []);
      next.has(id) ? next.delete(id) : next.add(id);
      p.targets = [...next];
      renderTargets(p.targets);
      await jfetch(`/profiles/${p.tag}/targets`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ targets: p.targets }),
      });
    };
    el.targetsList.appendChild(b);
  });
}

function renderProfile() {
  const p = activeProfile();
  if (!p) {
    el.profileSummary.innerHTML = `<p class="text-slate-400 text-xs">No profile selected</p>`;
    return;
  }
  el.profileSummary.innerHTML = `
    <div class="flex items-center gap-2 p-2 rounded-lg bg-gradient-to-r from-slate-50 to-violet-50">
      <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center text-white font-semibold text-sm">${esc(
        (p.name || "?").charAt(0)
      )}</div>
      <div class="min-w-0">
        <p class="font-semibold text-ink text-sm truncate">${esc(p.name || "Unknown")}</p>
        <p class="text-xs text-slate-500 font-mono">${esc(p.tag)}</p>
      </div>
    </div>
  `;
  renderTargets(p.targets || []);
}

function populateBaseTag() {
  el.baseTag.innerHTML = `<option value="">Use resume only</option>`;
  state.profiles.forEach((p) => {
    const o = document.createElement("option");
    o.value = p.tag;
    o.textContent = `${p.name} (${p.tag})`;
    el.baseTag.appendChild(o);
  });
  if (state.activeProfileTag) {
    el.baseTag.value = state.activeProfileTag;
  }
}

async function loadProfiles() {
  status("Loading...", "loading");
  state.profiles = await jfetch("/profiles");
  const tags = new Set(state.profiles.map((p) => p.tag));
  if (!state.activeProfileTag && storedActiveProfileTag && tags.has(storedActiveProfileTag)) {
    state.activeProfileTag = storedActiveProfileTag;
  }
  if (state.activeProfileTag && !tags.has(state.activeProfileTag)) {
    state.activeProfileTag = null;
  }
  if (!state.activeProfileTag && state.profiles.length) {
    state.activeProfileTag = state.profiles[0].tag;
  }
  persistActiveProfileTag(state.activeProfileTag);
  renderActiveProfileSelect();
  renderProfile();
  populateBaseTag();
  status("Ready", "ok");
}

async function loadStatuses() {
  if (!state.activeProfileTag) return;
  try {
    state.jobStatuses = await jfetch(`/jobs/statuses?profile_tag=${state.activeProfileTag}`);
  } catch {
    state.jobStatuses = {};
  }
}

function updateCount() {
  if (!el.jobsCount) return;
  el.jobsCount.textContent = state.jobs.length ? `${state.jobs.length} jobs` : "";
}

function jobsEndpoint(mode, limit, offset) {
  if (mode === "saved") return `/jobs/saved?profile_tag=${state.activeProfileTag}&limit=${limit}&offset=${offset}`;
  if (mode === "applied") return `/jobs/applied?profile_tag=${state.activeProfileTag}&limit=${limit}&offset=${offset}`;
  return `/jobs?limit=${limit}&offset=${offset}`;
}

async function loadJobs(mode = "all", append = false) {
  if (!append) {
    resetPaging();
    state.jobs = [];
  }
  const offset = state.jobsOffset;
  const limit = state.jobsPageSize;
  const endpoint = jobsEndpoint(mode, limit, offset);
  const page = await jfetch(endpoint);
  const incoming = arr(page);
  state.jobs = append ? mergeJobs(state.jobs, incoming) : incoming;
  state.jobsOffset += incoming.length;
  state.jobsHasMore = incoming.length >= limit;
  updateCount();
  return incoming;
}

function defaultQueries() {
  const p = activeProfile();
  if (!p) return ["machine learning engineer remote US"];
  const q = arr(p.targets).slice(0, 2).map((t) => `${t.replace(/_/g, " ")} remote US`);
  return q.length ? q : ["machine learning engineer remote US"];
}

async function ingestJobs(queries = [], urls = []) {
  status("Searching...", "loading");
  loading("Searching job boards...");
  try {
    const q = arr(queries).filter(Boolean);
    const u = arr(urls).filter(Boolean);
    state.lastSearchQueries = q.length ? q : defaultQueries();
    state.lastSearchUrls = u;

    await jfetch("/jobs/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        queries: state.lastSearchQueries,
        urls: state.lastSearchUrls,
        limit: 120,
        profile_tag: state.activeProfileTag,
      }),
    });
    await loadStatuses();
    await loadJobs("all", false);
    updateCount();
    await refreshMatches();
    startAutoPull();
    status("Found", "ok");
  } catch (err) {
    if (isRateLimited(err)) status("Rate limited while searching. Retrying later.", "warn");
    else {
      console.warn("search ingest failed:", err);
      status("Search failed", "warn");
    }
  } finally {
    doneLoading();
  }
}

async function uploadResume(e) {
  e.preventDefault();
  const form = new FormData(el.uploadForm);
  const file = form.get("resume");
  if (!file || !file.name) return;

  status("Uploading...", "loading");
  loading("Parsing resume...");
  try {
    const payload = new FormData();
    payload.append("file", file);
    const baseTag = form.get("baseTag");
    if (baseTag) payload.append("base_tag", baseTag);
    const headers = {};
    if (apiKey) headers["x-api-key"] = apiKey;
    const res = await fetch(`${API_BASE}/profiles/from-resume${baseTag ? `?base_tag=${baseTag}` : ""}`, {
      method: "POST",
      body: payload,
      headers,
    });
    if (!res.ok) throw new Error(await res.text());
    const p = await res.json();
    state.activeProfileTag = p.tag;
    el.uploadResult.innerHTML = `<span class="text-emerald-600">Created ${esc(p.tag)}</span>`;
    await loadProfiles();
    await loadJobs(state.currentTab, false);
    await refreshMatches();
    if (!state.lastSearchQueries.length) state.lastSearchQueries = defaultQueries();
    startAutoPull();
    status("Ready", "ok");
  } catch (err) {
    el.uploadResult.innerHTML = `<span class="text-red-600">Error: ${esc(err.message || "Upload failed")}</span>`;
    status("Error", "warn");
  } finally {
    doneLoading();
  }
}

function scoreColor(score) {
  if (score >= 75) return "from-emerald-500 to-teal-500";
  if (score >= 55) return "from-amber-500 to-orange-500";
  if (score >= 35) return "from-slate-400 to-slate-500";
  return "from-red-400 to-red-500";
}

function buildMatchIndex(matches) {
  const idx = {};
  arr(matches).forEach((m) => {
    if (m?.job_id != null) idx[String(m.job_id)] = m;
  });
  return idx;
}

async function scoreIndividually(jobs) {
  const out = [];
  for (const job of arr(jobs)) {
    try {
      const res = await jfetch(`/match?profile_tag=${state.activeProfileTag}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(job),
      });
      out.push(res);
      await sleep(90);
    } catch (err) {
      if (isRateLimited(err)) break;
    }
  }
  return out;
}

async function refreshMatches(targetJobs = null) {
  if (scoreRetryTimer) {
    clearTimeout(scoreRetryTimer);
    scoreRetryTimer = null;
  }
  const jobsToScore = arr(targetJobs).length ? arr(targetJobs) : state.jobs;
  const replaceAll = !arr(targetJobs).length;
  if (!state.activeProfileTag || !jobsToScore.length) {
    if (replaceAll) state.matchesByJobId = {};
    renderJobs();
    return;
  }
  state.scoreState = "running";
  status("Scoring...", "loading");
  const useOverlay = replaceAll && jobsToScore.length > 20;
  if (useOverlay) loading("Matching jobs...");

  let matches = [];
  try {
    const batch = await jfetch(`/match/batch?profile_tag=${state.activeProfileTag}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(jobsToScore),
    });
    if (!Array.isArray(batch)) throw new Error("bad batch response");
    matches = batch;
    const idx = buildMatchIndex(matches);
    const missing = jobsToScore.filter((j) => !idx[String(j.id)]);
    if (missing.length && missing.length <= 8) {
      matches = matches.concat(await scoreIndividually(missing.slice(0, 8)));
      state.scoreState = "partial";
    } else if (missing.length) {
      state.scoreState = "partial";
    } else {
      state.scoreState = "ok";
    }
  } catch (err) {
    console.warn("batch scoring failed:", err);
    if (isRateLimited(err)) {
      state.scoreState = "failed";
      const waitSec = Math.max(2, Number(err?.retryAfter || 5));
      status(`Rate limited. Retrying in ${waitSec}s`, "warn");
      if (replaceAll) {
        if (scoreRetryTimer) clearTimeout(scoreRetryTimer);
        scoreRetryTimer = setTimeout(() => refreshMatches(), waitSec * 1000);
      }
    } else {
      matches = await scoreIndividually(jobsToScore.slice(0, 10));
      state.scoreState = matches.length ? "partial" : "failed";
    }
  } finally {
    const idx = buildMatchIndex(matches);
    if (replaceAll) state.matchesByJobId = idx;
    else state.matchesByJobId = { ...state.matchesByJobId, ...idx };
    renderJobs();
    if (useOverlay) doneLoading();
    if (state.scoreState === "failed") status("Scoring failed", "warn");
    else if (state.scoreState === "partial") status("Scores loaded (partial)", "warn");
    else status("Ready", "ok");
  }
}

function scoreBadge(match) {
  if (match?.fit_score == null) {
    return `
      <div class="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center">
        <span class="text-[10px] text-slate-500">${state.scoreState === "running" ? "..." : "-"}</span>
      </div>
    `;
  }
  return `
    <div class="w-12 h-12 rounded-full bg-gradient-to-br ${scoreColor(match.fit_score)} flex items-center justify-center shadow">
      <span class="text-lg font-bold text-white">${Math.round(match.fit_score)}</span>
    </div>
    <p class="text-xs text-slate-500 mt-1">${esc(match.confidence || "")}</p>
  `;
}

function renderJobs() {
  el.jobsGrid.innerHTML = "";
  updateCount();

  let jobs = state.jobs;
  if (state.currentTab === "saved") jobs = jobs.filter((j) => state.jobStatuses[j.id] === "saved");
  if (state.currentTab === "applied") jobs = jobs.filter((j) => state.jobStatuses[j.id] === "applied");
  if (state.currentTab === "all") jobs = jobs.filter((j) => state.jobStatuses[j.id] !== "hidden");
  if (!jobs.length) {
    el.jobsGrid.innerHTML = `<div class="text-center py-8 text-slate-400 text-sm">No jobs found</div>`;
    return;
  }

  jobs
    .sort((a, b) => (state.matchesByJobId[b.id]?.fit_score || 0) - (state.matchesByJobId[a.id]?.fit_score || 0))
    .forEach((job) => {
      const match = state.matchesByJobId[String(job.id)];
      const missMust = arr(match?.keyword_coverage?.must_have_missing).slice(0, 6);
      const missStack = arr(match?.keyword_coverage?.stack_missing).slice(0, 6);
      const why = arr(match?.why).slice(0, 2);
      const gaps = arr(match?.gaps).slice(0, 2);
      const status = state.jobStatuses[job.id];
      const sourceUrl = job.source?.url || "#";
      const sourceHost = host(sourceUrl);
      const sourceType = job.source?.source_type || "direct";
      const posted = postedLabel(job);
      const jobTitle = displayTitleForJob(job);
      const jobLocation = displayLocationForJob(job, jobTitle);

      const card = document.createElement("div");
      card.className = "glass rounded-xl p-4 card";
      card.innerHTML = `
        <div class="flex items-start justify-between gap-3">
          <div class="min-w-0 flex-1">
            <p class="text-xs font-semibold text-ember uppercase tracking-wider">${esc(job.company || "Unknown")}</p>
            <h3 class="text-base font-semibold text-ink mt-0.5 leading-tight">${esc(jobTitle || "Unknown")}</h3>
            <p class="text-xs text-slate-500 mt-1">
              ${esc(jobLocation || "Remote")}${job.level ? ` • ${esc(job.level)}` : ""}${posted ? ` • ${esc(posted)}` : ""}
            </p>
            <div class="flex flex-wrap gap-1.5 mt-1.5">
              <span class="px-1.5 py-0.5 rounded border text-[10px] ${
                sourceType === "aggregator" ? "bg-amber-50 text-amber-700 border-amber-200" : "bg-emerald-50 text-emerald-700 border-emerald-200"
              }">${esc(sourceType)}</span>
              ${sourceHost ? `<span class="px-1.5 py-0.5 rounded border text-[10px] bg-slate-50 text-slate-600 border-slate-200">${esc(sourceHost)}</span>` : ""}
            </div>
          </div>
          <div class="text-center flex-shrink-0">${scoreBadge(match)}</div>
        </div>

        <div class="grid grid-cols-2 gap-3 mt-3 text-xs">
          <div><p class="font-semibold text-emerald-600 mb-1">Why high</p>${why.map((x) => `<p class="text-slate-600">${esc(x)}</p>`).join("") || `<p class="text-slate-400">-</p>`}</div>
          <div><p class="font-semibold text-amber-600 mb-1">Why low</p>${gaps.map((x) => `<p class="text-slate-600">${esc(x)}</p>`).join("") || `<p class="text-slate-400">-</p>`}</div>
        </div>

        ${
          missMust.length || missStack.length
            ? `<div class="mt-2 text-[11px] text-slate-600">${missMust.length ? `Missing must-have: ${esc(missMust.join(", "))}<br/>` : ""}${
                missStack.length ? `Missing stack: ${esc(missStack.join(", "))}` : ""
              }</div>`
            : ""
        }

        <div class="flex flex-wrap items-center gap-2 mt-3 pt-3 border-t border-slate-100">
          <button data-save="${job.id}" class="px-3 py-1.5 rounded-lg text-xs font-semibold ${status === "saved" ? "bg-violet-100 text-violet-700" : "bg-slate-100 text-slate-600"}">${status === "saved" ? "Saved" : "Save"}</button>
          <button data-apply="${job.id}" class="px-3 py-1.5 rounded-lg text-xs font-semibold ${status === "applied" ? "bg-emerald-100 text-emerald-700" : "bg-slate-100 text-slate-600"}">${
        status === "applied" ? "Applied" : "Auto Apply"
      }</button>
          <a href="${esc(sourceUrl)}" target="_blank" class="px-3 py-1.5 rounded-lg bg-slate-100 text-slate-600 text-xs font-medium">View</a>
          <button data-tailor="${job.id}" class="px-3 py-1.5 rounded-lg btn-primary text-white text-xs font-semibold">Tailor</button>
          <button data-editor="${job.id}" class="px-3 py-1.5 rounded-lg bg-slate-100 text-slate-700 text-xs font-semibold">Editor</button>
        </div>
      `;
      el.jobsGrid.appendChild(card);
    });

  document.querySelectorAll("[data-save]").forEach((b) => (b.onclick = () => toggleStatus(b.dataset.save, "saved")));
  document.querySelectorAll("[data-apply]").forEach((b) => (b.onclick = () => runAutoApply(b.dataset.apply)));
  document.querySelectorAll("[data-tailor]").forEach((b) => (b.onclick = () => startTailor(b.dataset.tailor)));
  document.querySelectorAll("[data-editor]").forEach((b) => (b.onclick = () => openEditor(b.dataset.editor)));
}

async function loadMoreJobsIfNeeded() {
  if (state.jobsLoadingMore || !state.jobsHasMore) return;
  state.jobsLoadingMore = true;
  try {
    const next = await loadJobs(state.currentTab, true);
    if (arr(next).length) {
      await refreshMatches(next);
    } else {
      state.jobsHasMore = false;
    }
  } catch (err) {
    if (isRateLimited(err)) status("Rate limited while loading more jobs", "warn");
    else console.warn("loadMoreJobs failed:", err);
  } finally {
    state.jobsLoadingMore = false;
  }
}

async function onJobsGridScroll() {
  if (!el.jobsGrid) return;
  const nearBottom = el.jobsGrid.scrollTop + el.jobsGrid.clientHeight >= el.jobsGrid.scrollHeight - 160;
  if (nearBottom) await loadMoreJobsIfNeeded();
}

function stopAutoPull() {
  if (autoPullTimer) {
    clearInterval(autoPullTimer);
    autoPullTimer = null;
  }
}

function startAutoPull() {
  stopAutoPull();
  if (!state.autoPullEnabled) return;
  if (!state.activeProfileTag) return;
  if (!arr(state.lastSearchQueries).length && !arr(state.lastSearchUrls).length) return;

  autoPullTimer = setInterval(async () => {
    if (state.autoPullInProgress) return;
    if (state.currentTab !== "all") return;
    state.autoPullInProgress = true;
    try {
      await jfetch("/jobs/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          queries: arr(state.lastSearchQueries),
          urls: arr(state.lastSearchUrls),
          limit: 40,
          profile_tag: state.activeProfileTag,
        }),
      });
      const latest = await jfetch(`/jobs?limit=${state.jobsPageSize}&offset=0`);
      const latestArr = arr(latest);
      const prevIds = new Set(arr(state.jobs).map((j) => String(j.id)));
      const newJobs = latestArr.filter((j) => !prevIds.has(String(j.id)));
      if (newJobs.length) {
        state.jobs = mergeJobs(newJobs, state.jobs);
        state.jobsOffset += newJobs.length;
        updateCount();
        await refreshMatches(newJobs);
        renderJobs();
        status(`Added ${newJobs.length} new jobs`, "ok");
      }
    } catch (err) {
      if (isRateLimited(err)) {
        status("Rate limited; background pull slowed", "warn");
      } else {
        console.warn("background pull failed:", err);
      }
    } finally {
      state.autoPullInProgress = false;
    }
  }, 60000);
}

async function toggleStatus(jobId, target) {
  const cur = state.jobStatuses[jobId];
  if (cur === target) {
    await jfetch(`/jobs/${jobId}/status?profile_tag=${state.activeProfileTag}`, { method: "DELETE" });
    delete state.jobStatuses[jobId];
  } else {
    await jfetch(`/jobs/${jobId}/status?profile_tag=${state.activeProfileTag}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: target }),
    });
    state.jobStatuses[jobId] = target;
  }
  renderJobs();
}

async function fetchTailored(jobId) {
  if (tailoredByJob[jobId]) return tailoredByJob[jobId];
  const t = await jfetch(`/tailor/${state.activeProfileTag}/${jobId}`);
  tailoredByJob[jobId] = t;
  return t;
}

async function startTailor(jobId) {
  const job = state.jobs.find((j) => String(j.id) === String(jobId));
  if (!job) return;
  status("Tailoring...", "loading");
  loading("Tailoring resume...");
  try {
    tailoredByJob[jobId] = await jfetch(`/tailor?profile_tag=${state.activeProfileTag}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(job),
    });
    state.editors[jobId] = await jfetch(`/tailor/${state.activeProfileTag}/${jobId}/editor`);
    renderJobs();
    await openEditor(jobId);
    status("Tailored", "ok");
  } finally {
    doneLoading();
  }
}

function showModal() {
  el.tailorModal.classList.remove("hidden");
  el.tailorModal.classList.add("flex");
}

function hideModal() {
  el.tailorModal.classList.add("hidden");
  el.tailorModal.classList.remove("flex");
}

function showApplyRunModal(metaText, runId) {
  if (!el.applyRunModal) return;
  activeApplyRunId = runId || activeApplyRunId;
  el.applyRunMeta.textContent = metaText || "";
  el.applyRunId.textContent = runId ? `Run: ${runId}` : "";
  el.applyRunSteps.innerHTML = `<p class="text-slate-400">Preparing auto-apply run...</p>`;
  el.applyRunStatus.textContent = "Queued";
  el.applyRunStatus.className = "text-xs font-semibold px-2 py-1 rounded bg-amber-50 text-amber-700";
  if (el.applyRunActions) el.applyRunActions.classList.add("hidden");
  el.applyRunModal.classList.remove("hidden");
  el.applyRunModal.classList.add("flex");
}

function hideApplyRunModal() {
  if (!el.applyRunModal) return;
  activeApplyRunId = null;
  el.applyRunModal.classList.add("hidden");
  el.applyRunModal.classList.remove("flex");
}

function renderApplyRun(run) {
  if (!el.applyRunModal) return;
  const tones = {
    queued: ["Queued", "bg-amber-50 text-amber-700"],
    running: ["Running", "bg-blue-50 text-blue-700"],
    waiting_for_user: ["Action Required", "bg-orange-50 text-orange-700"],
    closing_browser: ["Closing Browser", "bg-violet-50 text-violet-700"],
    completed: ["Completed", "bg-emerald-50 text-emerald-700"],
    failed: ["Failed", "bg-red-50 text-red-700"],
  };
  const key = (run?.status || "queued").toLowerCase();
  const [label, toneClass] = tones[key] || ["Running", "bg-blue-50 text-blue-700"];
  el.applyRunStatus.textContent = label;
  el.applyRunStatus.className = `text-xs font-semibold px-2 py-1 rounded ${toneClass}`;
  el.applyRunId.textContent = run?.run_id ? `Run: ${run.run_id}` : "";

  const steps = arr(run?.steps);
  let html = "";
  if (steps.length) {
    html = steps
      .map(
        (step, idx) => `
        <div class="flex items-start gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2">
          <span class="text-[11px] text-slate-400 font-mono">${idx + 1}.</span>
          <p class="text-xs text-slate-700">${esc(step)}</p>
        </div>
      `
      )
      .join("");
  } else {
    html = `<p class="text-slate-400">Waiting for agent activity...</p>`;
  }

  if (arr(run?.missing_fields).length) {
    html += `<div class="rounded-lg border border-orange-200 bg-orange-50 px-3 py-2 text-xs text-orange-700">
      Missing required fields: ${esc(arr(run.missing_fields).slice(0, 8).join(", "))}
    </div>`;
  }

  if (run?.status === "completed" && run?.result?.message) {
    html += `<div class="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs text-emerald-700">${esc(
      run.result.message
    )}</div>`;
    if (run.result?.apply_url) {
      html += `<a href="${esc(run.result.apply_url)}" target="_blank" class="inline-flex mt-2 px-3 py-1.5 rounded-lg bg-slate-100 text-slate-700 text-xs font-semibold hover:bg-slate-200">Open application page</a>`;
    }
  }
  if (run?.status === "failed" && run?.error) {
    html += `<div class="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">${esc(run.error)}</div>`;
  }

  el.applyRunSteps.innerHTML = html;
  el.applyRunSteps.scrollTop = el.applyRunSteps.scrollHeight;

  if (el.applyRunActions) {
    const needsAction = run?.status === "waiting_for_user";
    el.applyRunActions.classList.toggle("hidden", !needsAction);
    if (needsAction && el.applyRunPrompt) {
      el.applyRunPrompt.textContent =
        run?.user_prompt || "Complete OTP/required fields and submit manually, then close the browser from here.";
    }
  }
}

async function pollApplyRun(runId) {
  const token = ++applyRunPollToken;
  activeApplyRunId = runId;
  while (token === applyRunPollToken) {
    const run = await jfetch(`/jobs/auto-apply/${runId}`);
    renderApplyRun(run);
    if (run.status === "completed" || run.status === "failed") return run;
    await new Promise((resolve) => setTimeout(resolve, 900));
  }
  return null;
}

async function sendApplyRunAction(manualSubmitted) {
  if (!activeApplyRunId) return;
  const action = manualSubmitted ? "close_and_mark_applied" : "close_browser";
  await jfetch(`/jobs/auto-apply/${activeApplyRunId}/action`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action }),
  });
  status("Closing browser...", "loading");
}

function localPreview(job, editor) {
  const d = editor?.draft || { summary: "", skills: [], bullets: [] };
  return `
    <div class="space-y-3">
      <p class="text-lg font-semibold text-ink">${esc(activeProfile()?.name || "Candidate")}</p>
      <p class="text-xs text-slate-500">${esc(job.title || "Role")} at ${esc(job.company || "Company")}</p>
      <div><p class="text-[11px] uppercase text-slate-500">Summary</p><p class="text-sm text-slate-700">${esc(d.summary)}</p></div>
      <div><p class="text-[11px] uppercase text-slate-500">Skills</p><p class="text-sm text-slate-700">${esc(arr(d.skills).join(", "))}</p></div>
      <div><p class="text-[11px] uppercase text-slate-500">Bullets</p><ul class="list-disc pl-5 text-sm text-slate-700">${arr(d.bullets)
        .map((b) => `<li>${esc(b)}</li>`)
        .join("")}</ul></div>
    </div>
  `;
}

function bindEditor(jobId) {
  const editor = state.editors[jobId];
  const job = state.jobs.find((j) => String(j.id) === String(jobId));
  const preview = document.getElementById("editor-live-preview");
  const summary = document.getElementById("editor-summary");
  const skills = document.getElementById("editor-skills");
  const bullets = document.querySelectorAll("[data-ebullet]");

  summary.oninput = () => {
    editor.draft.summary = summary.value;
    preview.innerHTML = localPreview(job, editor);
    scheduleSync(jobId, false);
  };
  skills.oninput = () => {
    editor.draft.skills = skills.value
      .split(",")
      .map((x) => x.trim())
      .filter(Boolean);
    preview.innerHTML = localPreview(job, editor);
    scheduleSync(jobId, false);
  };
  bullets.forEach((b) => {
    b.oninput = () => {
      const idx = Number(b.dataset.ebullet);
      editor.draft.bullets[idx] = b.value;
      preview.innerHTML = localPreview(job, editor);
      scheduleSync(jobId, false);
    };
  });

  document.querySelectorAll("[data-addkw]").forEach((btn) => {
    btn.onclick = () => {
      const kw = (btn.dataset.addkw || "").trim();
      if (!kw) return;
      if (!editor.draft.skills.some((s) => s.toLowerCase() === kw.toLowerCase())) editor.draft.skills.push(kw);
      openEditor(jobId);
      scheduleSync(jobId, false);
    };
  });

  document.getElementById("editor-save-btn").onclick = () => syncEditor(jobId, true);
  document.getElementById("editor-ai-btn").onclick = () => startTailor(jobId);
  document.getElementById("editor-apply-btn").onclick = async () => {
    await syncEditor(jobId, true);
    await runAutoApply(jobId);
  };
}

function scheduleSync(jobId, persist) {
  clearTimeout(syncTimers[jobId]);
  syncTimers[jobId] = setTimeout(() => syncEditor(jobId, persist), persist ? 120 : 700);
}

async function syncEditor(jobId, persist) {
  const editor = state.editors[jobId];
  if (!editor) return;
  const data = await jfetch(`/tailor/${state.activeProfileTag}/${jobId}/editor`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      summary: editor.draft.summary || "",
      skills: arr(editor.draft.skills),
      bullets: arr(editor.draft.bullets),
      persist: !!persist,
    }),
  });
  state.editors[jobId] = { ...data, draft: persist ? data.draft : editor.draft };
  if (persist) {
    tailoredByJob[jobId] = await fetchTailored(jobId);
    status("Draft saved", "ok");
    await openEditor(jobId);
    renderJobs();
  } else {
    const s = document.getElementById("editor-score");
    if (s) s.textContent = Math.round(data.fit_score || 0);
  }
}

async function openEditor(jobId) {
  const job = state.jobs.find((j) => String(j.id) === String(jobId));
  if (!job) return;
  status("Loading editor...", "loading");
  loading("Loading tailored editor...");
  try {
    const tailored = await fetchTailored(jobId);
    const editor = state.editors[jobId] || (await jfetch(`/tailor/${state.activeProfileTag}/${jobId}/editor`));
    state.editors[jobId] = editor;

    el.tailorModalMeta.textContent = `${job.title || "Role"} at ${job.company || "Company"}`;
    el.tailorDownloadTxt.href = tailored?.exports?.download_txt_url || editor?.exports?.download_txt_url || "#";
    el.tailorDownloadHtml.href = tailored?.exports?.download_html_url || editor?.exports?.download_html_url || "#";

    const d = editor.draft || { summary: "", skills: [], bullets: [] };
    const missing = arr(editor.missing_keywords).slice(0, 12);
    el.tailorModalBody.innerHTML = `
      <div class="grid lg:grid-cols-12 gap-4">
        <div class="lg:col-span-7 rounded-xl border border-slate-200 bg-white p-4 max-h-[68vh] overflow-y-auto">
          <div id="editor-live-preview">${localPreview(job, editor)}</div>
        </div>
        <div class="lg:col-span-5 rounded-xl border border-slate-200 bg-slate-50 p-4 max-h-[68vh] overflow-y-auto space-y-3">
          <div class="rounded-lg bg-white border border-slate-200 p-3">
            <p class="text-xs uppercase tracking-wider text-slate-500">Live Fit Score</p>
            <p class="text-2xl font-bold text-ink" id="editor-score">${Math.round(editor.fit_score || 0)}</p>
            <p class="text-[11px] text-slate-500">${esc(editor.confidence || "unknown")} confidence</p>
          </div>
          <div class="rounded-lg bg-white border border-slate-200 p-3">
            <label class="text-xs font-semibold text-slate-600">Summary</label>
            <textarea id="editor-summary" rows="4" class="mt-1 w-full border border-slate-200 rounded-lg px-2 py-1.5 text-xs">${esc(
              d.summary
            )}</textarea>
          </div>
          <div class="rounded-lg bg-white border border-slate-200 p-3">
            <label class="text-xs font-semibold text-slate-600">Skills (comma-separated)</label>
            <textarea id="editor-skills" rows="2" class="mt-1 w-full border border-slate-200 rounded-lg px-2 py-1.5 text-xs">${esc(
              arr(d.skills).join(", ")
            )}</textarea>
            <div class="mt-2 flex flex-wrap gap-1.5">${missing
              .map((kw) => `<button data-addkw="${esc(kw)}" class="px-2 py-1 rounded border border-amber-200 bg-amber-50 text-amber-700 text-[11px]">+ ${esc(kw)}</button>`)
              .join("")}</div>
          </div>
          <div class="rounded-lg bg-white border border-slate-200 p-3">
            <p class="text-xs font-semibold text-slate-600 mb-1">Bullets</p>
            <div class="space-y-2">${arr(d.bullets)
              .map(
                (b, i) =>
                  `<textarea data-ebullet="${i}" rows="3" class="w-full border border-slate-200 rounded px-2 py-1.5 text-xs">${esc(
                    b
                  )}</textarea>`
              )
              .join("")}</div>
          </div>
          <div class="flex flex-wrap gap-2">
            <button id="editor-ai-btn" class="px-3 py-1.5 rounded-lg bg-slate-100 text-slate-700 text-xs font-semibold">AI Rewrite</button>
            <button id="editor-save-btn" class="px-3 py-1.5 rounded-lg btn-primary text-white text-xs font-semibold">Save Draft</button>
            <button id="editor-apply-btn" class="px-3 py-1.5 rounded-lg bg-emerald-600 text-white text-xs font-semibold">Apply Now</button>
          </div>
        </div>
      </div>
    `;
    bindEditor(jobId);
    showModal();
    status("Ready", "ok");
  } finally {
    doneLoading();
  }
}

async function runAutoApply(jobId) {
  const job = state.jobs.find((j) => String(j.id) === String(jobId));
  const meta = `${job?.title || "Role"} at ${job?.company || "Company"} | Manual submit required`;
  showApplyRunModal(meta, "");
  status("Auto-fill agent running...", "loading");

  try {
    const runStart = await jfetch(`/jobs/${jobId}/auto-apply/start?profile_tag=${state.activeProfileTag}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ auto_submit: false, headless: false }),
    });

    showApplyRunModal(meta, runStart.run_id || "");
    const run = await pollApplyRun(runStart.run_id);

    if (run?.status === "completed" && run?.result) {
      const result = run.result;
      state.autoApplyResults[jobId] = result;
      if (result.submitted) state.jobStatuses[jobId] = "applied";
      else if (result.status === "ready_to_submit") state.jobStatuses[jobId] = "saved";

      await loadStatuses();
      if (result.status === "submitted") status("Application submitted", "ok");
      else if (result.status === "ready_to_submit") status("Manual review pending", "warn");
      else if (result.status === "unsupported") status("Auto-apply unsupported", "warn");
      else status("Auto-apply completed", "ok");
    } else if (run?.status === "failed") {
      status("Auto-apply failed", "warn");
    } else {
      status("Auto-apply stopped", "warn");
    }
  } catch (err) {
    if (el.applyRunSteps) {
      const msg = esc(err?.message || "Failed to start auto-apply run");
      el.applyRunSteps.innerHTML = `<div class="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">${msg}</div>`;
      if (el.applyRunStatus) {
        el.applyRunStatus.textContent = "Failed";
        el.applyRunStatus.className = "text-xs font-semibold px-2 py-1 rounded bg-red-50 text-red-700";
      }
    }
    status("Auto-apply failed", "warn");
  } finally {
    renderJobs();
  }
}

function setTab(tab) {
  state.currentTab = tab;
  ["all", "saved", "applied"].forEach((x) => {
    const b = document.getElementById(`tab-${x}`);
    if (!b) return;
    b.className = x === tab ? "tab-active pb-2 text-sm font-medium transition" : "tab-inactive pb-2 text-sm font-medium transition";
  });
}

function bindTabs() {
  ["all", "saved", "applied"].forEach((tab) => {
    const b = document.getElementById(`tab-${tab}`);
    if (!b) return;
    b.onclick = async () => {
      setTab(tab);
      await loadJobs(tab, false);
      await refreshMatches();
    };
  });
}

async function runSearch(e) {
  if (e) e.preventDefault();
  setTab("all");
  const q = (el.searchInput?.value || "").trim();
  const queries = q ? [q] : defaultQueries();
  await ingestJobs(queries, []);
}

function bind() {
  document.getElementById("refresh-profiles").onclick = async () => {
    await loadProfiles();
    await loadStatuses();
    await loadJobs(state.currentTab, false);
    await refreshMatches();
    if (!state.lastSearchQueries.length) state.lastSearchQueries = defaultQueries();
    startAutoPull();
  };
  if (el.activeProfileSelect) {
    el.activeProfileSelect.addEventListener("change", async (e) => {
      const tag = e.target.value;
      await switchActiveProfile(tag);
    });
  }
  el.uploadForm.addEventListener("submit", uploadResume);
  el.searchForm.addEventListener("submit", runSearch);
  el.saveApiKeyBtn.addEventListener("click", (e) => {
    e.preventDefault();
    saveApiKey();
  });
  el.tailorModalClose.addEventListener("click", hideModal);
  el.tailorModal.addEventListener("click", (e) => {
    if (e.target === el.tailorModal) hideModal();
  });
  if (el.applyRunClose) {
    el.applyRunClose.addEventListener("click", hideApplyRunModal);
  }
  if (el.applyRunModal) {
    el.applyRunModal.addEventListener("click", (e) => {
      if (e.target === el.applyRunModal) hideApplyRunModal();
    });
  }
  if (el.applyRunCloseOnly) {
    el.applyRunCloseOnly.addEventListener("click", async () => {
      await sendApplyRunAction(false);
    });
  }
  if (el.applyRunCloseSubmitted) {
    el.applyRunCloseSubmitted.addEventListener("click", async () => {
      await sendApplyRunAction(true);
    });
  }
  if (el.jobsGrid) {
    el.jobsGrid.addEventListener("scroll", () => {
      onJobsGridScroll();
    });
  }
}

async function bootstrap() {
  bind();
  bindTabs();
  await loadProfiles();
  await loadStatuses();
  await loadJobs(state.currentTab, false);
  await refreshMatches();
  if (!state.lastSearchQueries.length) state.lastSearchQueries = defaultQueries();
  startAutoPull();
}

bootstrap();
