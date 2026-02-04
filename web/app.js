const API_BASE = "";

const state = {
  profiles: [],
  activeProfileTag: null,
  jobs: [],
  matches: [],
  matchesByJobId: {},
  jobStatuses: {}, // {job_id: 'saved' | 'applied' | 'hidden'}
  currentTab: "all", // 'all' | 'saved' | 'applied'
};

// DOM Elements
const statusPill = document.getElementById("status-pill");
const profileSummaryEl = document.getElementById("profile-summary");
const targetsListEl = document.getElementById("targets-list");
const baseTagSelect = document.getElementById("base-tag");
const uploadForm = document.getElementById("upload-form");
const uploadResultEl = document.getElementById("upload-result");
const jobsGrid = document.getElementById("jobs-grid");
const searchForm = document.getElementById("search-form");
const searchInput = document.getElementById("search-query");
const apiKeyInput = document.getElementById("api-key-input");
const saveApiKeyBtn = document.getElementById("save-api-key");
const loadingOverlay = document.getElementById("loading-overlay");
const loadingText = document.getElementById("loading-text");
const jobsCountEl = document.getElementById("jobs-count");

let apiKey = localStorage.getItem("apiKey") || "";
if (apiKey && apiKeyInput) apiKeyInput.value = apiKey;

const tailoredByJob = {};

// Loading helpers
function showLoading(text = "Processing...") {
  if (loadingOverlay) {
    loadingOverlay.classList.remove("hidden");
    loadingOverlay.classList.add("flex");
    if (loadingText) loadingText.textContent = text;
  }
}

function hideLoading() {
  if (loadingOverlay) {
    loadingOverlay.classList.add("hidden");
    loadingOverlay.classList.remove("flex");
  }
}

function setStatus(text, tone = "neutral") {
  const colors = {
    neutral: { bg: "bg-white/10", dot: "bg-slate-400" },
    ok: { bg: "bg-emerald-500/20", dot: "bg-emerald-400" },
    warn: { bg: "bg-amber-500/20", dot: "bg-amber-400" },
    loading: { bg: "bg-blue-500/20", dot: "bg-blue-400" }
  };
  const c = colors[tone] || colors.neutral;
  statusPill.innerHTML = `<span class="w-2 h-2 rounded-full ${c.dot} ${tone === 'loading' ? 'animate-pulse' : ''}"></span>${text}`;
  statusPill.className = `inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold ${c.bg} border border-white/20`;
}

async function fetchJSON(path, options = {}) {
  const headers = options.headers ? { ...options.headers } : {};
  if (apiKey) headers["x-api-key"] = apiKey;
  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });
  if (!res.ok) throw new Error(await res.text() || res.statusText);
  return res.json();
}

function saveApiKey() {
  apiKey = apiKeyInput.value.trim();
  if (apiKey) {
    localStorage.setItem("apiKey", apiKey);
    setStatus("Saved", "ok");
  } else {
    localStorage.removeItem("apiKey");
    setStatus("Cleared", "ok");
  }
}

// Profile rendering
function renderProfileSummary() {
  const profile = state.profiles.find((p) => p.tag === state.activeProfileTag);
  if (!profile) {
    profileSummaryEl.innerHTML = `<p class='text-slate-400 text-xs'>No profile selected</p>`;
    return;
  }
  profileSummaryEl.innerHTML = `
    <div class="flex items-center gap-2 p-2 rounded-lg bg-gradient-to-r from-slate-50 to-violet-50">
      <div class="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center text-white font-semibold text-sm">
        ${profile.name.charAt(0)}
      </div>
      <div class="min-w-0">
        <p class="font-semibold text-ink text-sm truncate">${profile.name}</p>
        <p class="text-xs text-slate-500 font-mono">${profile.tag}</p>
      </div>
    </div>
  `;
  renderTargets(profile.targets || []);
}

function renderTargets(selectedTargets) {
  const allTargets = [
    { id: "data_scientist", label: "DS", icon: "📊" },
    { id: "machine_learning_engineer", label: "MLE", icon: "🤖" },
    { id: "ai_engineer", label: "AI", icon: "🧠" },
    { id: "software_engineer_ml_ai", label: "SWE", icon: "💻" }
  ];
  targetsListEl.innerHTML = "";
  allTargets.forEach((t) => {
    const isOn = selectedTargets.includes(t.id);
    const btn = document.createElement("button");
    btn.className = `px-2 py-1 rounded-full text-xs font-medium transition ${isOn ? "bg-ink text-white" : "bg-slate-100 text-slate-600 hover:bg-slate-200"
      }`;
    btn.innerHTML = `${t.icon} ${t.label}`;
    btn.onclick = () => toggleTarget(t.id);
    targetsListEl.appendChild(btn);
  });
}

function toggleTarget(targetId) {
  const profile = state.profiles.find((p) => p.tag === state.activeProfileTag);
  if (!profile) return;
  const nextTargets = new Set(profile.targets || []);
  nextTargets.has(targetId) ? nextTargets.delete(targetId) : nextTargets.add(targetId);
  profile.targets = Array.from(nextTargets);
  renderTargets(profile.targets);
  persistTargets(profile.tag, profile.targets);
}

async function persistTargets(tag, targets) {
  try {
    await fetchJSON(`/profiles/${tag}/targets`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ targets })
    });
    setStatus("Saved", "ok");
  } catch { setStatus("Error", "warn"); }
}

function populateBaseTagOptions() {
  baseTagSelect.innerHTML = `<option value="">Use resume only</option>`;
  state.profiles.forEach((p) => {
    const opt = document.createElement("option");
    opt.value = p.tag;
    opt.textContent = `${p.name} (${p.tag})`;
    baseTagSelect.appendChild(opt);
  });
}

function defaultQueriesFromProfile() {
  const profile = state.profiles.find((p) => p.tag === state.activeProfileTag);
  if (!profile) return [];
  const targets = profile.targets || [];
  const queries = targets.slice(0, 2).map((t) => `${t.replace(/_/g, " ")} remote US`);
  return queries.length ? queries : ["machine learning engineer remote US"];
}

async function loadProfiles() {
  setStatus("Loading...", "loading");
  try {
    state.profiles = await fetchJSON("/profiles");
    if (!state.activeProfileTag && state.profiles.length > 0) state.activeProfileTag = state.profiles[0].tag;
    renderProfileSummary();
    populateBaseTagOptions();
    await loadJobStatuses();
    setStatus("Ready", "ok");
  } catch { setStatus("Error", "warn"); }
}

async function loadJobStatuses() {
  if (!state.activeProfileTag) return;
  try {
    state.jobStatuses = await fetchJSON(`/jobs/statuses?profile_tag=${state.activeProfileTag}`);
  } catch { state.jobStatuses = {}; }
}

async function loadJobs() {
  try {
    state.jobs = await fetchJSON("/jobs?limit=30");
    updateJobsCount();
  } catch { console.error("Failed to load jobs"); }
}

async function loadSavedJobs() {
  if (!state.activeProfileTag) return;
  try {
    state.jobs = await fetchJSON(`/jobs/saved?profile_tag=${state.activeProfileTag}`);
    updateJobsCount();
  } catch { state.jobs = []; }
}

async function loadAppliedJobs() {
  if (!state.activeProfileTag) return;
  try {
    state.jobs = await fetchJSON(`/jobs/applied?profile_tag=${state.activeProfileTag}`);
    updateJobsCount();
  } catch { state.jobs = []; }
}

function updateJobsCount() {
  if (jobsCountEl) {
    const count = state.jobs.length;
    jobsCountEl.textContent = count > 0 ? `${count} jobs` : '';
  }
}

async function ingestJobs(queries = [], urls = []) {
  setStatus("Searching...", "loading");
  showLoading("Searching job boards...");
  try {
    const body = { queries, urls, limit: 25, profile_tag: state.activeProfileTag };
    state.jobs = await fetchJSON("/jobs/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    updateJobsCount();
    setStatus("Found", "ok");
    await refreshMatches();
  } catch { setStatus("Error", "warn"); }
  finally { hideLoading(); }
}

async function handleUpload(e) {
  e.preventDefault();
  const formData = new FormData(uploadForm);
  const file = formData.get("resume");
  if (!file || !file.name) {
    uploadResultEl.innerHTML = `<span class="text-amber-600">Select a PDF</span>`;
    return;
  }
  setStatus("Uploading...", "loading");
  showLoading("Parsing resume...");
  const payload = new FormData();
  payload.append("file", file);
  const baseTag = formData.get("baseTag");
  if (baseTag) payload.append("base_tag", baseTag);

  try {
    const headers = {};
    if (apiKey) headers["x-api-key"] = apiKey;
    const res = await fetch(`${API_BASE}/profiles/from-resume${baseTag ? `?base_tag=${baseTag}` : ""}`, {
      method: "POST", body: payload, headers
    });
    if (!res.ok) throw new Error(await res.text());
    const profile = await res.json();
    uploadResultEl.innerHTML = `<span class="text-emerald-600">✓ Created: ${profile.tag}</span>`;
    await loadProfiles();
    state.activeProfileTag = profile.tag;
    renderProfileSummary();
    setStatus("Ready", "ok");
  } catch (err) {
    uploadResultEl.innerHTML = `<span class="text-red-600">Error: ${err.message}</span>`;
    setStatus("Error", "warn");
  } finally { hideLoading(); }
}

function getScoreColor(score) {
  if (score >= 70) return "from-emerald-500 to-teal-500";
  if (score >= 50) return "from-amber-500 to-orange-500";
  if (score >= 30) return "from-slate-400 to-slate-500";
  return "from-red-400 to-red-500";
}

function renderJobs() {
  jobsGrid.innerHTML = "";
  updateJobsCount();

  // Filter by current tab
  let jobsToShow = state.jobs;
  if (state.currentTab === "saved") {
    jobsToShow = state.jobs.filter(j => state.jobStatuses[j.id] === "saved");
  } else if (state.currentTab === "applied") {
    jobsToShow = state.jobs.filter(j => state.jobStatuses[j.id] === "applied");
  } else {
    // All tab: exclude hidden
    jobsToShow = state.jobs.filter(j => state.jobStatuses[j.id] !== "hidden");
  }

  if (!jobsToShow.length) {
    jobsGrid.innerHTML = `
      <div class="text-center py-8">
        <p class="text-slate-400 text-sm">No jobs found</p>
        <p class="text-slate-400 text-xs mt-1">
          ${state.currentTab === "all" ? "Search for jobs to get started" : `No ${state.currentTab} jobs yet`}
        </p>
      </div>
    `;
    return;
  }

  // Sort by score
  const sortedJobs = [...jobsToShow].sort((a, b) => {
    const matchA = state.matchesByJobId[String(a.id)];
    const matchB = state.matchesByJobId[String(b.id)];
    return (matchB?.fit_score || 0) - (matchA?.fit_score || 0);
  });

  sortedJobs.forEach((job, idx) => {
    const match = state.matchesByJobId[String(job.id)];
    const score = match?.fit_score ?? null;
    const confidence = match?.confidence ?? null;
    const exactPct = match?.exact_match?.score != null ? Math.round(match.exact_match.score * 100) : null;
    const company = job.company || "Unknown";
    const title = job.title || "Unknown";
    const loc = job.location || "Remote";
    const level = job.level || "";
    const why = match?.why?.slice(0, 2) || [];
    const gaps = match?.gaps?.slice(0, 2) || [];
    const status = state.jobStatuses[job.id];
    const yearsReq = job.requirements?.years_experience_gate || "";
    const mustHave = job.requirements?.must_have || [];
    const niceHave = job.requirements?.nice_to_have || [];
    const stack = job.stack || [];
    const reqLines = [];
    if (yearsReq) reqLines.push(`Experience: ${yearsReq}`);
    if (mustHave.length) reqLines.push(`Must: ${mustHave.slice(0, 5).join(", ")}`);
    if (niceHave.length) reqLines.push(`Nice: ${niceHave.slice(0, 5).join(", ")}`);
    if (!reqLines.length && stack.length) reqLines.push(`Stack: ${stack.slice(0, 6).join(", ")}`);

    const card = document.createElement("div");
    card.className = "glass rounded-xl p-4 card job-card-enter";
    card.style.animationDelay = `${idx * 30}ms`;

    card.innerHTML = `
      <div class="flex items-start justify-between gap-3">
        <div class="min-w-0 flex-1">
          <p class="text-xs font-semibold text-ember uppercase tracking-wider">${company}</p>
          <h3 class="text-base font-semibold text-ink mt-0.5 leading-tight">${title}</h3>
          <p class="text-xs text-slate-500 mt-1">${loc}${level ? ` • ${level}` : ''}</p>
        </div>
        <div class="text-center flex-shrink-0">
          ${score !== null ? `
            <div class="w-12 h-12 rounded-full bg-gradient-to-br ${getScoreColor(score)} flex items-center justify-center shadow">
              <span class="text-lg font-bold text-white">${Math.round(score)}</span>
            </div>
            <p class="text-xs text-slate-500 mt-1">${confidence || ''}</p>
            ${exactPct !== null ? `<p class="text-[10px] text-slate-400 mt-0.5">exact ${exactPct}%</p>` : ''}
          ` : `<div class="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center"><span class="text-slate-400">-</span></div>`}
        </div>
      </div>
      
      ${why.length || gaps.length ? `
        <div class="grid grid-cols-2 gap-3 mt-3 text-xs">
          <div>
            <p class="font-semibold text-emerald-600 mb-1">✓ Strengths</p>
            ${why.length ? why.map(w => `<p class="text-slate-600 truncate">${w}</p>`).join('') : '<p class="text-slate-400 italic">—</p>'}
          </div>
          <div>
            <p class="font-semibold text-amber-600 mb-1">⚠ Gaps</p>
            ${gaps.length ? gaps.map(g => `<p class="text-slate-600 truncate">${g}</p>`).join('') : '<p class="text-slate-400 italic">—</p>'}
          </div>
        </div>
      ` : ''}

      <div class="mt-3 text-xs text-slate-600 space-y-1">
        ${reqLines.length ? reqLines.map(r => `<p class="truncate">${r}</p>`).join('') : '<p class="text-slate-400 italic">Requirements not parsed</p>'}
      </div>
      
      <div class="flex flex-wrap items-center gap-2 mt-3 pt-3 border-t border-slate-100">
        <button data-job="${job.id}" data-action="save" class="save-btn px-3 py-1.5 rounded-lg text-xs font-semibold transition ${status === 'saved' ? 'bg-violet-100 text-violet-700' : 'bg-slate-100 text-slate-600 hover:bg-violet-50'
      }">
          ${status === 'saved' ? '★ Saved' : '☆ Save'}
        </button>
        <button data-job="${job.id}" data-action="applied" class="applied-btn px-3 py-1.5 rounded-lg text-xs font-semibold transition ${status === 'applied' ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-100 text-slate-600 hover:bg-emerald-50'
      }">
          ${status === 'applied' ? '✓ Applied' : 'Mark Applied'}
        </button>
        <a href="${job.source?.url || '#'}" target="_blank" class="px-3 py-1.5 rounded-lg bg-slate-100 text-slate-600 text-xs font-medium hover:bg-slate-200 transition">
          View →
        </a>
        <button data-job="${job.id}" class="tailor-btn px-3 py-1.5 rounded-lg btn-primary text-white text-xs font-semibold">
          Tailor
        </button>
      </div>
    `;
    jobsGrid.appendChild(card);
  });

  // Bind action buttons
  document.querySelectorAll(".save-btn").forEach(btn => {
    btn.onclick = () => toggleJobStatus(btn.dataset.job, "saved");
  });
  document.querySelectorAll(".applied-btn").forEach(btn => {
    btn.onclick = () => toggleJobStatus(btn.dataset.job, "applied");
  });
  document.querySelectorAll(".tailor-btn").forEach(btn => {
    btn.onclick = () => startTailor(btn.dataset.job);
  });
}

async function toggleJobStatus(jobId, newStatus) {
  if (!state.activeProfileTag) return;

  const currentStatus = state.jobStatuses[jobId];

  try {
    if (currentStatus === newStatus) {
      // Remove status
      await fetchJSON(`/jobs/${jobId}/status?profile_tag=${state.activeProfileTag}`, { method: "DELETE" });
      delete state.jobStatuses[jobId];
    } else {
      // Set new status
      await fetchJSON(`/jobs/${jobId}/status?profile_tag=${state.activeProfileTag}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: newStatus })
      });
      state.jobStatuses[jobId] = newStatus;
    }
    renderJobs();
  } catch (err) {
    console.error("Failed to update status:", err);
  }
}

async function startTailor(jobId) {
  const job = state.jobs.find((j) => j.id === jobId);
  if (!job || !state.activeProfileTag) return;
  setStatus("Tailoring...", "loading");
  showLoading("Tailoring resume...");
  try {
    const tailored = await fetchJSON(`/tailor?profile_tag=${state.activeProfileTag}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(job)
    });
    tailoredByJob[jobId] = tailored;
    setStatus("Done", "ok");
    alert("Resume tailored! Check console for details.");
    console.log("Tailored resume:", tailored);
  } catch { setStatus("Error", "warn"); }
  finally { hideLoading(); }
}

async function searchAndMatch(e) {
  if (e) e.preventDefault();
  const q = searchInput.value.trim();
  const queries = q ? [q] : defaultQueriesFromProfile();
  state.currentTab = "all";
  updateTabs();
  await ingestJobs(queries, []);
}

async function refreshMatches() {
  if (!state.activeProfileTag || !state.jobs.length) {
    renderJobs();
    return;
  }
  setStatus("Scoring...", "loading");
  showLoading("Matching jobs...");
  try {
    // Prefer a single batch call to avoid rate-limit/fan-out issues.
    let matches = await fetchJSON(`/match/batch?profile_tag=${state.activeProfileTag}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state.jobs)
    });

    // Back-compat fallback for older servers.
    if (!Array.isArray(matches)) {
      const results = await Promise.allSettled(
        state.jobs.map((job) =>
          fetchJSON(`/match?profile_tag=${state.activeProfileTag}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(job)
          })
        )
      );
      matches = results.filter(r => r.status === "fulfilled").map(r => r.value);
    }

    state.matches = matches;
    state.matchesByJobId = {};
    state.matches.forEach((m) => {
      const id = m?.job_id ?? m?.jobId ?? null;
      if (id !== null && id !== undefined) state.matchesByJobId[String(id)] = m;
    });
    setStatus("Ready", "ok");
    renderJobs();
  } catch (err) {
    console.error("Match refresh failed:", err);
    state.matches = [];
    state.matchesByJobId = {};
    setStatus("Error", "warn");
    renderJobs();
  }
  finally { hideLoading(); }
}

// Tab handling
function updateTabs() {
  ["all", "saved", "applied"].forEach(tab => {
    const el = document.getElementById(`tab-${tab}`);
    if (el) {
      el.className = state.currentTab === tab ? "tab-active pb-2 text-sm font-medium transition" : "tab-inactive pb-2 text-sm font-medium transition";
    }
  });
}

function bindTabs() {
  ["all", "saved", "applied"].forEach(tab => {
    const el = document.getElementById(`tab-${tab}`);
    if (el) {
      el.onclick = async () => {
        state.currentTab = tab;
        updateTabs();
        if (tab === "all") {
          await loadJobs();
        } else if (tab === "saved") {
          await loadSavedJobs();
        } else if (tab === "applied") {
          await loadAppliedJobs();
        }
        await refreshMatches();
      };
    }
  });
}

function bindButtons() {
  document.getElementById("refresh-profiles").onclick = loadProfiles;
  uploadForm.addEventListener("submit", handleUpload);
  searchForm?.addEventListener("submit", searchAndMatch);
  saveApiKeyBtn?.addEventListener("click", (e) => { e.preventDefault(); saveApiKey(); });
}

async function bootstrap() {
  bindButtons();
  bindTabs();
  await loadProfiles();
  await loadJobs();
  await refreshMatches();
}

bootstrap();
