/* mcp-sim test runner. No framework, no build step.
 *
 * Every piece of data from the API (scenario text, transcripts carrying arbitrary web content,
 * model output) reaches the page through textContent / text nodes only: h() never parses
 * markup, and no attribute ever takes a URL built from data.
 */
'use strict';
(() => {
  const TOKEN = (document.querySelector('meta[name="mcpsim-token"]') || {}).content || '';
  const ROLES = ['planner', 'agent', 'user', 'observer', 'judge'];
  const MODEL_SPEC = /^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}$/;
  const POLL_MS = 1500;
  const STATUS = {
    passed: { glyph: '✓', label: 'Passed', dot: 'd-pass' },
    failed: { glyph: '✕', label: 'Failed', dot: 'd-fail' },
    partial: { glyph: '◐', label: 'Partial', dot: 'd-partial' },
    running: { glyph: '●', label: 'Running', dot: 'd-running' },
    queued: { glyph: '…', label: 'Queued', dot: 'd-running' },
    never: { glyph: '○', label: 'Never run', dot: 'd-never' },
    incomplete: { glyph: '–', label: 'Incomplete', dot: 'd-never' },
    error: { glyph: '!', label: 'Error', dot: 'd-error' },
    cancelled: { glyph: '–', label: 'Cancelled', dot: 'd-never' },
    done: { glyph: '✓', label: 'Done', dot: 'd-pass' },
  };

  // --- DOM helpers ------------------------------------------------------------------------
  const $ = (sel, root = document) => root.querySelector(sel);

  function h(tag, props, ...children) {
    const el = document.createElement(tag);
    if (props) {
      for (const [k, v] of Object.entries(props)) {
        if (v === null || v === undefined || v === false) continue;
        if (k === 'class') el.className = v;
        else if (k === 'text') el.textContent = String(v);
        else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
        else el.setAttribute(k, v === true ? '' : String(v));
      }
    }
    append(el, children);
    return el;
  }
  function append(el, children) {
    for (const c of children.flat(Infinity)) {
      if (c === null || c === undefined || c === false) continue;
      el.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return el;
  }
  function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }

  const SVG_NS = 'http://www.w3.org/2000/svg';
  const ICONS = {
    play: 'M6 4l10 6-10 6z',
    chev: 'M5 7l5 5 5-5',
  };
  function icon(name, size = 14) {
    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('viewBox', '0 0 20 20');
    svg.setAttribute('width', String(size));
    svg.setAttribute('height', String(size));
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('focusable', 'false');
    const path = document.createElementNS(SVG_NS, 'path');
    path.setAttribute('d', ICONS[name]);
    if (name === 'play') { path.setAttribute('fill', 'currentColor'); } else {
      path.setAttribute('fill', 'none');
      path.setAttribute('stroke', 'currentColor');
      path.setAttribute('stroke-width', '2');
      path.setAttribute('stroke-linecap', 'round');
      path.setAttribute('stroke-linejoin', 'round');
    }
    svg.append(path);
    return svg;
  }

  function statusPill(status, text) {
    const s = STATUS[status] || { glyph: '?', label: status || 'unknown' };
    return h('span', { class: `status st-${STATUS[status] ? status : 'unknown'}` },
      h('span', { class: 'glyph', 'aria-hidden': 'true', text: s.glyph }), text || s.label);
  }
  function verdictPill(passed) {
    if (passed === true) return statusPill('passed', 'Pass');
    if (passed === false) return statusPill('failed', 'Fail');
    return statusPill('never', 'Not graded');
  }
  function yesNo(value, missing = 'not graded') {
    if (value === true) return statusPill('passed', 'Yes');
    if (value === false) return statusPill('failed', 'No');
    return h('span', { class: 'muted', text: missing });
  }

  // --- formatting -------------------------------------------------------------------------
  const pct = (r) => (typeof r === 'number' ? `${Math.round(r * 1000) / 10}%` : '–');
  const money = (c) => (typeof c === 'number' ? `$${c.toFixed(c >= 1 ? 2 : 4)}` : '–');
  function dur(s) {
    if (typeof s !== 'number') return '–';
    if (s < 60) return `${s < 10 ? s.toFixed(1) : Math.round(s)} s`;
    const m = Math.floor(s / 60);
    if (m < 60) return `${m} min ${Math.round(s % 60)} s`;
    return `${Math.floor(m / 60)} h ${m % 60} min`;
  }
  function when(iso) {
    if (!iso) return '';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }
  function passK(pk) {
    if (!pk || typeof pk !== 'object' || !pk.k) return '–';
    return `${pk.all_passed ? '✓' : '✕'} pass^${pk.k}`;
  }
  function json(value) {
    try { return JSON.stringify(value, null, 2); } catch (e) { return String(value); }
  }
  function brief(value) {
    if (value === null) return 'null';
    if (Array.isArray(value)) return `${value.length} item${value.length === 1 ? '' : 's'}`;
    if (typeof value === 'object') return `{${Object.keys(value).length} keys}`;
    if (typeof value === 'string') return value.length > 40 ? `"${value.slice(0, 37)}…"` : `"${value}"`;
    return String(value);
  }
  function firstLine(text, max = 140) {
    const line = String(text || '').split('\n').find((l) => l.trim()) || '';
    return line.length > max ? `${line.slice(0, max - 1)}…` : line;
  }
  function argsInline(args) {
    if (!args || typeof args !== 'object') return '';
    return Object.entries(args).map(([k, v]) => `${k}=${typeof v === 'string' ? JSON.stringify(v.length > 30 ? `${v.slice(0, 29)}…` : v) : brief(v)}`).join(', ');
  }
  const norm = (s) => String(s || '').toLowerCase().replace(/\s+/g, ' ').trim();

  // --- persistence (per-browser conveniences only) -----------------------------------------
  function load(key, fallback) {
    try { const raw = localStorage.getItem(key); return raw ? JSON.parse(raw) : fallback; } catch (e) { return fallback; }
  }
  function save(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* storage unavailable */ }
  }

  // --- API --------------------------------------------------------------------------------
  async function api(path, body) {
    const init = { headers: { Accept: 'application/json' }, credentials: 'same-origin' };
    if (body !== undefined) {
      init.method = 'POST';
      init.headers['Content-Type'] = 'application/json';
      init.headers['X-MCPSim-Token'] = TOKEN;
      init.body = JSON.stringify(body);
    }
    const res = await fetch(path, init);
    let data = null;
    try { data = await res.json(); } catch (e) { data = null; }
    if (!res.ok) {
      const stale = res.status === 403 && data && data.code === 'bad_token';
      const message = stale
        ? 'This page belongs to an earlier start of the server (each start has a new token). Reload the page and try again.'
        : (data && data.error) || `${res.status} ${res.statusText}`;
      const err = new Error(message);
      err.status = res.status;
      err.data = data;
      throw err;
    }
    return data;
  }
  const enc = encodeURIComponent;

  // --- state ------------------------------------------------------------------------------
  const state = {
    config: null,
    list: null,
    filter: { q: '', status: 'all' },
    sel: { name: null, run: null, stem: null },
    scenario: null,
    runs: null,
    run: null,
    transcript: null,
    jobs: new Map(),
    opts: load('mcpsim.ui.opts', {}),
    collapsed: load('mcpsim.ui.collapsed', {}),
    seq: 0,
    showAllRuns: false,
  };

  function announce(text) {
    const el = $('#announcer');
    el.textContent = '';
    window.setTimeout(() => { el.textContent = text; }, 30);
  }
  let toastTimer = 0;
  function toast(text, ok = false) {
    const el = $('#toast');
    el.textContent = text;
    el.classList.toggle('ok', ok);
    el.hidden = false;
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => { el.hidden = true; }, ok ? 3500 : 7000);
  }

  // --- routing: #/<scenario>/<run id>/<transcript stem> -------------------------------------
  function parseHash() {
    const parts = location.hash.replace(/^#\/?/, '').split('/').filter(Boolean).map((p) => {
      try { return decodeURIComponent(p); } catch (e) { return ''; }
    });
    return { name: parts[0] || null, run: parts[1] || null, stem: parts[2] || null };
  }
  function setHash(sel, push) {
    const parts = [sel.name, sel.run, sel.stem].filter(Boolean).map(enc);
    const hash = parts.length ? `#/${parts.join('/')}` : '#';
    if (location.hash === hash) return;
    if (push) history.pushState(null, '', hash); else history.replaceState(null, '', hash);
  }
  function go(name, run = null, stem = null, push = true) {
    setHash({ name, run, stem }, push);
    return route();
  }
  function focusDetailIfNarrow() {
    // Below 900px the list and the detail are separate screens: move focus with the view.
    if (!window.matchMedia('(max-width: 900px)').matches) return;
    const title = $('#detail-title');
    if (title) { title.setAttribute('tabindex', '-1'); title.focus(); }
  }

  async function route() {
    const want = parseHash();
    document.body.classList.toggle('show-detail', Boolean(want.name));
    if (!want.name) {
      state.sel = { name: null, run: null, stem: null };
      renderEmpty();
      renderList();
      return;
    }
    const seq = ++state.seq;
    if (want.name !== state.sel.name || !state.scenario) {
      state.sel = { name: want.name, run: null, stem: null };
      state.scenario = null; state.runs = null; state.run = null; state.transcript = null;
      state.showAllRuns = false;
      renderList();
      renderLoading();
      try {
        const [scenario, runs] = await Promise.all([
          api(`/api/scenarios/${enc(want.name)}`),
          api(`/api/scenarios/${enc(want.name)}/runs`),
        ]);
        if (seq !== state.seq) return;
        state.scenario = scenario;
        state.runs = runs.runs;
      } catch (e) {
        if (seq !== state.seq) return;
        renderError(e.status === 404 ? `No scenario named “${want.name}”.` : e.message);
        return;
      }
      renderDetail();
    }
    await selectRun(want.run, want.stem, seq);
  }

  function pickRun(requested) {
    const runs = state.runs || [];
    if (requested && runs.some((r) => r.run_id === requested)) return requested;
    const judged = runs.find((r) => r.runs > 0);
    return (judged || runs[0] || {}).run_id || null;
  }

  async function selectRun(requested, stem, seq) {
    const runId = pickRun(requested);
    if (!runId) {
      state.sel.run = null; state.run = null; state.transcript = null;
      renderRunParts();
      return;
    }
    if (runId !== state.sel.run || !state.run) {
      state.sel.run = runId; state.sel.stem = null; state.run = null; state.transcript = null;
      renderRunParts(true);
      try {
        const run = await api(`/api/runs/${enc(state.sel.name)}/${enc(runId)}`);
        if (seq !== state.seq) return;
        state.run = run;
      } catch (e) {
        if (seq !== state.seq) return;
        toast(`Could not load run ${runId}: ${e.message}`);
        return;
      }
    }
    await selectTranscript(stem, seq);
  }

  function pickStem(requested) {
    const rows = (state.run && state.run.transcripts) || [];
    if (requested && rows.some((t) => t.stem === requested)) return requested;
    const failing = rows.find((t) => t.passed === false && t.file);
    return ((failing || rows.find((t) => t.file) || rows[0]) || {}).stem || null;
  }

  async function selectTranscript(requested, seq) {
    const stem = pickStem(requested);
    state.sel.stem = stem;
    setHash(state.sel, false);
    const row = ((state.run && state.run.transcripts) || []).find((t) => t.stem === stem);
    if (!row || !row.file) {
      state.transcript = null;
      renderRunParts();
      return;
    }
    if (!state.transcript || state.transcript.stem !== stem) {
      state.transcript = null;
      renderRunParts(false, true);
      try {
        const data = await api(`/api/runs/${enc(state.sel.name)}/${enc(state.sel.run)}/transcripts/${enc(row.file)}`);
        if (seq !== state.seq) return;
        state.transcript = data;
      } catch (e) {
        if (seq !== state.seq) return;
        toast(`Could not load transcript: ${e.message}`);
      }
    }
    renderRunParts();
  }

  // --- scenario list ----------------------------------------------------------------------
  function matches(s) {
    const f = state.filter;
    if (f.status !== 'all' && s.status !== f.status) return false;
    if (!f.q) return true;
    const hay = norm([s.name, s.title, s.category, s.user_instructions].join(' '));
    return norm(f.q).split(' ').every((term) => hay.includes(term));
  }
  function filtered() { return ((state.list && state.list.scenarios) || []).filter(matches); }
  function filterActive() { return Boolean(state.filter.q) || state.filter.status !== 'all'; }

  async function loadList() {
    try {
      state.list = await api('/api/scenarios');
    } catch (e) {
      toast(`Could not load scenarios: ${e.message}`);
      return;
    }
    renderList();
    for (const w of state.list.warnings || []) console.warn('mcpsim:', w);
  }

  function countsEl(counts) {
    const order = ['passed', 'failed', 'partial', 'running', 'never'];
    const label = order.filter((k) => counts[k]).map((k) => `${counts[k]} ${STATUS[k].label.toLowerCase()}`).join(', ');
    return h('span', { class: 'counts', 'aria-label': label || 'no scenarios' },
      order.filter((k) => counts[k]).map((k) => h('span', { class: `c-${k}`, 'aria-hidden': 'true' },
        STATUS[k].glyph, String(counts[k]))));
  }

  function renderList() {
    const host = $('#scenario-list');
    const runBtn = $('#run-filtered');
    if (!state.list) return;
    const rows = filtered();
    const runnable = rows.filter((s) => !s.error && s.status !== 'running');
    runBtn.textContent = filterActive() ? `Run ${runnable.length} shown` : 'Run all';
    runBtn.disabled = runnable.length === 0;

    const total = { passed: 0, failed: 0, partial: 0, running: 0, never: 0 };
    rows.forEach((s) => { total[s.status] = (total[s.status] || 0) + 1; });
    const all = state.list.scenarios.length;
    $('#list-summary').textContent = `${rows.length === all ? `${all}` : `${rows.length} of ${all}`} scenario${all === 1 ? '' : 's'}` +
      (rows.length ? ` · ${Object.entries(total).filter(([, n]) => n).map(([k, n]) => `${n} ${STATUS[k].label.toLowerCase()}`).join(' · ')}` : '');

    clear(host);
    if (!rows.length) {
      host.append(h('p', { class: 'muted pad', text: all ? 'No scenario matches the search.' : `No scenario files found in ${(state.config && state.config.scenario_sources || []).join(', ') || 'the configured sources'}.` }));
      return;
    }
    const groups = new Map();
    for (const s of rows) {
      if (!groups.has(s.category)) groups.set(s.category, []);
      groups.get(s.category).push(s);
    }
    let gi = 0;
    for (const [category, items] of [...groups.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
      const id = `group-${gi++}`;
      const collapsed = Boolean(state.collapsed[category]) && !filterActive();
      const counts = { passed: 0, failed: 0, partial: 0, running: 0, never: 0 };
      items.forEach((s) => { counts[s.status] = (counts[s.status] || 0) + 1; });
      const list = h('ul', { class: 'rows', id, role: 'list' },
        items.map((s) => scenarioRow(s)));
      list.hidden = collapsed;
      const countText = Object.entries(counts).filter(([, n]) => n).map(([k, n]) => `${n} ${STATUS[k].label.toLowerCase()}`).join(', ');
      const head = h('button', {
        class: 'group-head', type: 'button', 'aria-expanded': String(!collapsed), 'aria-controls': id,
        'aria-label': `${category}: ${items.length} scenario${items.length === 1 ? '' : 's'}${countText ? `, ${countText}` : ''}`,
        onclick: () => {
          state.collapsed[category] = !state.collapsed[category];
          save('mcpsim.ui.collapsed', state.collapsed);
          renderList();
        },
      }, h('span', { class: 'chev' }, icon('chev', 12)),
      h('span', { class: 'group-name', text: category }),
      h('span', { class: 'muted small', text: String(items.length) }),
      countsEl(counts));
      host.append(h('div', { class: 'group' }, head, list));
    }
  }

  function scenarioRow(s) {
    const st = STATUS[s.status] || STATUS.never;
    const selected = s.name === state.sel.name;
    const rate = s.error ? 'invalid' : (s.last_run ? `${s.last_run.passed}/${s.last_run.runs}` : '');
    const said = [s.error ? 'invalid file' : st.label.toLowerCase()];
    if (!s.error && s.last_run && s.last_run.runs) said.push(`${s.last_run.passed} of ${s.last_run.runs} runs passed`);
    if (!s.error && s.pass_k && s.pass_k.k) said.push(`pass^${s.pass_k.k} ${s.pass_k.all_passed ? 'held' : 'not held'}`);
    const main = h('button', {
      class: 'row-main', type: 'button', 'data-name': s.name,
      'aria-current': selected ? 'true' : null,
      'aria-label': `${s.title} (${s.name}): ${said.join(', ')}`,
      onclick: () => { go(s.name).then(focusDetailIfNarrow); },
    },
    h('span', { class: `dot ${s.error ? 'd-error' : st.dot}`, 'aria-hidden': 'true' }),
    h('span', { class: 'row-title', text: s.title }),
    h('span', { class: 'row-rate', text: rate }),
    h('span', { class: 'row-sub' }, h('span', { class: 'sr-only', text: `${s.error ? 'Invalid file' : st.label}. ` }), s.name,
      s.pass_k ? ` · ${passK(s.pass_k)}` : ''));
    const run = h('button', {
      class: 'row-run', type: 'button', 'aria-label': `Run ${s.title}`, title: `Run ${s.title}`,
      disabled: Boolean(s.error) || s.status === 'running',
      onclick: () => startRun([s.name], s.title),
    }, icon('play', 13));
    return h('li', { class: 'row' }, main, run);
  }

  function onListKey(ev) {
    if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(ev.key)) return;
    const buttons = [...document.querySelectorAll('#scenario-list .row-main')].filter((b) => b.offsetParent !== null);
    if (!buttons.length) return;
    const i = buttons.indexOf(document.activeElement);
    let next = i;
    if (ev.key === 'ArrowDown') next = i < 0 ? 0 : Math.min(buttons.length - 1, i + 1);
    if (ev.key === 'ArrowUp') next = i < 0 ? 0 : Math.max(0, i - 1);
    if (ev.key === 'Home') next = 0;
    if (ev.key === 'End') next = buttons.length - 1;
    ev.preventDefault();
    buttons[next].focus();
  }

  // --- detail -----------------------------------------------------------------------------
  function renderEmpty() {
    const d = clear($('#detail'));
    d.append(h('div', { class: 'empty-state' },
      h('h1', { text: 'Pick a scenario' }),
      h('p', { class: 'muted' }, 'Choose a scenario to see its simulated user, expected behavior, run history and transcripts. Press ',
        h('kbd', { text: '/' }), ' to search.')));
  }
  function renderLoading() {
    clear($('#detail')).append(h('p', { class: 'muted', text: 'Loading…' }));
  }
  function renderError(message) {
    const d = clear($('#detail'));
    d.append(backButton(), h('div', { class: 'card error-box' }, h('h2', { text: 'Not available' }), h('p', { text: message })));
  }
  function backButton() {
    return h('button', {
      class: 'btn back', type: 'button',
      onclick: () => {
        const name = state.sel.name;
        go(null).then(() => {
          const row = name && [...document.querySelectorAll('#scenario-list .row-main')].find((b) => b.dataset.name === name);
          if (row) row.focus();
        });
      },
    }, '← All scenarios');
  }

  function specView() {
    // What was run (scenario.json in the run directory) when a run is selected, else the file.
    return (state.run && state.run.scenario) || (state.scenario && state.scenario.scenario) || {};
  }

  function renderDetail() {
    const data = state.scenario;
    const s = data.scenario;
    const d = clear($('#detail'));
    const hasRuns = (state.runs || []).length > 0;
    const rs = data.run_settings || {};
    const runBtn = h('button', {
      class: 'btn btn-primary', type: 'button', disabled: Boolean(s.error) || data.status === 'running',
      onclick: () => startRun([s.name], s.title),
    }, icon('play', 12), hasRuns ? 'Re-run' : 'Run');
    d.append(
      backButton(),
      h('header', { class: 'd-head' },
        h('div', null,
          h('p', { class: 'eyebrow', text: s.category }),
          h('h1', { id: 'detail-title', text: s.title }),
          h('p', { class: 'd-meta' },
            h('span', { class: 'mono', text: s.name }),
            h('span', { class: 'mono', text: s.file }),
            s.server ? h('span', { text: `server: ${s.server}` }) : null,
            rs.repeat ? h('span', { text: `repeat ${rs.repeat.value}` }) : null)),
        h('div', { class: 'd-actions' }, statusPill(s.error ? 'error' : data.status, s.error ? 'Invalid file' : null), runBtn)),
    );
    if (s.error) {
      d.append(h('section', { class: 'card error-box', 'aria-labelledby': 'err-title' },
        h('h2', { id: 'err-title', text: 'This scenario file does not validate' }), h('pre', { text: s.error })));
    }
    for (const w of data.warnings || []) d.append(h('p', { class: 'note', text: w }));
    d.append(
      h('section', { class: 'card', 'aria-labelledby': 'hist-title' },
        h('h2', { id: 'hist-title' }, 'Run history', h('span', { class: 'aside', id: 'hist-count' })),
        h('div', { id: 'd-history' }),
        h('div', { id: 'd-runsum' })),
      h('div', { class: 'd-grid' },
        h('div', { class: 'd-col' },
          h('section', { class: 'card', 'aria-labelledby': 'user-title', id: 'd-user' }),
          h('section', { class: 'card', 'aria-labelledby': 'eb-title', id: 'd-expected' }),
          h('section', { class: 'card', 'aria-labelledby': 'judge-title', id: 'd-judge' }),
          h('section', { class: 'card', 'aria-labelledby': 'plan-title', id: 'd-plan' })),
        // One column below 1440px: these follow the cards above, in this DOM (and focus) order.
        h('div', { class: 'd-col d-col-right' },
          h('section', { class: 'card', 'aria-labelledby': 'sims-title', id: 'd-sims' }),
          h('section', { class: 'card', 'aria-labelledby': 'convo-title', id: 'd-convo' }),
          h('section', { class: 'card', 'aria-labelledby': 'src-title', id: 'd-source' }))),
    );
    renderRunParts();
  }

  function renderRunParts(runLoading = false, transcriptLoading = false) {
    if (!state.scenario || !$('#d-history')) return;
    renderHistory();
    renderRunSummary(runLoading);
    renderUser();
    renderExpected();
    renderJudge(transcriptLoading);
    renderPlan();
    renderSource();
    renderSims();
    renderConversation(transcriptLoading || runLoading);
  }

  function renderHistory() {
    const host = clear($('#d-history'));
    const runs = state.runs || [];
    $('#hist-count').textContent = runs.length ? `${runs.length} run director${runs.length === 1 ? 'y' : 'ies'}` : '';
    if (!runs.length) {
      host.append(h('p', { class: 'muted', text: 'Not run yet. Run it to get a plan, transcripts and verdicts.' }));
      return;
    }
    const shown = state.showAllRuns ? runs : runs.slice(0, 12);
    const strip = h('div', { class: 'history', role: 'group', 'aria-label': 'Runs, newest first' },
      shown.map((r) => h('button', {
        class: 'hist', type: 'button', 'aria-pressed': String(r.run_id === state.sel.run),
        onclick: () => { setHash({ name: state.sel.name, run: r.run_id }, true); selectRun(r.run_id, null, state.seq); },
      },
      h('span', { class: 'when', text: when(r.started_at) || r.run_id }),
      h('span', { class: 'line' }, statusPill(r.status), r.status === 'running' ? 'in progress' : r.runs ? `${r.passed}/${r.runs}` : 'no verdicts'),
      h('span', { class: 'line' }, passK(r.pass_k), ' · ', money(r.cost_usd), ' · ', dur(r.duration_s)))));
    host.append(strip);
    if (runs.length > shown.length) {
      host.append(h('button', { class: 'btn btn-sm', type: 'button', onclick: () => { state.showAllRuns = true; renderHistory(); } },
        `Show all ${runs.length}`));
    }
  }

  function metric(k, v) { return h('div', { class: 'metric' }, h('span', { class: 'k', text: k }), h('span', { class: 'v' }, v)); }

  function renderRunSummary(loading) {
    const host = clear($('#d-runsum'));
    if (!state.sel.run) return;
    if (loading || !state.run) { host.append(h('p', { class: 'muted small', text: 'Loading run…' })); return; }
    const s = state.run.summary;
    host.append(h('div', { class: 'metrics' },
      metric('Run', h('span', { class: 'mono', text: s.run_id })),
      metric('Status', statusPill(s.status)),
      metric('Passed', s.runs ? `${s.passed} / ${s.runs}` : '–'),
      metric('Pass rate', pct(s.pass_rate)),
      metric('pass^k', passK(s.pass_k)),
      metric('Mean score', typeof s.mean_score === 'number' ? s.mean_score.toFixed(2) : '–'),
      metric('Cost', money(s.cost_usd)),
      metric('Duration', dur(s.duration_s)),
      metric('Judge', (s.judge_models || []).join(', ') || '–')));
    const models = s.models || {};
    const effective = (state.scenario && state.scenario.models) || {};
    const line = h('p', { class: 'models-line' }, h('span', { text: 'Models as run:' }));
    for (const role of ROLES) {
      const spec = models[role] || (role === 'user' || role === 'observer' ? models.agent : null);
      if (spec) line.append(h('span', null, `${role} `, h('b', { text: spec })));
    }
    if (line.childNodes.length === 1) {
      for (const role of ROLES) if (effective[role]) line.append(h('span', null, `${role} `, h('b', { text: effective[role].spec })));
      line.firstChild.textContent = 'Models (resolved now):';
    }
    host.append(line);
    if (s.dry_run) host.append(h('p', { class: 'note', text: 'Dry run: no model was called; verdicts come from the matcher alone.' }));
  }

  function renderUser() {
    const host = clear($('#d-user'));
    const v = specView();
    host.append(h('h2', { id: 'user-title' }, 'Simulated user',
      v.user_instructions_derived ? h('span', { class: 'aside', text: 'built from role and goal' }) : null));
    host.append(h('p', { class: 'prose', text: v.user_instructions || '–' }));
    const ctx = v.context || {};
    const cells = [['Device', ctx.device], ['Location', ctx.location], ['Language', ctx.language]];
    const details = Object.entries(ctx.details || {});
    if (!cells.some(([, val]) => val) && !details.length) {
      host.append(h('p', { class: 'note', text: 'No context (device, location, language) in this scenario.' }));
    } else {
      host.append(h('div', { class: 'context-grid', role: 'list', 'aria-label': 'Context' },
        cells.map(([k, val]) => h('div', { class: 'ctx', role: 'listitem' }, h('span', { class: 'k', text: k }), h('span', { class: val ? '' : 'muted', text: val || 'not set' }))),
        details.map(([k, val]) => h('div', { class: 'ctx', role: 'listitem' }, h('span', { class: 'k', text: k }), h('span', { text: val })))));
      host.append(h('p', { class: 'note', text: ctx.agent_visible ? 'The agent sees this context too.' : 'Only the simulated user sees this context.' }));
    }
    if (v.agent_skill || v.agent_notes || v.agent_skill_text) {
      const sop = v.agent_skill_name || v.agent_skill || 'inline';
      host.append(h('dl', { class: 'kv small sop' },
        v.agent_skill || v.agent_skill_text ? [h('dt', { text: 'Agent SOP' }), h('dd', null,
          h('span', { class: 'mono', text: sop }),
          v.agent_skill && v.agent_skill !== sop ? h('span', { class: 'muted mono', text: ` (${v.agent_skill})` }) : null,
          v.agent_skill_text ? h('details', null, h('summary', { class: 'small', text: 'Procedure the agent ran on' }),
            v.agent_skill_path ? h('p', { class: 'muted small mono', text: v.agent_skill_path }) : null,
            h('pre', { text: v.agent_skill_text })) : null)] : null,
        v.agent_notes ? [h('dt', { text: 'Agent notes' }), h('dd', { class: 'prose', text: v.agent_notes })] : null));
    }
  }

  function selectedVerdict() {
    if (state.transcript && state.transcript.verdict) return state.transcript.verdict;
    if (state.run && state.sel.stem) return (state.run.verdicts || {})[state.sel.stem] || null;
    return null;
  }

  function findItem(checklist, item, index, total) {
    const n = norm(item);
    const exact = checklist.find((c) => norm(c.item) === n);
    if (exact) return exact;
    const loose = checklist.find((c) => norm(c.item).includes(n) || n.includes(norm(c.item)));
    if (loose && norm(loose.item)) return loose;
    return checklist.length === total ? checklist[index] : null;
  }

  function renderExpected() {
    const host = clear($('#d-expected'));
    const v = specView();
    const items = v.expected_behavior || [];
    const verdict = selectedVerdict();
    const checklist = (verdict && Array.isArray(verdict.checklist)) ? verdict.checklist : [];
    const all = state.run ? Object.values(state.run.verdicts || {}) : [];
    host.append(h('h2', { id: 'eb-title' }, 'Expected behavior',
      h('span', { class: 'aside', text: verdict ? `verdict for ${state.sel.stem}` : (state.run ? 'no verdict for this transcript' : 'not run') })));
    if (!items.length) { host.append(h('p', { class: 'muted', text: 'No expected behavior listed.' })); return; }
    const used = new Set();
    const list = h('ol', { class: 'checklist' });
    items.forEach((item, i) => {
      const hit = findItem(checklist, item, i, items.length);
      if (hit) used.add(hit);
      let agg = null;
      if (all.length) {
        const graded = all.map((vd) => findItem(Array.isArray(vd.checklist) ? vd.checklist : [], item, i, items.length)).filter(Boolean);
        if (graded.length) agg = `passed in ${graded.filter((c) => c.passed).length} of ${graded.length} graded run${graded.length === 1 ? '' : 's'}`;
      }
      list.append(h('li', { class: 'check' },
        h('span', { class: 'verdict' }, verdictPill(hit ? hit.passed : null)),
        h('span', { class: 'item', text: item }),
        agg ? h('span', { class: 'agg', text: agg }) : null,
        hit && hit.evidence ? h('blockquote', { class: `evidence ${hit.passed ? 'v-pass' : 'v-fail'}`, text: hit.evidence }) : null));
    });
    host.append(list);
    // The judge grades one standing item after the scenario's own: honesty (judge.HONESTY_ITEM).
    const honestyText = norm((state.config && state.config.honesty_item) || '');
    const left = checklist.filter((c) => !used.has(c));
    const standing = left.filter((c) => honestyText && norm(c.item) === honestyText);
    const extra = left.filter((c) => !standing.includes(c));
    const checkRow = (c) => h('li', { class: 'check' },
      h('span', { class: 'verdict' }, verdictPill(c.passed)), h('span', { class: 'item', text: c.item }),
      c.evidence ? h('blockquote', { class: `evidence ${c.passed ? 'v-pass' : 'v-fail'}`, text: c.evidence }) : null);
    if (standing.length) {
      host.append(h('h3', { class: 'small muted', text: 'Standing check (every scenario)' }));
      host.append(h('ol', { class: 'checklist' }, standing.map(checkRow)));
    }
    if (extra.length) {
      host.append(h('h3', { class: 'small muted', text: 'Other items the judge graded' }));
      host.append(h('ol', { class: 'checklist' }, extra.map(checkRow)));
    }
    if (v.expected_behavior_derived) host.append(h('p', { class: 'note', text: 'The scenario lists no expected_behavior; its instructions are graded instead.' }));
    if (verdict && !checklist.length) host.append(h('p', { class: 'note', text: `No checklist in this verdict (judge: ${verdict.judge_model || 'unknown'}).` }));
  }

  function renderJudge(loading) {
    const host = clear($('#d-judge'));
    host.append(h('h2', { id: 'judge-title' }, 'Judge', h('span', { class: 'aside', text: state.sel.stem || '' })));
    const verdict = selectedVerdict();
    const facts = state.transcript ? state.transcript.facts : null;
    if (!verdict) {
      host.append(h('p', { class: 'muted', text: loading ? 'Loading…' : (state.run ? 'This transcript has no verdict.' : 'Select a run to see verdicts.') }));
      if (!facts) return;
    }
    if (verdict) {
      host.append(h('dl', { class: 'kv' },
        h('dt', { text: 'Verdict' }), h('dd', null, verdictPill(verdict.passed), ` score ${typeof verdict.score === 'number' ? verdict.score.toFixed(2) : '–'}`),
        h('dt', { text: 'Goal achieved' }), h('dd', null, yesNo(verdict.goal_achieved)),
        h('dt', { text: 'SOP followed' }), h('dd', null, yesNo(verdict.sop_followed, specView().agent_skill || specView().agent_skill_text ? 'not graded' : 'no SOP in this scenario')),
        h('dt', { text: 'Judge' }), h('dd', { class: 'mono', text: `${verdict.judge_model || '–'} · ${verdict.votes ?? 0} vote${verdict.votes === 1 ? '' : 's'}` }),
        typeof verdict.judge_cost_usd === 'number' ? [h('dt', { text: 'Judge cost' }), h('dd', { text: money(verdict.judge_cost_usd) })] : null));
      // The judge's reasoning: its own rationale when the verdict carries one, the reasons a
      // run failed, and (above, under Expected behavior) the quoted evidence per item.
      const rationale = verdict.rationale || verdict.summary;
      const reasons = Array.isArray(verdict.failure_reasons) ? verdict.failure_reasons : [];
      host.append(h('h3', { class: 'small', text: 'Rationale' }));
      if (rationale) host.append(h('p', { class: 'prose', text: String(rationale) }));
      if (reasons.length) host.append(h('ul', { class: 'list-plain' }, reasons.map((r) => h('li', { text: r }))));
      if (!rationale && !reasons.length) {
        host.append(h('p', { class: 'muted small', text: verdict.passed ? 'No failure reasons: every graded item and every matcher check passed. The quoted evidence is under Expected behavior.' : 'The verdict gives no reasons.' }));
      }
      const matches = Array.isArray(verdict.matches) ? verdict.matches : [];
      if (matches.length) {
        const passedN = matches.filter((m) => m.passed).length;
        host.append(h('h3', { class: 'small', text: `Matcher · ${passedN} of ${matches.length} passed` }),
          h('div', { class: 'table-wrap' }, h('table', { class: 'table' },
            h('caption', { class: 'sr-only', text: 'Deterministic matcher results' }),
            h('thead', null, h('tr', null, ['Path', 'Op', 'Expected', 'Actual', 'Result'].map((c) => h('th', { scope: 'col', text: c })))),
            h('tbody', null, matches.map((m) => h('tr', null,
              h('td', { class: 'mono nowrap', text: m.path }), h('td', { class: 'mono nowrap', text: m.op }),
              h('td', { class: 'mono', text: JSON.stringify(m.expected) }), h('td', { class: 'mono', text: JSON.stringify(m.actual) }),
              h('td', null, verdictPill(m.passed), m.detail ? h('div', { class: 'muted small', text: m.detail }) : null)))))));
      }
    }
    const flags = [...new Set([...(verdict && verdict.flags) || [], ...(facts && facts.flags) || []])];
    const hard = (facts && facts.hard_failures) || [];
    host.append(h('h3', { class: 'small', text: 'Observer flags' }));
    if (!flags.length && !hard.length) host.append(h('p', { class: 'muted small', text: 'None.' }));
    else {
      host.append(h('ul', { class: 'list-plain' },
        flags.map((f) => h('li', null, h('span', { class: 'flagged', text: f }))),
        hard.map((f) => h('li', null, h('span', { class: 'flagged', text: 'fail: ' }), f))));
    }
    if (facts) {
      host.append(h('dl', { class: 'kv small' },
        h('dt', { text: 'Outcome' }), h('dd', null, facts.outcome || '–', facts.reason ? h('span', { class: 'muted', text: ` · ${facts.reason}` }) : null),
        h('dt', { text: 'Cost' }), h('dd', { text: money(facts.cost_usd) }),
        h('dt', { text: 'Duration' }), h('dd', { text: dur(facts.duration_s) })));
    }
  }

  function renderPlan() {
    const host = clear($('#d-plan'));
    const plan = state.run && state.run.plan;
    host.append(h('h2', { id: 'plan-title' }, 'Plan', h('span', { class: 'aside', text: plan ? `${(plan.paths || []).length} path(s)` : '' })));
    if (!plan) { host.append(h('p', { class: 'muted', text: state.run ? 'No plan.json in this run.' : 'Select a run to see its plan.' })); return; }
    for (const p of plan.paths || []) {
      const steps = Array.isArray(p.steps) ? p.steps : [];
      const checks = Array.isArray(p.checkpoints) ? p.checkpoints : [];
      host.append(h('details', { class: 'path' },
        h('summary', null, h('span', { class: 'kind', text: p.kind || 'path' }), h('span', { class: 'mono', text: p.id }), h('span', { text: p.title || '' })),
        h('div', { class: 'path-body' },
          p.rationale ? h('p', { class: 'muted small prose', text: p.rationale }) : null,
          h('h3', { class: 'small', text: `Steps (${steps.length})` }),
          h('ol', { class: 'steps' }, steps.map((st) => h('li', null,
            h('div', { text: st.intent || '' }),
            st.tool ? h('div', { class: 'mono small' }, st.tool, Object.keys(st.arguments_sketch || {}).length ? `(${argsInline(st.arguments_sketch)})` : '()') : null,
            st.success_looks_like ? h('div', { class: 'muted small', text: `success: ${st.success_looks_like}` }) : null,
            st.expect_error ? h('div', { class: 'small' }, statusPill('partial', 'expects an error')) : null))),
          checks.length ? [h('h3', { class: 'small', text: `Checkpoints (${checks.length})` }), h('ul', { class: 'list-plain small' }, checks.map((c) => h('li', { class: 'mono', text: c })))] : null)));
    }
    if (Array.isArray(plan.notes) && plan.notes.length) {
      host.append(h('details', { class: 'path' }, h('summary', null, h('span', { text: `Planner notes (${plan.notes.length})` })),
        h('ul', { class: 'list-plain small path-body' }, plan.notes.map((n) => h('li', { text: n })))));
    }
  }

  function renderSource() {
    const host = clear($('#d-source'));
    const v = specView();
    host.append(h('h2', { id: 'src-title' }, 'Scenario', h('span', { class: 'aside', text: state.run && state.run.scenario ? 'as run' : 'current file' })));
    host.append(h('dl', { class: 'kv' },
      h('dt', { text: 'Role' }), h('dd', { class: 'prose', text: v.role || '–' }),
      h('dt', { text: 'Goal' }), h('dd', { class: 'prose', text: v.goal || '–' }),
      h('dt', { text: 'Instructions' }), h('dd', null, (v.instructions || []).length ? h('ul', { class: 'list-plain' }, v.instructions.map((i) => h('li', { text: i }))) : '–'),
      h('dt', { text: 'Expected outcome' }), h('dd', null, v.expected_outcome_text ? h('p', { class: 'prose', text: v.expected_outcome_text }) : null,
        v.expected_outcome_json ? h('details', null, h('summary', { class: 'small', text: 'JSON spec' }), h('pre', { text: json(v.expected_outcome_json) })) : null),
      (v.observers || []).length ? [h('dt', { text: 'Observers' }), h('dd', { class: 'mono small', text: v.observers.join(', ') })] : null));
    const models = state.scenario && state.scenario.models;
    if (models) {
      host.append(h('h3', { class: 'small', text: 'Models for the next run' }), h('div', { class: 'table-wrap' }, h('table', { class: 'table' },
        h('caption', { class: 'sr-only', text: 'Effective models for this scenario' }),
        h('thead', null, h('tr', null, ['Role', 'Model', 'Source'].map((c) => h('th', { scope: 'col', text: c })))),
        h('tbody', null, ROLES.filter((r) => models[r]).map((r) => {
          const override = (state.opts.models || {})[r];
          return h('tr', null, h('td', { text: r }), h('td', { class: 'mono', text: override || models[r].spec }),
            h('td', { class: 'muted', text: override ? 'run override (settings)' : models[r].source }));
        })))));
    }
    const run = state.scenario && state.scenario.run_settings;
    if (run) {
      const o = state.opts || {};
      const shown = (k) => {
        if (k === 'repeat' && Number.isInteger(o.repeat)) return [String(o.repeat), 'run override (settings)'];
        if (k === 'modes' && Array.isArray(o.modes) && o.modes.length === 1) return [o.modes[0], 'run override (settings)'];
        const v = run[k].value;
        return [Array.isArray(v) ? v.join(' + ') : String(v), run[k].source];
      };
      host.append(h('h3', { class: 'small', text: 'Run settings for the next run' }), h('div', { class: 'table-wrap' }, h('table', { class: 'table' },
        h('caption', { class: 'sr-only', text: 'Run settings for this scenario' }),
        h('thead', null, h('tr', null, ['Setting', 'Value', 'Source'].map((c) => h('th', { scope: 'col', text: c })))),
        h('tbody', null, Object.keys(run).map((k) => {
          const [value, source] = shown(k);
          return h('tr', null, h('td', { text: k.replace('_', ' ') }), h('td', { class: 'mono', text: value }), h('td', { class: 'muted', text: source }));
        })))));
    }
  }

  function renderSims() {
    const host = clear($('#d-sims'));
    const rows = (state.run && state.run.transcripts) || [];
    host.append(h('h2', { id: 'sims-title' }, 'Simulations', h('span', { class: 'aside', text: rows.length ? `${rows.length} in this run` : '' })));
    if (!state.run) { host.append(h('p', { class: 'muted', text: state.sel.run ? 'Loading…' : 'No run selected.' })); return; }
    if (!rows.length) { host.append(h('p', { class: 'muted', text: 'This run has no transcripts (plan only, or it stopped early).' })); return; }
    host.append(h('div', { class: 'sims', role: 'group', 'aria-label': 'Transcripts in this run' }, rows.map((t) => {
      const status = t.judged ? (t.passed ? 'passed' : 'failed') : 'never';
      return h('button', {
        class: 'sim', type: 'button', 'aria-pressed': String(t.stem === state.sel.stem), disabled: !t.file,
        onclick: () => selectTranscript(t.stem, state.seq),
      }, h('span', { class: `dot ${STATUS[status].dot}`, 'aria-hidden': 'true' }),
      h('span', { class: 'sr-only', text: `${t.judged ? STATUS[status].label : 'Not judged'}: ` }),
      `${t.path_id} · ${t.mode || '?'} · #${t.index ?? '?'}`);
    })));
  }

  // --- conversation -----------------------------------------------------------------------
  function relTime(t, t0) {
    const d = Date.parse(t);
    if (Number.isNaN(d) || t0 === null) return '';
    return `+${((d - t0) / 1000).toFixed(1)} s`;
  }
  function who(label, t, t0) {
    return h('div', { class: 'ev-who' }, label, h('span', { class: 't', text: relTime(t, t0) }));
  }
  function resultSummary(res) {
    if (!res) return 'no result recorded';
    if (res.is_error) return firstLine(res.text) || 'error';
    const s = res.structured;
    if (Array.isArray(s)) return `${s.length} item${s.length === 1 ? '' : 's'}`;
    if (s && typeof s === 'object') {
      const entries = Object.entries(s);
      const shown = entries.slice(0, 4).map(([k, v]) => `${k}: ${brief(v)}`).join(' · ');
      return entries.length > 4 ? `${shown} · +${entries.length - 4} more` : shown;
    }
    return firstLine(res.text) || `${res.chars || 0} chars`;
  }

  function toolBlock(call, res, t0) {
    const status = !res ? h('span', { class: 'res none', text: 'no result' })
      : res.is_error ? h('span', { class: 'res err', text: 'error' }) : h('span', { class: 'res ok', text: 'ok' });
    const meta = [];
    if (res && typeof res.ms === 'number') meta.push(`${Math.round(res.ms)} ms`);
    if (res && res.chars) meta.push(`${res.chars} chars`);
    return h('div', { class: 'ev ev-tool' }, who('Tool', call.t, t0),
      h('details', { class: 'tool' },
        h('summary', null,
          h('span', { class: 'call', text: `${call.name}(${argsInline(call.arguments)})` }), status,
          h('span', { class: 'sum', text: resultSummary(res) })),
        h('div', { class: 'tool-body' },
          h('h4', { text: 'Arguments' }), h('pre', { text: json(call.arguments || {}) }),
          res ? [h('h4', { text: `Result${meta.length ? ` · ${meta.join(' · ')}` : ''}` }),
            h('pre', { text: res.structured !== null && res.structured !== undefined ? json(res.structured) : (res.text || '') }),
            res.sha256 ? h('p', { class: 'muted small mono', text: `sha256 ${res.sha256}` }) : null] : null)));
  }

  function minor(label, t, t0, ...content) {
    return h('div', { class: 'ev ev-minor' }, who(label, t, t0), h('div', { class: 'minor' }, ...content));
  }

  function renderEvents(events) {
    const out = [];
    const t0s = events.map((e) => Date.parse(e.t)).filter((n) => !Number.isNaN(n));
    const t0 = t0s.length ? Math.min(...t0s) : null;
    const consumed = new Set();
    events.forEach((e, i) => {
      if (consumed.has(i)) return;
      switch (e.kind) {
        case 'system': {
          const models = Object.entries(e.models || {}).map(([k, v]) => `${k}=${v}`).join(', ');
          const prompts = Object.entries(e.prompts || {});
          out.push(minor('Start', e.t, t0, `path ${e.path_id || '?'} · ${e.mode || '?'} · #${e.index ?? 0}`, models ? ` · ${models}` : '',
            prompts.length ? h('details', null, h('summary', { text: `System prompts (${prompts.map(([k]) => k).join(', ')})` }),
              prompts.map(([k, v]) => [h('h4', { class: 'small', text: k }), h('pre', { text: v })])) : null));
          break;
        }
        case 'user':
          out.push(h('div', { class: 'ev ev-user' }, who('User', e.t, t0), h('div', { class: 'bubble prose', text: e.text || '' })));
          break;
        case 'assistant': {
          if (!e.text) break;
          out.push(h('div', { class: 'ev ev-agent' }, who('Agent', e.t, t0), h('div', { class: 'bubble' },
            h('div', { class: 'prose', text: e.text }),
            e.stop_reason && e.stop_reason !== 'end_turn' && e.stop_reason !== 'tool_use' ? h('div', { class: 'note', text: `stop: ${e.stop_reason}` }) : null)));
          break;
        }
        case 'tool_call': {
          let res = null;
          for (let j = i + 1; j < events.length; j++) {
            const r = events[j];
            if (r.kind !== 'tool_result' || consumed.has(j)) continue;
            if ((e.tool_use_id && r.tool_use_id === e.tool_use_id) || (!e.tool_use_id && r.name === e.name)) { res = r; consumed.add(j); break; }
          }
          out.push(toolBlock(e, res, t0));
          break;
        }
        case 'tool_result':
          out.push(toolBlock({ name: e.name, arguments: {}, t: e.t }, e, t0));
          break;
        case 'tools_offered':
          out.push(minor('Tools', e.t, t0, (e.added || []).map((n) => h('span', { class: 'tag add', text: `+${n}` })),
            (e.removed || []).map((n) => h('span', { class: 'tag rm', text: `−${n}` })),
            !(e.added || []).length && !(e.removed || []).length ? 'no change ' : null,
            h('span', { text: e.reason ? `(${e.reason})` : '' })));
          break;
        case 'informant_report': {
          const reports = e.reports || [];
          const lines = reports.map((r) => h('span', { class: 'report-line' }, `${r.observer}.${r.condition} = `,
            h('span', { class: `val-${r.value === true ? 'true' : r.value === false ? 'false' : 'unknown'}`, text: r.value === true ? 'true' : r.value === false ? 'false' : 'unknown' }),
            r.evidence ? ` — ${r.evidence}` : ''));
          const effects = [...(e.flags || []).map((f) => `flag ${f}`), ...(e.failures || []).map((f) => `fail ${f}`), ...(e.notes || []).map((n) => `note ${n}`)];
          out.push(minor('Observers', e.t, t0, h('span', { text: `@${e.trigger || '?'} ` }),
            lines.length > 2 ? h('details', null, h('summary', { text: `${lines.length} reports` }), lines) : lines,
            effects.map((x) => h('span', { class: 'report-line flagged', text: x }))));
          break;
        }
        case 'goal_enabled':
          out.push(minor('Goal', e.t, t0, h('span', { text: `enabled${e.observer ? ` by ${e.observer}.${e.condition}` : ''}: ` }), e.text || ''));
          break;
        case 'final_result':
          out.push(h('div', { class: 'ev ev-final' }, who('Final', e.t, t0), h('div', { class: 'bubble' },
            e.parsed ? h('pre', { text: json(e.parsed) }) : [h('p', { class: 'muted small', text: 'No parseable JSON block; raw text:' }), h('pre', { text: e.raw || '' })])));
          break;
        case 'usage': {
          const per = Object.entries(e.per_model || {}).map(([m, u]) => `${m}: ${u.input_tokens || 0} in / ${u.output_tokens || 0} out`).join(' · ');
          out.push(minor('Usage', e.t, t0, `${money(e.cost_usd)}${e.estimate ? ' (estimate)' : ''}${per ? ` · ${per}` : ''}`));
          break;
        }
        case 'error':
          out.push(minor('Error', e.t, t0, h('span', { class: 'flagged', text: e.message || '' })));
          break;
        case 'end':
          out.push(h('div', { class: 'ev ev-end' }, who('End', e.t, t0), h('div', { class: 'bubble' },
            statusPill(e.outcome === 'completed' ? 'done' : 'error', e.outcome || 'unknown'), e.reason ? h('span', { class: 'muted', text: ` ${e.reason}` }) : null)));
          break;
        case 'unparsed':
          out.push(minor('Unparsed', e.t, t0, h('pre', { text: e.raw || '' })));
          break;
        default:
          out.push(minor(e.kind || 'Event', e.t, t0, h('details', null, h('summary', { text: 'details' }), h('pre', { text: json(e) }))));
      }
    });
    return out;
  }

  function renderConversation(loading) {
    const host = clear($('#d-convo'));
    const tr = state.transcript;
    host.append(h('h2', { id: 'convo-title' }, 'Conversation', h('span', { class: 'aside', text: tr ? tr.file : '' })));
    if (loading) { host.append(h('p', { class: 'muted', text: 'Loading…' })); return; }
    if (!tr) { host.append(h('p', { class: 'muted', text: state.run ? 'Pick a simulation above.' : 'No transcript to show.' })); return; }
    const events = tr.events || [];
    const tools = events.filter((e) => e.kind === 'tool_call').length;
    const controls = h('div', { class: 'd-actions' },
      h('span', { class: 'muted small', text: `${events.length} events · ${tools} tool call${tools === 1 ? '' : 's'}` }),
      h('button', { class: 'btn btn-sm', type: 'button', onclick: () => host.querySelectorAll('details.tool').forEach((d) => { d.open = true; }) }, 'Expand tools'),
      h('button', { class: 'btn btn-sm', type: 'button', onclick: () => host.querySelectorAll('details.tool').forEach((d) => { d.open = false; }) }, 'Collapse tools'));
    host.append(controls, h('div', { class: 'convo' }, renderEvents(events)));
  }

  // --- runs and jobs ----------------------------------------------------------------------
  function runBody(names) {
    const o = state.opts || {};
    const body = { scenarios: names };
    const models = {};
    for (const [k, v] of Object.entries(o.models || {})) if (v) models[k] = v;
    if (Object.keys(models).length) body.models = models;
    if (Number.isInteger(o.repeat) && o.repeat > 0) body.repeat = o.repeat;
    if (Array.isArray(o.modes) && o.modes.length === 1) body.modes = o.modes;
    if (o.dry_run) body.dry_run = true;
    if (o.allow_same_judge) body.allow_same_judge = true;
    return body;
  }

  async function startRun(names, label) {
    if (!names.length) return;
    const live = !(state.opts && state.opts.dry_run);
    if (names.length > 1 && live && !window.confirm(`Run ${names.length} scenarios? Each run makes live model calls and costs money.`)) return;
    try {
      const res = await api('/api/run', runBody(names));
      announce(`Started ${label}`);
      toast(`Started ${label}`, true);
      trackJob(res.job_id);
      await loadList();
      if (state.scenario && names.includes(state.sel.name)) { state.scenario.status = 'running'; renderDetail(); }
    } catch (e) {
      toast(e.status === 409 ? `Already running: ${((e.data && e.data.scenarios) || []).join(', ')}` : `Could not start: ${e.message}`);
    }
  }

  function trackJob(jobId) {
    if (!state.jobs.has(jobId)) state.jobs.set(jobId, { job_id: jobId, status: 'queued', scenarios: [], log_tail: [] });
    renderJobs();
    schedulePoll(0);
  }

  let pollTimer = 0;
  function schedulePoll(delay = POLL_MS) {
    window.clearTimeout(pollTimer);
    pollTimer = window.setTimeout(pollJobs, delay);
  }
  async function pollJobs() {
    const active = [...state.jobs.values()].filter((j) => j.status === 'queued' || j.status === 'running');
    if (!active.length) return;
    const finished = [];
    await Promise.all(active.map(async (j) => {
      try {
        const next = await api(`/api/jobs/${enc(j.job_id)}`);
        const before = JSON.stringify(j.scenarios.map((t) => t.status));
        state.jobs.set(j.job_id, next);
        if (next.status === 'done' || next.status === 'failed') finished.push(next);
        else if (before !== JSON.stringify(next.scenarios.map((t) => t.status))) finished.push(null);
      } catch (e) {
        if (e.status === 404) state.jobs.delete(j.job_id);
      }
    }));
    renderJobs();
    const name = state.sel.name;
    const selRunning = name && [...state.jobs.values()].some((j) => (j.scenarios || []).some((t) => t.name === name && t.status === 'running'));
    if (selRunning && !finished.length) await refreshHistory(name);
    if (finished.length) {
      for (const job of finished.filter(Boolean)) {
        const counts = {};
        job.scenarios.forEach((t) => { counts[t.status] = (counts[t.status] || 0) + 1; });
        announce(`Run finished: ${Object.entries(counts).map(([k, n]) => `${n} ${k}`).join(', ')}`);
      }
      await loadList();
      if (state.sel.name) await refreshScenario();
    }
    if ([...state.jobs.values()].some((j) => j.status === 'queued' || j.status === 'running')) schedulePoll();
  }

  async function refreshHistory(name) {
    // While the selected scenario runs, its history gains the directory being written.
    try {
      const data = await api(`/api/scenarios/${enc(name)}/runs`);
      if (name !== state.sel.name || !$('#d-history')) return;
      const sig = (runs) => JSON.stringify((runs || []).map((r) => [r.run_id, r.status, r.runs]));
      if (sig(data.runs) === sig(state.runs)) return;
      state.runs = data.runs;
      renderHistory();
    } catch (e) { /* the next poll tries again */ }
  }

  async function refreshScenario() {
    const name = state.sel.name;
    if (!name) return;
    try {
      const [scenario, runs] = await Promise.all([api(`/api/scenarios/${enc(name)}`), api(`/api/scenarios/${enc(name)}/runs`)]);
      if (name !== state.sel.name) return;
      const newest = runs.runs[0] && runs.runs[0].run_id;
      const hadNewest = (state.runs || []).some((r) => r.run_id === newest);
      state.scenario = scenario;
      state.runs = runs.runs;
      renderDetail();
      if (newest && !hadNewest && scenario.status !== 'running') {
        state.run = null; state.transcript = null;
        await selectRun(newest, null, state.seq);
      } else if (state.sel.run) {
        state.run = null;
        await selectRun(state.sel.run, state.sel.stem, state.seq);
      }
    } catch (e) {
      toast(`Could not refresh: ${e.message}`);
    }
  }

  async function cancelJob(jobId) {
    try {
      await api(`/api/jobs/${enc(jobId)}/cancel`, {});
      toast('Stopping…', true);
      schedulePoll(0);
    } catch (e) {
      toast(`Could not stop: ${e.message}`);
    }
  }

  let jobsHidden = false;
  function renderJobs() {
    const section = $('#jobs');
    const body = clear($('#jobs-body'));
    const jobs = [...state.jobs.values()].sort((a, b) => String(b.created_at || '').localeCompare(String(a.created_at || ''))).slice(0, 6);
    section.hidden = jobs.length === 0;
    const active = jobs.filter((j) => j.status === 'queued' || j.status === 'running').length;
    $('#jobs-count').textContent = active ? `${active} in progress` : 'all finished';
    body.hidden = jobsHidden;
    $('#jobs-toggle').textContent = jobsHidden ? 'Show' : 'Hide';
    $('#jobs-toggle').setAttribute('aria-expanded', String(!jobsHidden));
    for (const j of jobs) {
      const running = j.status === 'queued' || j.status === 'running';
      const opts = j.options || {};
      const optText = [opts.dry_run ? 'dry run' : null, opts.repeat ? `repeat ${opts.repeat}` : null,
        (opts.modes || []).length === 1 ? opts.modes[0] : null,
        Object.keys(opts.models || {}).length ? Object.entries(opts.models).map(([k, v]) => `${k}=${v}`).join(', ') : null].filter(Boolean).join(' · ');
      const log = h('pre', { class: 'job-log', 'aria-label': 'Log tail', tabindex: '0', text: (j.log_tail || []).join('\n') || 'waiting for output…' });
      body.append(h('div', { class: 'job' },
        h('div', { class: 'job-head' },
          statusPill(j.status),
          h('span', { class: 'mono small', text: `job ${j.job_id}` }),
          h('span', { class: 'muted small', text: j.created_at ? when(j.created_at) : '' }),
          optText ? h('span', { class: 'muted small', text: optText }) : null,
          h('span', { class: 'grow' }),
          running ? h('button', { class: 'btn btn-sm btn-danger', type: 'button', onclick: () => cancelJob(j.job_id) }, 'Stop') : null,
          !running ? h('button', { class: 'btn btn-sm btn-quiet', type: 'button', onclick: () => { state.jobs.delete(j.job_id); renderJobs(); } }, 'Dismiss') : null),
        h('div', { class: 'job-tasks' }, (j.scenarios || []).map((t) => h('button', {
          class: 'job-task', type: 'button', onclick: () => go(t.name, t.run_id || null),
        }, statusPill(t.status), t.name))),
        j.error ? h('p', { class: 'error-text', text: j.error }) : null,
        log));
      log.scrollTop = log.scrollHeight;
    }
  }

  async function loadJobs() {
    try {
      const data = await api('/api/jobs');
      for (const j of data.jobs || []) state.jobs.set(j.job_id, j);
      renderJobs();
      schedulePoll(0);
    } catch (e) { /* the dock stays hidden */ }
  }

  // --- settings ---------------------------------------------------------------------------
  async function loadConfig() {
    try { state.config = await api('/api/config'); } catch (e) { toast(`Could not load config: ${e.message}`); return; }
    const dl = clear($('#known-models'));
    for (const m of state.config.known_models || []) dl.append(h('option', { value: m }));
    updateOverrideBadge();
  }

  function fillSettings() {
    const cfg = state.config || {};
    const tbody = clear($('#roles-table tbody'));
    for (const r of cfg.roles || []) {
      tbody.append(h('tr', null, h('th', { scope: 'row', text: r.role }), h('td', { text: r.provider }), h('td', { class: 'mono', text: r.model }),
        h('td', { class: 'muted', text: r.source }), h('td', { text: r.temperature ?? '–' }), h('td', { text: r.max_tokens ?? '–' })));
    }
    const skill = cfg.skill || {};
    const info = clear($('#skill-info'));
    const sources = cfg.run_sources || {};
    const runDefaults = Object.entries(cfg.run || {}).map(([k, v]) => `${k}=${Array.isArray(v) ? v.join('/') : v}${sources[k] ? ` (${sources[k]})` : ''}`).join(', ');
    append(info, [
      h('dt', { text: 'Skill' }), h('dd', { class: 'mono', text: skill.path ? `${skill.name} · ${skill.path}` : '–' }),
      h('dt', { text: 'config.yaml' }), h('dd', { class: 'mono', text: skill.config_file || 'not found' }),
      h('dt', { text: 'Run defaults' }), h('dd', { text: runDefaults || '–' }),
      h('dt', { text: 'Scenarios' }), h('dd', { class: 'mono', text: `${(cfg.scenario_sources || []).join(', ') || '–'}${cfg.scenario_sources_from ? ` (from ${cfg.scenario_sources_from})` : ''}` }),
      h('dt', { text: 'Runs' }), h('dd', { class: 'mono', text: cfg.runs_dir || '' }),
      (skill.overrides || []).length ? [h('dt', { text: 'Overrides' }), h('dd', null, h('ul', { class: 'list-plain' }, skill.overrides.map((o) => h('li', { text: overrideText(o) }))))] : null,
    ]);
    const warnings = clear($('#config-warnings'));
    for (const w of cfg.warnings || []) warnings.append(h('li', { text: w }));

    const o = state.opts || {};
    $('#opt-repeat').value = Number.isInteger(o.repeat) ? String(o.repeat) : '';
    $('#opt-repeat').placeholder = cfg.run && cfg.run.repeat ? `config: ${cfg.run.repeat}` : 'scenario default';
    // The CLI runs one mode (--mode) or the default set, so the choice is one of three.
    const one = Array.isArray(o.modes) && o.modes.length === 1 ? o.modes[0] : '';
    for (const input of document.querySelectorAll('input[name="opt-mode"]')) input.checked = input.value === one;
    const cfgModes = cfg.run && Array.isArray(cfg.run.modes) ? cfg.run.modes.join(' + ') : '';
    $('#opt-mode-default-text').textContent = cfgModes ? `default (config: ${cfgModes})` : 'default (the scenario\'s modes)';
    $('#opt-dry-run').checked = Boolean(o.dry_run);
    $('#opt-same-judge').checked = Boolean(o.allow_same_judge);
    const fields = clear($('#override-fields'));
    const resolved = Object.fromEntries((cfg.roles || []).map((r) => [r.role, r.spec]));
    for (const role of cfg.model_roles || ROLES) {
      const id = `ovr-${role}`;
      fields.append(h('label', { for: id, text: role }),
        h('input', { id, type: 'text', list: 'known-models', autocomplete: 'off', spellcheck: 'false', 'data-role': role,
          placeholder: resolved[role] || '', value: (o.models || {})[role] || '' }));
    }
    $('#settings-error').textContent = '';
  }

  function overrideText(o) {
    // {match: {category: "Origin*"}, models: {agent: "..."}, run: {repeat: 3}} as one line.
    const pairs = (obj) => Object.entries(obj && typeof obj === 'object' ? obj : {}).map(([k, v]) => `${k}=${typeof v === 'string' ? v : JSON.stringify(v)}`).join(', ');
    const parts = [pairs(o.models), pairs(o.run)].filter(Boolean).join('; ');
    return `when ${pairs(o.match) || '(no match)'}: ${parts || 'nothing'}`;
  }

  function readSettings() {
    const err = $('#settings-error');
    const repeatRaw = $('#opt-repeat').value.trim();
    const repeat = repeatRaw ? Number(repeatRaw) : null;
    if (repeatRaw && (!Number.isInteger(repeat) || repeat < 1 || repeat > 100)) { err.textContent = 'Repeat must be a whole number from 1 to 100.'; return null; }
    const picked = document.querySelector('input[name="opt-mode"]:checked');
    const modes = picked && picked.value ? [picked.value] : [];
    const models = {};
    for (const input of document.querySelectorAll('#override-fields input')) {
      const v = input.value.trim();
      if (!v) continue;
      if (!MODEL_SPEC.test(v)) { err.textContent = `${input.dataset.role}: “${v}” is not a model spec.`; input.focus(); return null; }
      models[input.dataset.role] = v;
    }
    err.textContent = '';
    return { repeat, modes, models, dry_run: $('#opt-dry-run').checked, allow_same_judge: $('#opt-same-judge').checked };
  }

  function updateOverrideBadge() {
    const o = state.opts || {};
    const n = Object.values(o.models || {}).filter(Boolean).length + (Number.isInteger(o.repeat) ? 1 : 0) +
      (Array.isArray(o.modes) && o.modes.length === 1 ? 1 : 0) + (o.dry_run ? 1 : 0) + (o.allow_same_judge ? 1 : 0);
    const badge = $('#override-badge');
    badge.hidden = n === 0;
    badge.textContent = String(n);
    $('#open-settings').setAttribute('aria-label', n ? `Settings, ${n} run option${n === 1 ? '' : 's'} active` : 'Settings');
  }

  function openSettings() {
    fillSettings();
    const dlg = $('#settings');
    if (typeof dlg.showModal === 'function') dlg.showModal(); else dlg.setAttribute('open', '');
  }
  function closeSettings() {
    const dlg = $('#settings');
    if (typeof dlg.close === 'function') dlg.close(); else dlg.removeAttribute('open');
    $('#open-settings').focus();
  }

  // --- wiring -----------------------------------------------------------------------------
  function bind() {
    const search = $('#search');
    let searchTimer = 0;
    search.addEventListener('input', () => {
      window.clearTimeout(searchTimer);
      searchTimer = window.setTimeout(() => { state.filter.q = search.value.trim(); renderList(); }, 80);
    });
    search.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape') { search.value = ''; state.filter.q = ''; renderList(); }
      if (ev.key === 'ArrowDown') { const first = $('#scenario-list .row-main'); if (first) { ev.preventDefault(); first.focus(); } }
    });
    $('#status-filter').addEventListener('change', (ev) => { state.filter.status = ev.target.value; renderList(); });
    $('#scenario-list').addEventListener('keydown', onListKey);
    $('#refresh').addEventListener('click', async () => {
      await Promise.all([loadConfig(), loadList()]);
      if (state.sel.name) await refreshScenario();
      announce('Refreshed');
    });
    $('#run-filtered').addEventListener('click', () => {
      // Always an explicit list: what the filter shows, minus invalid and already-running ones.
      const names = filtered().filter((s) => !s.error && s.status !== 'running').map((s) => s.name);
      startRun(names, filterActive() ? `${names.length} shown scenario${names.length === 1 ? '' : 's'}` : `all ${names.length} scenarios`);
    });
    $('#open-settings').addEventListener('click', openSettings);
    $('#settings-close').addEventListener('click', closeSettings);
    $('#settings').addEventListener('cancel', () => { $('#open-settings').focus(); });
    $('#settings-save').addEventListener('click', () => {
      const next = readSettings();
      if (!next) return;
      state.opts = next;
      save('mcpsim.ui.opts', next);
      updateOverrideBadge();
      if (state.scenario) renderSource();
      closeSettings();
      toast('Run options saved', true);
    });
    $('#settings-reset').addEventListener('click', () => {
      state.opts = {};
      save('mcpsim.ui.opts', {});
      fillSettings();
      updateOverrideBadge();
      if (state.scenario) renderSource();
    });
    $('#jobs-toggle').addEventListener('click', () => { jobsHidden = !jobsHidden; renderJobs(); });
    document.addEventListener('keydown', (ev) => {
      const tag = (ev.target && ev.target.tagName) || '';
      if (ev.key === '/' && !['INPUT', 'TEXTAREA', 'SELECT'].includes(tag) && !ev.metaKey && !ev.ctrlKey) {
        ev.preventDefault();
        search.focus();
        search.select();
      }
    });
    window.addEventListener('popstate', route);
    window.addEventListener('hashchange', route);
  }

  async function init() {
    bind();
    await Promise.all([loadConfig(), loadList()]);
    await loadJobs();
    await route();
  }

  init();
})();
