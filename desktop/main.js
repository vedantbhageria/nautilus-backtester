// Nautilus desktop shell.
//
//   npm start            (from this folder)   or   launch.bat (repo root)
//
// Startup: a splash window paints immediately. Then Redis is brought up
// inside WSL, and the backtester (:8002) and live dashboard (:8000) servers
// are started if nothing already answers on those ports. The main window
// hosts both in tabs under a thin title/tab bar (shell.html).
//
// Notifications are the app's own: a transparent always-on-top overlay
// (notifications.html) in the corner of the screen, fed over IPC. Pages call
// window.desktop.notify({title, body, kind, action}) (see preload.js).
// Clicking a toast focuses the app, switches to the tab that raised it and
// hands the action back to that page (e.g. open the finished job).
//
// Only processes this app started are stopped on quit. A server that was
// already running (e.g. from a terminal) is reused and left alone.
//
// The live trading node (scripts/binance_data.py) is started exactly as
// launch.bat always did: in its own "nautilus node" console window, reused if
// one is already running, and left running when the app closes. Set
// NAUTILUS_START_NODE=0 to skip it.

const { app, BaseWindow, BrowserWindow, WebContentsView, ipcMain, screen, shell } = require('electron');
const { spawn, execFile } = require('child_process');
const fs = require('fs');
const http = require('http');
const net = require('net');
const path = require('path');

app.setAppUserModelId('com.nautilus.desktop');
if (!app.requestSingleInstanceLock()) { app.quit(); }

// ---------------------------------------------------------------------------
// configuration
// ---------------------------------------------------------------------------
function projectRoot() {
  if (process.env.NAUTILUS_ROOT) return process.env.NAUTILUS_ROOT;
  // packaged portable exe: root.txt next to the real exe points at the repo
  const exeDir = process.env.PORTABLE_EXECUTABLE_DIR || path.dirname(process.execPath);
  try {
    const cfg = path.join(exeDir, 'root.txt');
    if (fs.existsSync(cfg)) return fs.readFileSync(cfg, 'utf8').trim();
  } catch (e) { /* fall through */ }
  return path.resolve(__dirname, '..');
}
const ROOT = projectRoot();

// Bare `python` on this machine is 3.12, which lacks the project's packages.
// The py launcher pins the interpreter that has them.
const PY = process.env.NAUTILUS_PYTHON || 'py';
const PY_ARGS = process.env.NAUTILUS_PYTHON ? [] : ['-3.14'];
const WSL_DISTRO = process.env.NAUTILUS_WSL_DISTRO || 'Ubuntu';
const REDIS_PORT = 6379;

const SERVICES = {
  backtester: {
    label: 'Backtester', port: 8002, probe: '/api/jobs', critical: true,
    args: [...PY_ARGS, path.join(ROOT, 'backtester', 'backend', 'server.py')],
  },
  live: {
    label: 'Live', port: 8000, probe: '/api/symbols', critical: false,
    args: [...PY_ARGS, '-m', 'uvicorn', '--app-dir', ROOT, 'server.app:app',
           '--host', '127.0.0.1', '--port', '8000'],
  },
};
const TABS = ['backtester', 'live'];
const START_NODE = process.env.NAUTILUS_START_NODE !== '0';
const BAR_H = 38;
const BG = '#0b0e14';

// ---------------------------------------------------------------------------
// logging
// ---------------------------------------------------------------------------
const LOG_DIR = path.join(app.getPath('userData'), 'logs');
fs.mkdirSync(LOG_DIR, { recursive: true });
const LOG = path.join(LOG_DIR, 'launcher.log');
function log(msg) {
  try { fs.appendFileSync(LOG, new Date().toISOString() + ' ' + msg + '\n'); } catch (e) { /* ignore */ }
}

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------
const state = {
  redis: 'wait', backtester: 'wait', live: 'wait',   // wait | ok | fail
  node: START_NODE ? 'wait' : 'off',                 // live trading node (+ 'off')
  detail: {},
  active: 'backtester',
};
const owned = {};          // service id -> ChildProcess we spawned
const restarts = {};       // service id -> [timestamps] of auto-restarts
let keepalive = null;      // `wsl sleep infinity`: stops WSL idling the distro (and Redis) away
let quitting = false;

let splash = null, win = null, barView = null, notifWin = null;
const views = {};          // tab id -> WebContentsView

// ---------------------------------------------------------------------------
// probes
// ---------------------------------------------------------------------------
function httpOk(port, pathname, timeout = 1500) {
  return new Promise(resolve => {
    const req = http.get({ host: '127.0.0.1', port, path: pathname, timeout }, res => {
      res.resume(); resolve(res.statusCode < 500);
    });
    req.on('error', () => resolve(false));
    req.on('timeout', () => { req.destroy(); resolve(false); });
  });
}

function redisPing(timeout = 1200) {
  return new Promise(resolve => {
    const sock = net.connect({ host: '127.0.0.1', port: REDIS_PORT });
    let buf = '';
    const done = ok => { sock.destroy(); resolve(ok); };
    sock.setTimeout(timeout, () => done(false));
    sock.on('connect', () => sock.write('PING\r\n'));
    sock.on('data', d => { buf += d; if (buf.includes('\r\n')) done(buf.startsWith('+PONG')); });
    sock.on('error', () => done(false));
  });
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

function setState(id, s, detail) {
  state[id] = s;
  if (detail !== undefined) state.detail[id] = detail;
  const payload = { redis: state.redis, backtester: state.backtester, live: state.live,
                    node: state.node, detail: state.detail, active: state.active };
  for (const w of [splash, barView]) {
    const wc = w && (w.webContents || w);
    if (wc && !wc.isDestroyed()) wc.send('shell:state', payload);
  }
}

// ---------------------------------------------------------------------------
// Redis (WSL)
// ---------------------------------------------------------------------------
function wsl(args, timeout = 20000) {
  return new Promise(resolve => {
    execFile('wsl.exe', ['-d', WSL_DISTRO, ...args], { windowsHide: true, timeout },
      (err, stdout, stderr) => resolve({ err, out: String(stdout || '') + String(stderr || '') }));
  });
}

function startKeepalive() {
  if (keepalive) return;
  keepalive = spawn('wsl.exe', ['-d', WSL_DISTRO, '--', 'sleep', 'infinity'],
                    { windowsHide: true, stdio: 'ignore' });
  keepalive.on('exit', c => { log(`wsl keepalive exited (${c})`); keepalive = null; });
}

async function ensureRedis() {
  setState('redis', 'wait', 'checking');
  // touching the distro boots it (and systemd starts redis-server if enabled)
  startKeepalive();
  for (let i = 0; i < 6; i++) {
    if (await redisPing()) { setState('redis', 'ok', 'localhost:6379'); return true; }
    await sleep(500);
  }
  setState('redis', 'wait', 'starting redis-server');
  log('redis not answering, starting it in WSL');
  let r = await wsl(['-u', 'root', '--', 'service', 'redis-server', 'start']);
  log('service redis-server start: ' + (r.err ? r.err.message : 'ok') + ' ' + r.out.trim());
  for (let i = 0; i < 16; i++) {
    if (await redisPing()) { setState('redis', 'ok', 'localhost:6379'); return true; }
    await sleep(500);
  }
  r = await wsl(['--', 'redis-server', '--daemonize', 'yes']);
  log('redis-server --daemonize: ' + r.out.trim());
  for (let i = 0; i < 10; i++) {
    if (await redisPing()) { setState('redis', 'ok', 'localhost:6379'); return true; }
    await sleep(500);
  }
  setState('redis', 'fail', 'not reachable on :6379');
  return false;
}

// ---------------------------------------------------------------------------
// Python servers
// ---------------------------------------------------------------------------
function spawnService(id) {
  const svc = SERVICES[id];
  const out = fs.openSync(path.join(LOG_DIR, `${id}.log`), 'a');
  fs.writeSync(out, `\n===== ${new Date().toISOString()} start ${PY} ${svc.args.join(' ')}\n`);
  const proc = spawn(PY, svc.args, {
    cwd: ROOT, windowsHide: true, stdio: ['ignore', out, out],
    env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
  });
  proc.on('error', e => log(`${id} spawn error: ${e.message}`));
  proc.on('exit', code => {
    log(`${id} exited (${code})`);
    if (owned[id] === proc) delete owned[id];
    if (!quitting) onServiceDied(id, code);
  });
  owned[id] = proc;
  log(`${id} started pid ${proc.pid}`);
  return proc;
}

async function ensureService(id) {
  const svc = SERVICES[id];
  setState(id, 'wait', 'checking :' + svc.port);
  if (await httpOk(svc.port, svc.probe)) {
    setState(id, 'ok', `:${svc.port} (already running)`);
    return true;
  }
  setState(id, 'wait', 'starting server');
  const proc = spawnService(id);
  for (let i = 0; i < 90; i++) {                 // imports can take ~20s cold
    if (await httpOk(svc.port, svc.probe)) { setState(id, 'ok', `:${svc.port}`); return true; }
    if (proc.exitCode !== null) break;
    await sleep(500);
  }
  setState(id, 'fail', `did not come up - see logs/${id}.log`);
  return false;
}

async function onServiceDied(id, code) {
  const svc = SERVICES[id];
  setState(id, 'fail', `stopped (exit ${code})`);
  const now = Date.now();
  restarts[id] = (restarts[id] || []).filter(t => now - t < 60000);
  if (restarts[id].length >= 3) {
    notify({ title: `${svc.label} server keeps stopping`,
             body: `Gave up after 3 restarts in a minute. Details: logs/${id}.log`, kind: 'error' });
    return;
  }
  restarts[id].push(now);
  notify({ title: `${svc.label} server stopped`, body: 'Restarting it now...', kind: 'warn' });
  if (await ensureService(id)) {
    notify({ title: `${svc.label} server back up`, body: `Listening on :${svc.port}`, kind: 'success' });
    const v = views[id];
    if (v && !v.webContents.isDestroyed()) v.webContents.reload();
  }
}

// ---------------------------------------------------------------------------
// live trading node
// ---------------------------------------------------------------------------
function nodeRunning() {
  // match python processes only (the query's own command line contains the text)
  const ps = "@(Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and " +
             "$_.CommandLine -like '*binance_data*' }).Count";
  return new Promise(resolve => {
    execFile('powershell.exe', ['-NoProfile', '-Command', ps], { windowsHide: true, timeout: 15000 },
      (err, out) => resolve(!err && parseInt(String(out).trim(), 10) > 0));
  });
}

async function ensureNode() {
  if (!START_NODE) { setState('node', 'off', 'disabled (NAUTILUS_START_NODE=0)'); return false; }
  setState('node', 'wait', 'checking');
  if (await nodeRunning()) { setState('node', 'ok', 'running (reused)'); return true; }
  if (state.redis !== 'ok') { setState('node', 'fail', 'needs Redis'); return false; }
  setState('node', 'wait', 'starting');
  const cmd = `start "nautilus node" cmd /k ${PY} ${PY_ARGS.join(' ')} scripts\\binance_data.py`;
  // verbatim: node would otherwise escape the quoted window title and break `start`;
  // not hidden: `start` inherits the show state, and this window is meant to be seen
  const p = spawn('cmd.exe', ['/d', '/s', '/c', `"${cmd}"`], {
    cwd: ROOT, detached: true, stdio: 'ignore', windowsVerbatimArguments: true });
  p.unref();
  log('trading node launched in its own window');
  for (let i = 0; i < 20; i++) {
    await sleep(1000);
    if (await nodeRunning()) { setState('node', 'ok', 'running'); return true; }
  }
  setState('node', 'fail', 'did not start - check the "nautilus node" window');
  return false;
}

function killTree(proc) {
  if (!proc || proc.exitCode !== null) return;
  // py.exe launches python.exe as a child - take the whole tree down
  try { execFile('taskkill', ['/pid', String(proc.pid), '/T', '/F'], { windowsHide: true }); }
  catch (e) { try { proc.kill(); } catch (e2) { /* ignore */ } }
}

// ---------------------------------------------------------------------------
// windows
// ---------------------------------------------------------------------------
const PRELOAD = path.join(__dirname, 'preload.js');
const webPrefs = extra => ({ preload: PRELOAD, contextIsolation: true, nodeIntegration: false,
                             sandbox: true, ...extra });

function createSplash() {
  splash = new BrowserWindow({
    width: 480, height: 320, frame: false, resizable: false, show: false,
    backgroundColor: BG, center: true, skipTaskbar: false, title: 'Nautilus',
    webPreferences: webPrefs(),
  });
  splash.loadFile(path.join(__dirname, 'splash.html'));
  splash.once('ready-to-show', () => splash.show());
  splash.on('closed', () => { splash = null; if (!win && !quitting) app.quit(); });
}

function serviceUrl(id) { return `http://127.0.0.1:${SERVICES[id].port}/`; }

function createMain() {
  const wa = screen.getPrimaryDisplay().workAreaSize;
  win = new BaseWindow({
    width: Math.min(1600, wa.width), height: Math.min(1000, wa.height),
    minWidth: 900, minHeight: 600, backgroundColor: BG, title: 'Nautilus', show: false,
    titleBarStyle: 'hidden',
    titleBarOverlay: { color: '#0d1017', symbolColor: '#9aa4b8', height: BAR_H },
  });

  barView = new WebContentsView({ webPreferences: webPrefs() });
  barView.setBackgroundColor('#0d1017');
  barView.webContents.loadFile(path.join(__dirname, 'shell.html'));
  win.contentView.addChildView(barView);

  for (const id of TABS) {
    const v = new WebContentsView({ webPreferences: webPrefs({ backgroundThrottling: false }) });
    v.setBackgroundColor(BG);
    wireView(id, v);
    win.contentView.addChildView(v);
    views[id] = v;
    if (state[id] === 'ok') v.webContents.loadURL(serviceUrl(id));
    else v.webContents.loadFile(path.join(__dirname, 'offline.html'), { query: { id, label: SERVICES[id].label } });
  }
  layout();
  showTab(state.active);
  win.on('resize', layout);
  win.on('closed', () => { win = null; app.quit(); });
  barView.webContents.once('did-finish-load', () => setState('redis', state.redis));
  views.backtester.webContents.once('did-finish-load', () => {
    win.show();
    if (splash) { splash.destroy(); splash = null; }
  });
  // don't hang behind the splash forever if the page itself fails to load
  setTimeout(() => { if (win && !win.isVisible()) { win.show(); if (splash) { splash.destroy(); splash = null; } } }, 15000);
}

function wireView(id, v) {
  const wc = v.webContents;
  // links with target=_blank (reports, presentations, docs) -> default browser
  wc.setWindowOpenHandler(({ url }) => {
    if (/^https?:/i.test(url)) shell.openExternal(url);
    return { action: 'deny' };
  });
  wc.on('before-input-event', (e, input) => {
    if (input.type !== 'keyDown') return;
    const k = input.key.toLowerCase();
    if (input.control && k === '1') { showTab('backtester'); e.preventDefault(); }
    else if (input.control && k === '2') { showTab('live'); e.preventDefault(); }
    else if (input.control && input.shift && k === 'i') { wc.toggleDevTools(); e.preventDefault(); }
    else if (k === 'f12') { wc.toggleDevTools(); e.preventDefault(); }
    else if (k === 'f5' || (input.control && k === 'r')) { wc.reload(); e.preventDefault(); }
  });
  wc.on('did-fail-load', (_e, code, desc, url, isMain) => {
    if (!isMain || code === -3) return;             // -3 = aborted (navigation replaced)
    log(`${id} failed to load ${url}: ${desc}`);
    wc.loadFile(path.join(__dirname, 'offline.html'), { query: { id, label: SERVICES[id].label } });
  });
}

function layout() {
  if (!win) return;
  const [w, h] = win.getContentSize();
  barView.setBounds({ x: 0, y: 0, width: w, height: BAR_H });
  for (const id of TABS) views[id].setBounds({ x: 0, y: BAR_H, width: w, height: h - BAR_H });
}

function showTab(id) {
  if (!views[id]) return;
  state.active = id;
  for (const t of TABS) views[t].setVisible(t === id);
  views[id].webContents.focus();
  setState('redis', state.redis);   // re-broadcast so the tab bar highlights
}

// ---------------------------------------------------------------------------
// notifications overlay
// ---------------------------------------------------------------------------
const NOTIF_W = 400;
const pending = new Map();      // toast id -> { source: tab id, action }
let notifSeq = 0;
const notifQueue = [];          // toasts raised before the overlay finished loading
let notifReady = false;

function createNotifWindow() {
  const wa = screen.getPrimaryDisplay().workArea;
  notifWin = new BrowserWindow({
    width: NOTIF_W, height: wa.height, x: wa.x + wa.width - NOTIF_W, y: wa.y,
    frame: false, transparent: true, resizable: false, movable: false,
    skipTaskbar: true, focusable: false, show: false, hasShadow: false,
    alwaysOnTop: true, webPreferences: webPrefs(),
  });
  notifWin.setAlwaysOnTop(true, 'screen-saver');
  notifWin.setIgnoreMouseEvents(true, { forward: true });
  notifWin.loadFile(path.join(__dirname, 'notifications.html'));
  notifWin.webContents.once('did-finish-load', () => {
    notifReady = true;
    while (notifQueue.length) sendToast(notifQueue.shift());
  });
  notifWin.on('closed', () => { notifWin = null; notifReady = false; });
}

function placeNotifWin() {
  // follow whichever display the main window is on
  const ref = win ? win.getBounds() : null;
  const disp = ref ? screen.getDisplayMatching(ref) : screen.getPrimaryDisplay();
  const wa = disp.workArea;
  notifWin.setBounds({ x: wa.x + wa.width - NOTIF_W, y: wa.y, width: NOTIF_W, height: wa.height });
}

function sendToast(t) {
  if (!notifWin || notifWin.isDestroyed()) return;
  placeNotifWin();
  if (!notifWin.isVisible()) notifWin.showInactive();
  notifWin.webContents.send('toast:add', t);
}

function notify(payload, sourceTab = null) {
  const t = {
    id: ++notifSeq,
    title: String(payload.title || 'Nautilus').slice(0, 140),
    body: String(payload.body || '').slice(0, 600),
    kind: ['success', 'error', 'warn', 'info'].includes(payload.kind) ? payload.kind : 'info',
    source: sourceTab ? SERVICES[sourceTab].label : 'Nautilus',
    time: Date.now(),
  };
  pending.set(t.id, { source: sourceTab, action: payload.action || null });
  if (win && !win.isFocused()) win.flashFrame?.(true);
  if (notifReady) sendToast(t); else notifQueue.push(t);
  log(`notify [${t.kind}] ${t.title} - ${t.body.replace(/\n/g, ' ')}`);
}

function tabOfSender(wc) {
  return TABS.find(id => views[id] && views[id].webContents === wc) || null;
}

ipcMain.on('desktop:notify', (e, payload) => notify(payload || {}, tabOfSender(e.sender)));

ipcMain.on('notif:click', (_e, id) => {
  const p = pending.get(id);
  pending.delete(id);
  if (!win) return;
  if (win.isMinimized()) win.restore();
  win.show(); win.focus(); win.flashFrame?.(false);
  if (p && p.source) {
    showTab(p.source);
    if (p.action) views[p.source].webContents.send('desktop:action', p.action);
  }
});
ipcMain.on('notif:dismiss', (_e, id) => pending.delete(id));
ipcMain.on('notif:hover', (_e, over) => {
  if (notifWin && !notifWin.isDestroyed()) notifWin.setIgnoreMouseEvents(!over, { forward: true });
});
ipcMain.on('notif:empty', () => {
  if (notifWin && !notifWin.isDestroyed()) {
    notifWin.setIgnoreMouseEvents(true, { forward: true });
    notifWin.hide();
  }
});

// tab bar
ipcMain.on('shell:tab', (_e, id) => showTab(id));
ipcMain.on('shell:reload', () => { const v = views[state.active]; if (v) v.webContents.reload(); });
ipcMain.on('shell:retry', async (_e, id) => {
  if (id === 'redis') { await ensureRedis(); return; }
  if (id === 'node') { await ensureNode(); return; }
  if (!SERVICES[id]) return;
  restarts[id] = [];
  if (await ensureService(id)) views[id].webContents.loadURL(serviceUrl(id));
});
ipcMain.on('shell:logs', () => shell.openPath(LOG_DIR));
ipcMain.handle('shell:get', () => ({ redis: state.redis, backtester: state.backtester,
                                      live: state.live, node: state.node,
                                      detail: state.detail, active: state.active,
                                      version: app.getVersion() }));
ipcMain.on('shell:test-notify', () => notify({ title: 'Notifications are working',
  body: 'Backtest results, report builds and data syncs show up here.', kind: 'success' }));

// ---------------------------------------------------------------------------
// health: keep the status dots honest and catch servers we didn't start
// ---------------------------------------------------------------------------
async function healthLoop() {
  let tick = 0;
  while (!quitting) {
    await sleep(5000);
    if (START_NODE && state.node !== 'wait' && ++tick % 3 === 0) {   // process scan is ~1s: every 15s
      const up = await nodeRunning();
      if (up !== (state.node === 'ok')) {
        setState('node', up ? 'ok' : 'fail', up ? 'running' : 'not running - click to start');
        if (!up) notify({ title: 'Trading node stopped', body: 'The live data/trading node is no longer running. Click the Node dot to start it.', kind: 'warn' });
      }
    }
    const r = await redisPing();
    if (r !== (state.redis === 'ok')) {
      setState('redis', r ? 'ok' : 'fail', r ? 'localhost:6379' : 'not reachable on :6379');
      if (!r) notify({ title: 'Redis is down', body: 'Live data and the dashboard stream need it. Click the Redis dot to restart it.', kind: 'error' });
    }
    for (const id of TABS) {
      if (owned[id]) continue;                 // our own children report via 'exit'
      if (state[id] === 'wait') continue;
      const ok = await httpOk(SERVICES[id].port, SERVICES[id].probe);
      if (ok !== (state[id] === 'ok')) setState(id, ok ? 'ok' : 'fail', ok ? `:${SERVICES[id].port}` : 'not answering');
    }
  }
}

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------
async function boot() {
  log(`--- launch ROOT=${ROOT} python=${PY} ${PY_ARGS.join(' ')}`);
  createSplash();
  createNotifWindow();
  await ensureRedis();
  const [bt] = await Promise.all([ensureService('backtester'), ensureService('live'), ensureNode()]);
  if (!bt) log('backtester failed to start - opening anyway with the offline page');
  createMain();
  if (state.redis !== 'ok')
    notify({ title: 'Redis did not start', body: `Check that WSL (${WSL_DISTRO}) has redis-server installed.`, kind: 'error' });
  if (state.live !== 'ok')
    notify({ title: 'Live dashboard unavailable', body: 'The Live tab will retry when you click its status dot.', kind: 'warn' });
  healthLoop();
  if (process.env.NAUTILUS_CAPTURE_DIR) captureViews(process.env.NAUTILUS_CAPTURE_DIR).catch(e => log('capture failed: ' + e.message));
}

// Dev aid: NAUTILUS_CAPTURE_DIR=<folder> renders the app's own views (tab bar,
// both tabs, the notification overlay) to PNGs a few seconds after startup.
// It captures only this app's pages, never the screen.
async function captureViews(dir) {
  const save = async (name, wc) => {
    const img = await wc.capturePage(undefined, { stayHidden: true, stayAwake: true });
    fs.writeFileSync(path.join(dir, name + '.png'), img.toPNG());
  };
  await sleep(8000);
  await save('bar', barView.webContents);
  await save('backtester', views.backtester.webContents);
  // (no sample notifications here: they show on the user's screen as if real)
  if (notifWin && notifWin.isVisible()) await save('notifications', notifWin.webContents);
  showTab('live');
  await sleep(3000);
  await save('live', views.live.webContents);
  showTab('backtester');
  log('captured views to ' + dir);
}

app.on('second-instance', () => {
  if (win) { if (win.isMinimized()) win.restore(); win.show(); win.focus(); }
  else if (splash) splash.focus();
});

app.whenReady().then(boot);

app.on('before-quit', () => {
  quitting = true;
  for (const id of Object.keys(owned)) killTree(owned[id]);
  if (keepalive) { try { keepalive.kill(); } catch (e) { /* ignore */ } }
});

app.on('window-all-closed', () => app.quit());
