// Bridge between the pages (dashboards + the app's own shell pages) and the
// main process. Pages run sandboxed with no Node access. This is the whole
// surface they get.
const { contextBridge, ipcRenderer } = require('electron');

const on = (channel, cb) => {
  const h = (_e, payload) => cb(payload);
  ipcRenderer.on(channel, h);
  return () => ipcRenderer.removeListener(channel, h);
};

contextBridge.exposeInMainWorld('desktop', {
  isDesktop: true,
  // {title, body, kind: 'success'|'error'|'warn'|'info', action?: {...}}
  notify: payload => ipcRenderer.send('desktop:notify', payload),
  // action from a clicked notification this page raised
  onAction: cb => on('desktop:action', cb),

  // used by shell.html / splash.html / offline.html / notifications.html
  shell: {
    get: () => ipcRenderer.invoke('shell:get'),
    onState: cb => on('shell:state', cb),
    tab: id => ipcRenderer.send('shell:tab', id),
    reload: () => ipcRenderer.send('shell:reload'),
    retry: id => ipcRenderer.send('shell:retry', id),
    logs: () => ipcRenderer.send('shell:logs'),
    testNotify: () => ipcRenderer.send('shell:test-notify'),
  },
  toasts: {
    onAdd: cb => on('toast:add', cb),
    click: id => ipcRenderer.send('notif:click', id),
    dismiss: id => ipcRenderer.send('notif:dismiss', id),
    hover: over => ipcRenderer.send('notif:hover', !!over),
    empty: () => ipcRenderer.send('notif:empty'),
  },
});
