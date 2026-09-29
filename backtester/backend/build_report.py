import glob
import html
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import _paths
import nt_engine
import symbol_classes
import pnl_stats
DATA_DIR = os.path.join(_paths.ROOT, 'datas')
OUT_DIR = os.path.join(_paths.ROOT, 'report_out')
SYMBOL_PREF = 'ETHUSDT'
START_CASH, COMMISSION, LEVERAGE = 100_000.0, 0.0002, 10.0
DS = 7000    # downsample the series to this many points (SVG Scatter)

DEFAULT_SELECTION = [
    {'name': 'KalmanTest'},
    {'name': 'ExtendedKalmanTest'},
    {'name': 'EMTest'},
]
PALETTE = ['#4c9be8', '#e8834c', '#5cc98a', '#b58cf0', '#e5566a', '#4fc6c0',
           '#d29922', '#ff7eb6', '#8b96a5', '#58d68d', '#c39bd3', '#5dade2']

LABELS = {'P00': 'Var(position)', 'P11': 'Var(velocity)', 'P22': 'Var(accel)',
          'P01': 'Cov(pos, vel)', 'P02': 'Cov(pos, accel)', 'P12': 'Cov(vel, accel)'}


def pick_symbol(prefer, data_dir=DATA_DIR, suffix='1m'):
    paths = sorted(glob.glob(os.path.join(data_dir, '*-%s.csv' % suffix)))
    if not paths:
        raise ValueError('no %s CSVs in %s — run a %s backtest or sync first'
                         % (suffix, data_dir, suffix))
    hit = [p for p in paths if os.path.basename(p).startswith('%s-%s' % (prefer, suffix))]
    p = hit[0] if hit else paths[0]
    return os.path.basename(p)[:-len('-%s.csv' % suffix)], p


def _window_start(mn, mx, days):
    if not days:
        return mn
    from datetime import timedelta
    return max(mn, mx - timedelta(days=days))


def _ensure_symbol_csv(prefer, timeframe, days=None):
    import run_backtest
    data_dir, suffix, _ = run_backtest._tf_cfg(timeframe)
    os.makedirs(data_dir, exist_ok=True)
    try:
        import db
        conn = db.get_conn()
    except Exception as e:
        print('[report] DB unavailable (%s) — using existing %s CSVs' % (e, suffix))
        return pick_symbol(prefer, data_dir, suffix)
    try:
        syms = db.symbols(conn, timeframe)
        sym = (next((s for s in syms if s == prefer), None)
               or next((s for s in syms if s.startswith(prefer)), None)
               or (syms[0] if syms else None))
        if not sym:
            raise ValueError('database has no %s bars' % timeframe)
        mn, mx = db.bars_span(conn, timeframe)
        start = _window_start(mn, mx, days)
        path = os.path.join(data_dir, '%s-%s.csv' % (sym, suffix))
        n = db.export_bars_csv(conn, sym, start, mx, path, interval=timeframe)
        print('[report] exported %s %s bars for %s (%s -> %s)'
              % (n, timeframe, sym, start, mx))
        return sym, path
    finally:
        conn.close()


def _ensure_all_csvs(timeframe, days=None):
    import run_backtest
    data_dir, suffix, _ = run_backtest._tf_cfg(timeframe)
    os.makedirs(data_dir, exist_ok=True)
    existing = lambda: [(os.path.basename(p)[:-len('-%s.csv' % suffix)], p)
                        for p in sorted(glob.glob(os.path.join(data_dir, '*-%s.csv' % suffix)))]
    try:
        import db
        conn = db.get_conn()
    except Exception as e:
        print('[report] DB unavailable (%s): using existing %s CSVs' % (e, suffix))
        return existing()
    try:
        syms = db.symbols(conn, timeframe)
        if not syms:
            raise ValueError('database has no %s bars' % timeframe)
        mn, mx = db.bars_span(conn, timeframe)
        start = _window_start(mn, mx, days)
        feeds = []
        for sym in syms:
            path = os.path.join(data_dir, '%s-%s.csv' % (sym, suffix))
            if db.export_bars_csv(conn, sym, start, mx, path, interval=timeframe):
                feeds.append((sym, path))
        print('[report] exported %d symbols (%s bars, %s -> %s)'
              % (len(feeds), timeframe, start, mx))
        return feeds
    finally:
        conn.close()


def _csv_frame(path):
    df = pd.read_csv(path)
    ts = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S', utc=True)
    return df[['open', 'high', 'low', 'close', 'volume']].set_index(ts)


def run_strategy(label, Strat, params, feeds, color, compression=1):
    """One strategy over the given CSV feeds on the Nautilus engine."""
    frames = {sym: _csv_frame(path) for sym, path in feeds}
    driver, strat, engine = nt_engine.build_and_run(
        frames, Strat, dict(params, diag=(len(feeds) == 1)),
        START_CASH, LEVERAGE, COMMISSION, bar_minutes=compression)
    end = driver.getvalue()
    engine.dispose()
    rows = strat._diag.get(feeds[0][0], []) if len(feeds) == 1 else []
    if rows:
        diag = pd.DataFrame(rows)
        diag['dt'] = pd.to_datetime(diag['t'], unit='s', utc=True)
        diag = diag.set_index('dt')
    else:
        diag = None
    eq = pd.DataFrame(strat.equity, columns=['dt', 'value'])
    eq['dt'] = pd.to_datetime(eq['dt']); eq = eq.set_index('dt')
    tl = strat.trade_log
    wins = sum(1 for t in tl if t['pnlcomm'] > 0)
    rms = None
    if diag is not None:
        innov = diag['innov'].to_numpy(float)
        rms = round(float(np.sqrt(np.mean(innov ** 2))), 4)
    stats = dict(end_value=round(end, 2), pnl=round(end - START_CASH, 2),
                 trades=len(tl), trades_total=sum(len(v) for v in strat.executed.values()),
                 wins=wins,
                 win_rate=round(100.0 * wins / len(tl), 2) if tl else 0.0,
                 innov_rms=rms)
    return dict(name=label, diag=diag, equity=eq, stats=stats, color=color)


def ds(index, y):
    y = np.asarray(y, float)
    step = max(1, len(y) // DS)
    return index[::step], y[::step]


LAYOUT = dict(template='plotly_dark', hovermode='x',
              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,24,39,0.4)',
              margin=dict(l=60, r=20, t=40, b=40),
              legend=dict(orientation='h', y=1.06, x=0),
              font=dict(family='system-ui, sans-serif', size=12),
              hoverlabel=dict(bgcolor='#1F2937', bordercolor='#38BDF8',
                              font=dict(color='#F3F4F6', size=12)))


def fig_pred(results):
    fig = make_subplots(rows=len(results), cols=1, vertical_spacing=0.07,
                        subplot_titles=[r['name'] for r in results])
    for i, res in enumerate(results, 1):
        d = res['diag']
        x, price = ds(d.index, d['price'].values)
        _, pred = ds(d.index, d['pred'].values)
        _, up = ds(d.index, d['upper'].values)
        _, lo = ds(d.index, d['lower'].values)
        show = (i == 1)
        fig.add_trace(go.Scatter(x=x, y=up, line=dict(width=0), hoverinfo='skip',
                                 showlegend=False, name='upper'), row=i, col=1)
        fig.add_trace(go.Scatter(x=x, y=lo, fill='tonexty', line=dict(width=0),
                                 fillcolor='rgba(120,165,215,0.40)', hoverinfo='skip',
                                 name='±kσ band', showlegend=show), row=i, col=1)
        fig.add_trace(go.Scatter(x=x, y=price, line=dict(color='#9aa4b8', width=1),
                                 name='price', showlegend=show,
                                 hovertemplate='price %{y:.2f}<extra></extra>'), row=i, col=1)
        fig.add_trace(go.Scatter(x=x, y=pred, line=dict(color='#e8590c', width=1.2),
                                 name='prediction', showlegend=show,
                                 hovertemplate='pred %{y:.2f}<extra></extra>'), row=i, col=1)
        fig.update_yaxes(title_text='price', row=i, col=1)
    fig.update_layout(height=300 * len(results), **LAYOUT)
    return fig


def fig_innov(results):
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.14,
                        subplot_titles=['Innovation (price − prediction)',
                                        'Innovation uncertainty √S'])
    for res in results:
        d = res['diag']
        x, inn = ds(d.index, d['innov'].values)
        _, s = ds(d.index, np.sqrt(d['S'].values))
        fig.add_trace(go.Scatter(x=x, y=inn, line=dict(width=0.9, color=res['color']),
                                 name='%s · RMS %.4g' % (res['name'], res['stats']['innov_rms']),
                                 hovertemplate='%{y:.3f}<extra></extra>'), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=s, line=dict(width=0.9, color=res['color']),
                                 name=res['name'], showlegend=False,
                                 hovertemplate='√S %{y:.3f}<extra></extra>'), row=2, col=1)
    fig.add_hline(y=0, row=1, col=1, line=dict(color='gray', width=0.5))
    fig.update_layout(height=560, **LAYOUT)
    return fig


def fig_P(res):
    d = res['diag']
    terms = [c for c in d.columns if c.startswith('P')]
    cols = 3
    rows = int(np.ceil(len(terms) / cols))
    fig = make_subplots(rows=rows, cols=cols, vertical_spacing=0.16, horizontal_spacing=0.08,
                        subplot_titles=['%s · %s' % (c, LABELS.get(c, c)) for c in terms])
    for i, c in enumerate(terms):
        r, cc = i // cols + 1, i % cols + 1
        x, y = ds(d.index, d[c].values)
        fig.add_trace(go.Scatter(x=x, y=y, line=dict(color='#e5566a', width=1),
                                 name=c, showlegend=False,
                                 hovertemplate='%{y:.4g}<extra></extra>'), row=r, col=cc)
        tail = d[c].iloc[200:].to_numpy(float)
        lo, hi = float(np.nanmin(tail)), float(np.nanmax(tail))
        pad = (hi - lo) * 0.2 or (abs(hi) * 0.2) or 1e-9
        fig.update_yaxes(range=[lo - pad, hi + pad], row=r, col=cc)
    fig.update_layout(height=300 * rows, **{k: v for k, v in LAYOUT.items() if k != 'hovermode'},
                      hovermode='closest')
    return fig


def fig_equity(results):
    fig = go.Figure()
    for res in results:
        e = res['equity']
        x, y = ds(e.index, e['value'].values)
        fig.add_trace(go.Scatter(x=x, y=y, line=dict(color=res['color'], width=1.3),
                                 name='%s · PnL %+.0f' % (res['name'], res['stats']['pnl']),
                                 hovertemplate='%{y:.0f}<extra></extra>'))
    fig.add_hline(y=START_CASH, line=dict(color='gray', dash='dash', width=0.6))
    fig.update_layout(height=450, **LAYOUT)
    fig.update_yaxes(title_text='Account Value ($)')
    return fig


def fig_winpnl(results):
    fig = make_subplots(rows=3, cols=1, vertical_spacing=0.12, 
                        subplot_titles=['PnL ($)', 'Win Rate (%)', 'Max Drawdown (%)'])
    names = [r['name'] for r in results]
    bar_c = [r['color'] for r in results]
    
    fig.add_trace(go.Bar(x=names, y=[r['stats']['pnl'] for r in results],
                         marker_color=bar_c, showlegend=False,
                         hovertemplate='$%{y:.2f}<extra></extra>'), row=1, col=1)
    
    fig.add_trace(go.Bar(x=names, y=[r['stats']['win_rate'] for r in results],
                         marker_color=bar_c, showlegend=False,
                         hovertemplate='%{y:.2f}%<extra></extra>'), row=2, col=1)
                         
    dds = []
    for r in results:
        if r.get('summary') and r['summary'].get('max_drawdown_pct') is not None:
            dds.append(-abs(r['summary']['max_drawdown_pct']))
        else:
            eq = r['equity']['value']
            cm = eq.cummax()
            dd = (eq - cm) / cm * 100
            dds.append(-abs(dd.min()) if not dd.empty else 0.0)
            
    fig.add_trace(go.Bar(x=names, y=dds,
                         marker_color=bar_c, showlegend=False,
                         hovertemplate='%{y:.2f}%<extra></extra>'), row=3, col=1)
                         
    fig.update_layout(height=720, template='plotly_dark',
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(17,24,39,0.4)',
                      margin=dict(l=50, r=20, t=40, b=30))
    return fig


_SUMMARY_LABELS = {'trades': 'positions', 'trades_total': 'trades'}

def summary_table(results):
    cols = ['end_value', 'pnl', 'trades', 'trades_total', 'wins', 'win_rate']
    head = ''.join('<th>%s</th>' % _SUMMARY_LABELS.get(c, c) for c in cols)
    body = ''
    for res in results:
        s = res['stats']
        cells = ''
        for c in cols:
            v = s[c]
            cls = ' pos' if c == 'pnl' and isinstance(v, (int, float)) and v > 0 \
                else ' neg' if c == 'pnl' and isinstance(v, (int, float)) and v < 0 else ''
            cells += '<td class="%s">%s</td>' % (cls.strip(), '&mdash;' if v is None else v)
        body += '<tr><th class="rowh">%s</th>%s</tr>' % (res['name'], cells)
        
    if len(results) == 2:
        s0, s1 = results[0]['stats'], results[1]['stats']
        cells = ''
        for c in cols:
            v0, v1 = s0[c], s1[c]
            if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
                delta = v0 - v1
                if c in ['end_value', 'pnl']:
                    delta = round(delta, 2)
                cls = 'pos' if delta > 0 else 'neg' if delta < 0 else ''
                fmt = '%+.2f' if c == 'win_rate' else '%+g'
                cells += '<td class="delta %s">%s</td>' % (cls, fmt % delta)
            else:
                cells += '<td class="delta">&mdash;</td>'
        body += '<tr class="delta-row"><th class="rowh">Δ (%s &minus; %s)</th>%s</tr>' % (results[0]['name'], results[1]['name'], cells)

    return '<div class="table-container"><table><thead><tr><th></th>%s</tr></thead><tbody>%s</tbody></table></div>' % (head, body)

_PSTAT_ROWS = [
    ('End value (USDT)',        'end_value',                    lambda v: '%.2f' % v),
    ('PnL (USDT)',              'pnl',                          lambda v: '%+.2f' % v),
    ('Return %',                'return_pct',                   lambda v: '%+.3f%%' % v),
    ('Positions',               'trades_closed',                lambda v: '%d' % v),
    ('Trades',                  'trades_total',                 lambda v: '%d' % v),
    ('Win rate %',              'win_rate_pct',                 lambda v: '%.1f%%' % v),
    ('Max drawdown %',          'max_drawdown_pct',             lambda v: '%.2f%%' % v),
    ('Ann. return (CAGR)',      'annual_return_cagr',           lambda v: '%+.2f%%' % (100*v)),
    ('Ann. volatility (ret)',   'annual_volatility_returns',    lambda v: '%.2f%%' % (100*v)),
    ('Ann. volatility (log)',   'annual_volatility_log_returns', lambda v: '%.2f%%' % (100*v)),
    ('Sharpe (arithmetic)',     'sharpe_arithmetic',            lambda v: '%.4f' % v),
    ('Sharpe (log)',            'sharpe_log_returns',           lambda v: '%.4f' % v),
]

def stats_table(results):
    runs = [r for r in results if r.get('summary')]
    if not runs:
        return ''
        
    is_pair = len(runs) == 2
    head = ''.join('<th>%s</th>' % r['name'] for r in runs)
    
    if is_pair:
        head += '<th>Δ (%s &minus; %s)</th>' % (runs[0]['name'], runs[1]['name'])

    body = ''
    for label, key, fmt in _PSTAT_ROWS:
        cells = ''
        vals = []
        for r in runs:
            v = r['summary'].get(key)
            vals.append(v)
            missing = v is None or v != v
            cls = '' if missing or not isinstance(v, (int, float)) else (' pos' if v > 0 else ' neg' if v < 0 else '')
            try:
                cells += '<td class="%s">%s</td>' % (cls.strip(), '&mdash;' if missing else fmt(v))
            except (TypeError, ValueError):
                cells += '<td>%s</td>' % v
                
        if is_pair:
            v0, v1 = vals[0], vals[1]
            if isinstance(v0, (int, float)) and isinstance(v1, (int, float)) and v0 == v0 and v1 == v1:
                delta = v0 - v1
                cls = 'pos' if delta > 0 else 'neg' if delta < 0 else ''
                try:
                    cells += '<td class="delta %s">%s</td>' % (cls, fmt(delta))
                except:
                    cells += '<td class="delta">&mdash;</td>'
            else:
                cells += '<td class="delta">&mdash;</td>'

        body += '<tr><th class="rowh">%s</th>%s</tr>' % (label, cells)
        
    return ('<div class="table-container">'
            '<table class="statstable"><thead><tr><th></th>%s</tr></thead><tbody>%s</tbody></table>'
            '</div>' % (head, body))


def _sym_compare_html(runs):
    runs = [r for r in runs if r.get('symdata')]
    if not runs:
        return ''
    syms = sorted(set().union(*[set(r['symdata']) for r in runs],
                              *[set(r.get('symstats') or {}) for r in runs]))
    if not syms:
        return ''
  
    sync_ctl = """
  <label style="font-size:13px;color:var(--ink);cursor:pointer;display:inline-flex;align-items:center;gap:6px;font-weight:600;">
    <input type="checkbox" id="symsync" checked> Sync Chart Pan & Zoom
  </label>
    """
  
    payload = json.dumps({r['name']: r['symdata'] for r in runs})
    statspayload = json.dumps({r['name']: (r.get('symstats') or {}) for r in runs})
    meta = json.dumps([{'name': r['name'], 'color': r['color']} for r in runs])
    options = ''.join('<option>%s</option>' % html.escape(s) for s in syms)
    
    run_checks = ''.join(
        '<label style="margin-right:20px;font-size:13px;color:%s;cursor:pointer;font-weight:600;display:inline-flex;align-items:center;gap:6px;">'
        '<input type="checkbox" class="symruncb" data-run="%s" checked> %s</label>'
        % (r['color'], html.escape(r['name'], quote=True), html.escape(r['name']))
        for r in runs)
        
    return """
<p class="note" style="margin-top:0;">Stats from the full position logs · one chart panel per run · drag to zoom.</p>
<div style="display: flex; flex-wrap: wrap; align-items: center; margin-bottom: 24px; padding: 16px 20px; background: rgba(255,255,255,0.02); border-radius: 8px; border: 1px solid var(--line);">
  
  <div style="display: flex; align-items: center; width: 100%%; gap: 32px; flex-wrap: wrap;">
      <select id="symsel" style="background:#111827;color:#F3F4F6;border:1px solid #374151;border-radius:6px;padding:8px 12px;font-size:14px;outline:none;cursor:pointer">%s</select>
      <div style="display: flex; align-items: center; gap: 24px;">
          %s
      </div>
  </div>
  
  <div id="symrunpicker" style="flex-basis: 100%%; margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--line);">
      <div style="margin-bottom: 12px; font-size: 11px; color: var(--muted); text-transform: uppercase; font-weight: 600; letter-spacing: 0.1em;">Compare Strategies</div>
      %s
  </div>
  
  <div id="addl_picker_container" style="flex-basis: 100%%; margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--line); display: none;">
      <div style="margin-bottom: 12px; font-size: 11px; color: var(--muted); text-transform: uppercase; font-weight: 600; letter-spacing: 0.1em;">Additional Data Subplots</div>
      <div id="addl_picker" style="display:flex; flex-wrap: wrap; gap: 10px;"></div>
  </div>
  
</div>
<div id="symstatstbl"></div>

<div id="symcmp-wrapper" style="position: relative; border-radius: 8px; overflow: hidden; border: 1px solid var(--line); background: var(--panel);">
    <button id="symcmp-expand" style="position: absolute; top: 12px; right: 12px; z-index: 100; background: var(--panel); border: 1px solid var(--line); color: var(--ink); padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 16px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); transition: all 0.2s;" title="Toggle Fullscreen">&#9974;</button>
    <div id="symcmp"></div>
</div>

<script>
const SYMDATA = %s;
const SYMSTATS = %s;
const SYMRUNS = %s;
const SYMMETRICS = [['Realised','pnl',2],['Unrealised','unrealised',2],['Total','total',2],
                    ['Positions','trades',0],['Win %%','win_rate',1],
                    ['Sharpe','sharpe',3],['Max DD','max_dd',2]];

let addlState = {};

function renderAddlCheckboxes(sym) {
    const shownRuns = new Set([...document.querySelectorAll('.symruncb:checked')].map(el => el.dataset.run));
    const addlNames = new Set();
    
    SYMRUNS.filter(r => shownRuns.has(r.name)).forEach(r => {
        const d = (SYMDATA[r.name] || {})[sym];
        if (d && d.lines) {
            d.lines.forEach(L => { if (L.additional) addlNames.add(L.name); });
        }
    });
    
    const containerBlock = document.getElementById('addl_picker_container');
    const pickerDiv = document.getElementById('addl_picker');
    
    if (addlNames.size === 0) {
        containerBlock.style.display = 'none';
        return;
    }
    
    containerBlock.style.display = 'block';
    let html = '';
    
    [...addlNames].sort().forEach(name => {
        if (addlState[name] === undefined) addlState[name] = false;
        const checked = addlState[name] ? 'checked' : '';
        html += `<label style="font-size:12px;color:var(--ink);cursor:pointer;display:inline-flex;align-items:center;gap:6px; background: rgba(255,255,255,0.05); padding: 6px 12px; border-radius: 6px; border: 1px solid var(--line); transition: border-color 0.2s;">
            <input type="checkbox" class="addlcb" value="${name}" ${checked}> ${name}
        </label>`;
    });
    
    pickerDiv.innerHTML = html;
    
    document.querySelectorAll('.addlcb').forEach(cb => {
        cb.onchange = e => {
            addlState[e.target.value] = e.target.checked;
            drawSym(document.getElementById('symsel').value);
        };
    });
}

function symStatsTable(sym) {
  const shown = new Set([...document.querySelectorAll('.symruncb:checked')].map(el => el.dataset.run));
  const runs = SYMRUNS.filter(r => shown.has(r.name));
  const rows = runs.map(r => (SYMSTATS[r.name] || {})[sym] || null);
  if (!rows.some(Boolean)) { document.getElementById('symstatstbl').innerHTML = ''; return; }
  const fmtc = (v, dp) => v == null ? '\\u2014' : Number(v).toFixed(dp);
  
  let head = '<tr><th></th>' + runs.map(r => '<th>' + r.name + '</th>').join('');
  const isPair = runs.length === 2;
  if (isPair) head += '<th>Δ (' + runs[0].name + ' &minus; ' + runs[1].name + ')</th>';
  head += '</tr>';
  
  const body = SYMMETRICS.map(([label, key, dp]) => {
    const vals = rows.map(r => r ? r[key] : null);
    const nums = vals.filter(v => typeof v === 'number');
    let bestV = null, worstV = null;
    if (nums.length > 1 && new Set(nums).size > 1) { bestV = Math.max(...nums); worstV = Math.min(...nums); }
    let cells = vals.map(v => {
      const cls = v == null ? '' : v === bestV ? 'pos' : v === worstV ? 'neg' : '';
      return '<td class="' + cls + '">' + fmtc(v, dp) + '</td>';
    }).join('');
    
    if (isPair) {
        const v0 = vals[0], v1 = vals[1];
        if (typeof v0 === 'number' && typeof v1 === 'number') {
            const delta = v0 - v1;
            const cls = delta > 0 ? 'pos' : delta < 0 ? 'neg' : '';
            const sign = delta > 0 ? '+' : '';
            cells += '<td class="delta ' + cls + '">' + sign + fmtc(delta, dp) + '</td>';
        } else {
            cells += '<td class="delta">&mdash;</td>';
        }
    }
    return '<tr><th class="rowh">' + label + '</th>' + cells + '</tr>';
  }).join('');
  document.getElementById('symstatstbl').innerHTML =
    '<div class="table-container" style="margin-bottom: 24px;"><table><thead>' + head + '</thead><tbody>' + body + '</tbody></table></div>';
}

function drawSym(sym) {
  symStatsTable(sym);
  const doSync = document.getElementById('symsync').checked;
  const shown = new Set([...document.querySelectorAll('.symruncb:checked')].map(el => el.dataset.run));
  const traces = [], ann = [];
  const layoutAxes = {};
  
  const runsToDraw = [];
  SYMRUNS.filter(run => shown.has(run.name)).forEach(run => {
      const d = (SYMDATA[run.name] || {})[sym];
      if (d) {
          const addlLines = d.lines.filter(L => L.additional && addlState[L.name]);
          runsToDraw.push({ run: run, d: d, addlLines: addlLines });
      }
  });

  if (runsToDraw.length === 0) {
      document.getElementById('symcmp').innerHTML = '<div style="padding: 48px 0; text-align: center; color: var(--muted); border: 1px dashed var(--line); border-radius: 8px; font-weight: 500;">No chart data recorded for this symbol (0 positions).</div>';
      return;
  }
  
  let rowCount = 0;
  let totalUnits = 0;
  const runSpacing = 0.4; 
  
  runsToDraw.forEach(item => {
      item.priceRow = ++rowCount;
      item.addlRows = item.addlLines.map(() => ++rowCount);
      item.units = 2.0 + item.addlLines.length * 0.8;
      totalUnits += item.units + runSpacing;
  });
  totalUnits -= runSpacing; 
  
  let currentUnit = totalUnits;
  const masterX = 'x'; 
  
  runsToDraw.forEach((item, index) => {
      const runName = item.run.name;
      const T = a => a.map(t => new Date(t * 1000));
      
      const priceTop = currentUnit / totalUnits;
      const priceBottom = (currentUnit - 2.0) / totalUnits;
      currentUnit -= 2.0;
      
      const isFirst = item.priceRow === 1;
      const xaPrice = isFirst ? 'x' : 'x' + item.priceRow;
      const yaPrice = isFirst ? 'y' : 'y' + item.priceRow;
      
      layoutAxes['yaxis' + (isFirst ? '' : item.priceRow)] = { 
          domain: [priceBottom, priceTop], 
          title: {text: 'Price', font: {size: 11, color: '#9CA3AF'}},
          fixedrange: false
      };
      
      const targetMatch = (doSync && !isFirst) ? masterX : undefined;
      layoutAxes['xaxis' + (isFirst ? '' : item.priceRow)] = {
          matches: targetMatch,
          showticklabels: item.addlLines.length === 0
      };
      
      traces.push({
          x: T(item.d.price.t), y: item.d.price.v, name: runName + ' Price', 
          xaxis: xaPrice, yaxis: yaPrice, line: {color: '#8a97ad', width: 1.5}, showlegend: false,
          hovertemplate: '%%{y:.5g}<extra></extra>'
      });
      
      item.d.lines.forEach(L => {
          if (!L.additional) {
              traces.push({
                  x: T(L.t), y: L.v, name: L.name, xaxis: xaPrice, yaxis: yaPrice,
                  showlegend: true, line: {color: L.color, width: 1, dash: 'solid'},
                  hovertemplate: '%%{y:.5g}<extra></extra>'
              });
          }
      });
      
      ann.push({
          text: '<b>' + runName + '</b>', showarrow: false, xref: 'paper', x: 0, xanchor: 'left',
          yref: 'paper', y: priceTop, yanchor: 'bottom', yshift: 10,
          font: {size: 14, color: item.run.color}
      });
      
      item.addlLines.forEach((L, idx) => {
          const addlRow = item.addlRows[idx];
          const xaAddl = 'x' + addlRow;
          const yaAddl = 'y' + addlRow;
          
          const addlTop = currentUnit / totalUnits;
          const addlBottom = (currentUnit - 0.8) / totalUnits;
          currentUnit -= 0.8;
          
          layoutAxes['yaxis' + addlRow] = { 
              domain: [addlBottom, addlTop],
              title: {text: L.name, font: {size: 10, color: '#9CA3AF'}}
          };
          
          layoutAxes['xaxis' + addlRow] = { 
              matches: (doSync ? masterX : xaPrice),
              showticklabels: idx === item.addlLines.length - 1
          };
          
          traces.push({
              x: T(L.t), y: L.v, name: runName + ' ' + L.name, xaxis: xaAddl, yaxis: yaAddl,
              showlegend: false, line: {color: L.color, width: 1.2, dash: 'solid'},
              hovertemplate: '%%{y:.5g}<extra></extra>'
          });
      });
      
      currentUnit -= runSpacing; 
  });
  
  const baseHeight = Math.max(350, 100 * totalUnits);
  
  const wrapper = document.getElementById('symcmp-wrapper');
  const isFull = wrapper.classList.contains('fullscreen-chart');
  
  const layout = {
      height: isFull ? window.innerHeight - 80 : baseHeight,
      annotations: ann,
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(17,24,39,0.4)',
      font: {color: '#F3F4F6', size: 12}, hovermode: 'x unified',
      margin: {l: 60, r: 20, t: 40, b: 30},
      legend: {orientation: 'h', y: 1.02, yanchor: 'bottom'},
      dragmode: 'zoom'
  };
  Object.assign(layout, layoutAxes);
  
  Plotly.react('symcmp', traces, layout, {responsive: true, scrollZoom: true});
  
  const expandBtn = document.getElementById('symcmp-expand');
  expandBtn.replaceWith(expandBtn.cloneNode(true)); 
  const newExpandBtn = document.getElementById('symcmp-expand');
  
  newExpandBtn.onclick = function() {
      const isFull = wrapper.classList.toggle('fullscreen-chart');
      this.innerHTML = isFull ? '&#10006;' : '&#9974;';
      const newHeight = isFull ? window.innerHeight - 80 : baseHeight;
      Plotly.relayout('symcmp', { height: newHeight });
  };
}

document.getElementById('symsel').onchange = () => {
    renderAddlCheckboxes(document.getElementById('symsel').value);
    drawSym(document.getElementById('symsel').value);
};
document.getElementById('symsync').onchange = () => drawSym(document.getElementById('symsel').value);
document.querySelectorAll('.symruncb').forEach(cb =>
  cb.onchange = () => {
      renderAddlCheckboxes(document.getElementById('symsel').value);
      drawSym(document.getElementById('symsel').value);
  });

// Initialize on Load
renderAddlCheckboxes(document.getElementById('symsel').value);
drawSym(document.getElementById('symsel').value);
</script>""" % (options, sync_ctl, run_checks, payload, statspayload, meta)


def _class_compare_html(runs):
    runs = [r for r in runs if r.get('classdata')]
    if not runs:
        return ''
    classes = sorted(set().union(*[set(r['classdata']) for r in runs]))
    if not classes:
        return ''
    payload = json.dumps({r['name']: r['classdata'] for r in runs})
    names = json.dumps([r['name'] for r in runs])
    metrics = json.dumps(_CLASS_METRICS)
    options = ''.join('<option>%s</option>' % html.escape(c) for c in classes)
    return """
<p class="note" style="margin-top:0;">Every symbol of a sector/category (Layer 1, DeFi, Meme, ...) pooled into one
trade-level line, so win rate/Sharpe/max DD are computed correctly rather than averaged
across the class's symbols.</p>
<div style="margin-bottom: 24px;">
  <select id="classsel" style="background:#111827;color:#F3F4F6;border:1px solid #374151;
   border-radius:6px;padding:8px 12px;font-size:14px;outline:none;cursor:pointer">%s</select>
</div>
<div id="classcmp"></div>
<script>
const CLASSDATA = %s;
const CLASSRUNS = %s;
const CLASSMETRICS = %s;
function fmtc(v, dp) { return v == null ? '\\u2014' : Number(v).toFixed(dp); }
function drawClass(cls) {
  const rows = CLASSRUNS.map(name => (CLASSDATA[name] || {})[cls] || null);
  const symsUnion = [...new Set(rows.filter(Boolean).flatMap(r => r.symbols))].sort();
  const isPair = CLASSRUNS.length === 2;
  
  let head = '<tr><th></th>' + CLASSRUNS.map(n => '<th>' + n + '</th>').join('');
  if (isPair) head += '<th>Δ (' + CLASSRUNS[0] + ' &minus; ' + CLASSRUNS[1] + ')</th>';
  head += '</tr>';
  
  let body = CLASSMETRICS.map(([label, key, dp]) => {
    const vals = rows.map(r => r ? r[key] : null);
    const nums = vals.filter(v => typeof v === 'number');
    let bestV = null, worstV = null;
    if (nums.length > 1 && new Set(nums).size > 1) {
      bestV = Math.max(...nums); worstV = Math.min(...nums);
    }
    let cells = vals.map(v => {
      const cls2 = v == null ? '' : v === bestV ? 'pos' : v === worstV ? 'neg' : '';
      return '<td class="' + cls2 + '">' + fmtc(v, dp) + '</td>';
    }).join('');
    
    if (isPair) {
        const v0 = vals[0], v1 = vals[1];
        if (typeof v0 === 'number' && typeof v1 === 'number') {
            const delta = v0 - v1;
            const cls2 = delta > 0 ? 'pos' : delta < 0 ? 'neg' : '';
            const sign = delta > 0 ? '+' : '';
            cells += '<td class="delta ' + cls2 + '">' + sign + fmtc(delta, dp) + '</td>';
        } else {
            cells += '<td class="delta">&mdash;</td>';
        }
    }
    
    return '<tr><th class="rowh">' + label + '</th>' + cells + '</tr>';
  }).join('');
  document.getElementById('classcmp').innerHTML =
    '<p class="note" style="margin-bottom: 16px;">' + symsUnion.length + ' symbols: ' + symsUnion.join(', ') + '</p>' +
    '<div class="table-container"><table><thead>' + head + '</thead><tbody>' + body + '</tbody></table></div>';
}
document.getElementById('classsel').onchange = e => drawClass(e.target.value);
drawClass(document.getElementById('classsel').value);
</script>""" % (options, payload, names, metrics)


def div(fig, first=False):
    return pio.to_html(fig, full_html=False, include_plotlyjs=('cdn' if first else False),
                       config={'displayModeBar': True, 'responsive': True, 'scrollZoom': True})


def build(symbol=SYMBOL_PREF, selections=None, timeframe='1m', days=None):
    import run_backtest
    timeframe = timeframe if timeframe in ('1m', '1h') else '1m'
    _, _, comp = run_backtest._tf_cfg(timeframe)
    if symbol:
        sym, path = _ensure_symbol_csv(symbol, timeframe, days)
        feeds, subtitle = [(sym, path)], sym
    else:
        feeds = _ensure_all_csvs(timeframe, days)
        if not feeds:
            raise ValueError('no %s data available' % timeframe)
        subtitle = 'all symbols (%d)' % len(feeds)
    print('[report] %s  timeframe: %s' % (subtitle, timeframe))
    results = []
    for i, sel in enumerate(selections or DEFAULT_SELECTION):
        name = sel.get('name')
        if name not in run_backtest.STRATEGIES:
            print('[report] skipping unknown strategy %r' % name)
            continue
        cls, params, _days = run_backtest._resolve_run_config(
            name, sel.get('params'), None)
        color = PALETTE[i % len(PALETTE)]
        print('[report] running %s ...' % name)
        res = run_strategy(name, cls, params, feeds, color, comp)
        res['params'] = dict(params, timeframe=timeframe)
        results.append(res)
    if not results:
        raise ValueError('no valid strategies selected')
    print('[report] building interactive figures ...')
    symbols = [f[0] for f in feeds]
    _render_page(results, subtitle=subtitle, symbols=symbols)


def _render_page(results, subtitle, symbols=None):
    with_diag = [r for r in results if r.get('diag') is not None]
    ref = with_diag[0]['diag'] if with_diag else results[0]['equity']
    
    start_dt = ref.index[0]
    end_dt = ref.index[-1]
    span_short = '%s → %s' % (start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d'))
    span_long = '%s &mdash; %s' % (start_dt.strftime('%b %d, %Y'), end_dt.strftime('%b %d, %Y'))

    tfs = sorted({str((r.get('params') or {}).get('timeframe', '1m')) for r in results})
    tf_badge = f'<span style="display:inline-block; margin-left:8px; font-size:0.85rem; color:var(--muted); background:rgba(255,255,255,0.05); padding:2px 8px; border-radius:4px; border:1px solid var(--line);">Interval: {", ".join(tfs)}</span>'

    overview_html = f'''
    <div style="{ 'margin-bottom: 24px; padding-bottom: 24px; border-bottom: 1px solid var(--line);' if symbols else '' }">
        <div style="color:var(--muted); text-transform:uppercase; font-size:0.8rem; font-weight:600; letter-spacing:0.05em; margin-bottom:8px;">Backtest Period</div>
        <div style="display:flex; align-items:center; flex-wrap:wrap; gap:4px;">
            <div style="font-size:1.25rem; font-weight:600; color:var(--ink);">{span_long}</div>
            <div style="color:var(--muted); font-size:0.95rem; margin-left:8px;">({len(ref)} bars)</div>
            {tf_badge}
        </div>
    </div>
    '''
    
    if symbols:
        by_class = {}
        for sym in symbols:
            cls = symbol_classes.classify(sym)
            by_class.setdefault(cls, []).append(sym)
        
        cls_blocks = []
        for cls in sorted(by_class.keys()):
            syms = sorted(by_class[cls])
            sym_spans = ''.join(f'<span style="display:inline-block; background:rgba(255,255,255,0.03); padding:4px 10px; border-radius:6px; margin:3px; font-size:0.85rem; border:1px solid var(--line); font-family: ui-monospace, SFMono-Regular, Consolas, monospace; color:var(--ink);">{html.escape(s)}</span>' for s in syms)
            cls_blocks.append(f'<div style="flex: 1 1 300px; background:rgba(0,0,0,0.2); padding:16px; border-radius:8px; border:1px solid var(--line);"><div style="font-weight:600; color:var(--accent); margin-bottom:12px; font-size:0.9rem; text-transform: uppercase; letter-spacing: 0.05em;">{html.escape(cls)} <span style="color:var(--muted); font-weight:normal; font-size:0.8rem;">({len(syms)})</span></div><div>{sym_spans}</div></div>')
        
        overview_html += f'''
        <div>
            <div style="color:var(--muted); text-transform:uppercase; font-size:0.8rem; font-weight:600; letter-spacing:0.05em; margin-bottom:12px;">Universe Overview ({len(symbols)} Symbols)</div>
            <div style="display:flex; flex-wrap:wrap; gap:16px;">
                {''.join(cls_blocks)}
            </div>
        </div>
        '''

    _skip = {'window_start', 'window_end'}
    keys = sorted({k for r in results for k in (r.get('params') or {})} - _skip)
    if keys:
        head = ''.join('<th>%s</th>' % r['name'] for r in results)
        body = ''
        for k in keys:
            vals = [(r.get('params') or {}).get(k) for r in results]
            differ = len({json.dumps(v, default=str) for v in vals}) > 1
            cells = ''.join(
                '<td%s>%s</td>' % (' style="color:#FBBF24;font-weight:600"' if differ else '',
                                   '&mdash;' if v is None else v)
                for v in vals)
            body += '<tr><th class="rowh">%s</th>%s</tr>' % (k, cells)
        cfg_lines = ('<div class="table-container"><table><thead><tr><th></th>%s</tr></thead><tbody>%s</tbody></table></div>'
                     '<p class="note" style="margin-top:8px">Highlighted values differ between runs</p>'
                     % (head, body))
    else:
        cfg_lines = ''

    parts = []
    if overview_html:
        parts.append(('Backtest Overview', overview_html))
        
    parts.extend([
        ('Summary', summary_table(results)),
        ('Portfolio Stats', stats_table(results)),
        ('Parameters', cfg_lines),
        ('Win Rate, PnL & Max DD', div(fig_winpnl(results), first=True)),
        ('Equity Curve', div(fig_equity(results))),
        ('By Class', _class_compare_html(results)),
        ('Per-symbol Compare', _sym_compare_html(results)),
    ])
    
    if with_diag:
        parts += [
            ('Prediction & Confidence Bands',
             '<p class="note" style="margin-top:0;">Drag-select to zoom into the prediction line and band.</p>'
             + div(fig_pred(with_diag))),
            ('Innovation & Uncertainty',
             '<p class="note" style="margin-top:0;">Prediction error, and the filter\'s expected error (√S).</p>'
             + div(fig_innov(with_diag))),
            ('Covariance Matrix P',
             '<p class="note" style="margin-top:0;">Post-warmup steady state · autoscale to see the initial transient.</p>'),
        ]
        for res in with_diag:
            parts.append((res['name'] + ' (Covariance)', div(fig_P(res))))

    names = ' vs '.join(r['name'] for r in results)
    sections, toc = [], []
    
    for i, (label, p) in enumerate(parts):
        if not p:
            continue
        sec_id = 'sec-%d' % i
        
        is_wide = label in ('Backtest Overview', 'Summary', 'Equity Curve', 'Per-symbol Compare', 'Prediction & Confidence Bands',
                            'Innovation & Uncertainty', 'Portfolio Stats', 'By Class') or 'Covariance' in label
        
        if len(results) >= 4 and label in ('Parameters', 'Win Rate, PnL & Max DD'):
            is_wide = True 
            
        css_class = "rpt-sec full-width" if is_wide else "rpt-sec"
        
        header_html = f'<h2>{html.escape(label)}</h2>' if not label.endswith('(Covariance)') else f'<h3>{html.escape(label)}</h3>'
        sections.append(f'<section class="{css_class}" id="{sec_id}">{header_html}{p}</section>')
        
        if label:
            toc.append(f'<a href="#{sec_id}">{html.escape(label)}</a>')
            
    body = '\n'.join(sections)
    
    toc_html = '<nav class="rpt-toc"><div class="toc-title">Dashboard Sections</div>' + ''.join(toc) + '</nav>' if toc else ''
    
    page_html = TEMPLATE.format(symbol=subtitle, span=span_short, names=names,
                                bars=len(ref), body=body, toc=toc_html)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, 'index.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(page_html)
    print('[report] wrote %s (%.1f MB)' % (out, os.path.getsize(out) / 1e6))


MAX_SYMDATA_SYMBOLS = 250
SYM_DS = 300

def _job_symdata(chart_dir, per_symbol=None, max_symbols=MAX_SYMDATA_SYMBOLS):
    paths = sorted(glob.glob(os.path.join(chart_dir, '*.json')))
    if len(paths) > max_symbols:
        if per_symbol:
            def _rank(p):
                s = per_symbol.get(os.path.basename(p)[:-5]) or {}
                return abs(s.get('pnl') or 0) + (s.get('trades') or 0)
            paths = sorted(paths, key=_rank, reverse=True)[:max_symbols]
        else:
            paths = paths[:max_symbols]
    out = {}
    for p in paths:
        sym = os.path.basename(p)[:-5]
        try:
            with open(p, encoding='utf-8') as f:
                cd = json.load(f)
        except (OSError, ValueError):
            continue
        candles = cd.get('candles') or []
        if len(candles) < 2:
            continue
        st = max(1, len(candles) // SYM_DS)
        entry = {'price': {'t': [c['time'] for c in candles[::st]],
                           'v': [c['close'] for c in candles[::st]]},
                 'lines': []}
        for L in (cd.get('lines') or []):
            pts = L.get('points') or []
            if len(pts) < 2:
                continue
            st = max(1, len(pts) // SYM_DS)
            entry['lines'].append({'name': L.get('name', ''),
                                   'color': L.get('color', '#58a6ff'),
                                   'additional': L.get('kind') == 'additional',
                                   't': [q['time'] for q in pts[::st]],
                                   'v': [q['value'] for q in pts[::st]]})
        out[sym.upper()] = entry
    return out


def _read_unrealised(pdir):
    path = os.path.join(os.path.dirname(pdir.rstrip(r'\/')), 'unrealised.json')
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f) or {}
    except (OSError, ValueError):
        return {}


def _job_symstats(pdir):
    if not os.path.isdir(pdir):
        return None
    by_sym = pnl_stats.read_position_pnls(pdir)
    unreal = _read_unrealised(pdir)
    if not by_sym and not unreal:
        return None
    out = {}
    for sym in set(by_sym) | set(unreal):
        st = pnl_stats.pnl_stats(by_sym.get(sym, []))
        st['unrealised'] = round(float(unreal.get(sym, 0.0)), 4)
        st['total'] = round((st.get('pnl') or 0.0) + st['unrealised'], 4)
        out[sym.upper()] = st
    return out


def _job_classdata(pdir):
    if not os.path.isdir(pdir):
        return None
    by_sym = pnl_stats.read_position_pnls(pdir)
    unreal = _read_unrealised(pdir)
    if not by_sym and not unreal:
        return None
    by_class = {}
    class_syms = {}
    class_unreal = {}
    for sym in set(by_sym) | set(unreal):
        upper_sym = sym.upper()
        cls = symbol_classes.classify(upper_sym)
        by_class.setdefault(cls, []).extend(by_sym.get(sym, []))
        class_syms.setdefault(cls, []).append(upper_sym)
        class_unreal[cls] = class_unreal.get(cls, 0.0) + float(unreal.get(sym, 0.0))
    out = {}
    for cls, pnls in by_class.items():
        st = dict(symbols=sorted(class_syms[cls]), **pnl_stats.pnl_stats(pnls))
        st['unrealised'] = round(class_unreal.get(cls, 0.0), 4)
        st['total'] = round((st.get('pnl') or 0.0) + st['unrealised'], 4)
        out[cls] = st
    return out


def build_from_folders(tags):
    import run_backtest
    import _paths
    tdir = run_backtest.TEST_DATA_DIR
    jdir = os.path.join(_paths.ROOT, 'reports', 'jobs')
    results = []
    for i, tag in enumerate(tags or []):
        if str(tag).startswith('job:'):
            jid = tag[4:]
            snap = None
            try:
                with open(os.path.join(jdir, jid, 'results.json'), encoding='utf-8') as f:
                    snap = json.load(f)
            except (OSError, ValueError):
                print('[report] job %r has no results.json — skipped' % jid)
                continue
            symdata = _job_symdata(os.path.join(jdir, jid, 'chartdata'),
                                   per_symbol=snap.get('per_symbol'))
            label_tag = jid
            classdata = _job_classdata(os.path.join(jdir, jid, 'positions'))
            symstats = _job_symstats(os.path.join(jdir, jid, 'positions'))
        else:
            snap_path = os.path.join(tdir, tag, 'run.json')
            if not os.path.exists(snap_path):
                print('[report] archived run %r has no run.json — skipped' % tag)
                continue
            with open(snap_path, encoding='utf-8') as f:
                snap = json.load(f)
            symdata = None
            label_tag = tag
            classdata = None
            symstats = None
        p, s = snap.get('params', {}), snap.get('summary', {})
        eq = pd.DataFrame(snap.get('equity', []), columns=['dt', 'value'])
        if eq.empty:
            continue
        eq['dt'] = pd.to_datetime(eq['dt']); eq = eq.set_index('dt')
        stats = dict(end_value=s.get('end_value'), pnl=s.get('pnl'),
                     trades=s.get('trades_closed'), trades_total=s.get('trades_total'),
                     wins=s.get('won'),
                     win_rate=s.get('win_rate_pct'), innov_rms=None)
        label = '%s · %s' % (p.get('strategy', '?'), snap.get('name') or label_tag)
        results.append(dict(name=label, diag=None, equity=eq, stats=stats,
                            summary=s, symdata=symdata, classdata=classdata, 
                            symstats=symstats, color=PALETTE[i % len(PALETTE)], params=p))
    if not results:
        raise ValueError('no runs with usable snapshots selected')
    print('[report] comparing %d run(s) ...' % len(results))
    
    all_syms = set()
    for r in results:
        if r.get('symdata'): all_syms.update(r['symdata'].keys())
        if r.get('symstats'): all_syms.update(r['symstats'].keys())
        if r.get('classdata'):
            for cls, dat in r['classdata'].items():
                all_syms.update(dat.get('symbols', []))
    symbols = sorted(list(all_syms))
    
    _render_page(results, subtitle='saved backtests', symbols=symbols)


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{names} · {symbol}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  
  :root {{ 
      --bg: #0B1121; 
      --ink: #F3F4F6; 
      --muted: #9CA3AF; 
      --line: #1F2937; 
      --accent: #38BDF8;
      --panel: #111827; 
      --pos: #34D399; 
      --neg: #F87171; 
      --delta-bg: rgba(56, 189, 248, 0.04);
  }}
  
  * {{ box-sizing: border-box; }}
  
  body {{ 
      margin: 0; 
      background: var(--bg); 
      color: var(--ink);
      font-family: 'Inter', system-ui, -apple-system, sans-serif; 
      line-height: 1.6;
      -webkit-font-smoothing: antialiased;
  }}
  
  /* Core Layout Architecture */
  .dashboard-layout {{ 
      display: flex; 
      gap: 32px; 
      max-width: 1700px; 
      margin: 0 auto; 
      padding: 32px 24px 80px; 
      align-items: flex-start;
  }}
  
  /* Sidebar styles */
  .sidebar {{ 
      flex: 0 0 280px; 
      position: sticky; 
      top: 32px; 
      display: flex; 
      flex-direction: column; 
      gap: 20px; 
      max-height: calc(100vh - 64px); 
      overflow-y: auto;
      padding-right: 8px;
  }}
  
  /* Custom scrollbar for sidebar */
  .sidebar::-webkit-scrollbar {{ width: 6px; }}
  .sidebar::-webkit-scrollbar-track {{ background: transparent; }}
  .sidebar::-webkit-scrollbar-thumb {{ background: var(--line); border-radius: 4px; }}
  
  .sidebar-header p {{ 
      color: var(--muted); 
      margin: 0.5em 0 0; 
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace; 
      font-size: 0.85rem; 
      letter-spacing: 0.02em;
  }}
  
  h1 {{ 
      font-size: 1.75rem; 
      font-weight: 700; 
      letter-spacing: -0.025em; 
      margin: 0 0 4px; 
      background: linear-gradient(to right, #38BDF8, #818CF8);
      -webkit-background-clip: text;
      color: transparent;
  }}
  
  /* Grid Content styles */
  .content-grid {{ 
      flex: 1; 
      display: grid; 
      grid-template-columns: repeat(2, 1fr); 
      gap: 24px; 
      min-width: 0; /* Prevents CSS grid blowout */
  }}
  
  .rpt-sec {{ 
      background: var(--panel); 
      border: 1px solid var(--line); 
      border-radius: 12px;
      padding: 24px 28px; 
      margin: 0; 
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
      transition: transform 0.2s, box-shadow 0.2s;
      overflow: hidden;
  }}
  
  .rpt-sec:hover {{ 
      box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05); 
  }}
  
  .rpt-sec:empty {{ display: none; }}
  .rpt-sec.full-width {{ grid-column: 1 / -1; }}
  
  /* New Cinematic Headers */
  h2 {{ 
      font-size: 1.25rem; 
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin: 0 0 20px; 
      padding-bottom: 12px; 
      border-bottom: 1px solid rgba(255, 255, 255, 0.05); 
      color: #FFFFFF;
      display: flex;
      align-items: center;
  }}
  
  h2::before {{
      content: '';
      display: inline-block;
      width: 6px;
      height: 22px;
      background: linear-gradient(to bottom, #38BDF8, #818CF8);
      margin-right: 12px;
      border-radius: 4px;
      box-shadow: 0 0 10px rgba(56, 189, 248, 0.4);
  }}
  
  h3 {{ 
      font-size: 1rem; 
      color: var(--ink); 
      margin: 0 0 12px; 
      font-weight: 600; 
      text-transform: uppercase;
      letter-spacing: 0.05em;
      display: flex;
      align-items: center;
  }}
  
  h3::before {{
      content: '▹';
      color: var(--accent);
      margin-right: 8px;
      font-size: 1.2rem;
  }}
  
  .table-container {{ overflow-x: auto; width: 100%%; border-radius: 8px; }}
  
  table {{ 
      width: 100%%;
      border-collapse: collapse; 
      margin: 8px 0; 
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace; 
      font-size: 0.9rem; 
  }}
  
  th, td {{ padding: 12px 14px; text-align: right; border-bottom: 1px solid var(--line); }}
  
  thead th {{ 
      color: var(--accent); 
      font-weight: 600; 
      text-transform: uppercase; 
      font-size: 0.75rem; 
      letter-spacing: 0.05em; 
      border-bottom: 2px solid var(--line);
  }}
  
  .rowh {{ text-align: left; color: var(--ink); font-weight: 600; }}
  tbody tr:nth-child(even) {{ background-color: rgba(255, 255, 255, 0.015); }}
  tbody tr:hover {{ background-color: rgba(255, 255, 255, 0.04); transition: background-color 0.2s ease; }}
  tbody td {{ font-variant-numeric: tabular-nums; }}
  
  .note {{ color: var(--muted); font-size: 0.85rem; max-width: 80ch; margin: 4px 0 12px; }}
  .note b {{ color: var(--ink); font-weight: 600; }}
  .plotly-graph-div {{ margin: 0 0 12px; border-radius: 8px; overflow: hidden; border: 1px solid var(--line); }}
  
  .pos {{ color: var(--pos); font-weight: 600; }}
  .neg {{ color: var(--neg); font-weight: 600; }}
  
  /* Delta Styling */
  .delta {{ background-color: var(--delta-bg); border-left: 1px solid rgba(56, 189, 248, 0.2); }}
  tr.delta-row {{ background-color: var(--delta-bg) !important; border-top: 2px solid rgba(56, 189, 248, 0.2); }}
  tr.delta-row th {{ color: var(--accent); }}
  
  /* Search Bar */
  .lookup-bar {{ margin-bottom: 8px; }}
  .lookup-bar input {{ 
      width: 100%%; 
      background: rgba(17, 24, 39, 0.8);
      color: var(--ink); 
      border: 1px solid var(--line); 
      border-radius: 8px;
      padding: 10px 14px; 
      font-size: 0.95rem; 
      box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
      transition: border-color 0.2s, box-shadow 0.2s;
  }}
  .lookup-bar input:focus {{ 
      outline: none;
      border-color: var(--accent); 
      box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.2), inset 0 2px 4px rgba(0,0,0,0.1);
  }}
  .lookup-hint {{ color: var(--muted); font-size: 0.85rem; margin: 8px 0 0; font-weight: 500; }}
  .rpt-sec.rpt-dim {{ opacity: 0.3; filter: grayscale(100%%); }}
  
  /* Navigation TOC */
  .rpt-toc {{ 
      display: flex; 
      flex-direction: column; 
      gap: 6px; 
  }}
  .toc-title {{ 
      font-size: 0.75rem; 
      text-transform: uppercase; 
      letter-spacing: 0.1em; 
      color: var(--muted); 
      margin-bottom: 8px; 
      font-weight: 600; 
      padding-left: 4px; 
  }}
  .rpt-toc a {{ 
      padding: 8px 12px; 
      border-radius: 6px; 
      background: rgba(255,255,255,0.015); 
      border: 1px solid var(--line); 
      color: var(--muted);
      font-size: 0.9rem;
      font-weight: 500;
      text-decoration: none;
      transition: all 0.2s; 
  }}
  .rpt-toc a:hover {{ 
      background: rgba(56, 189, 248, 0.08); 
      border-color: var(--accent); 
      color: var(--accent); 
      transform: translateX(4px); 
  }}
  
  /* Fullscreen Graph Feature */
  .fullscreen-chart {{
      position: fixed !important;
      top: 0 !important;
      left: 0 !important;
      width: 100vw !important;
      height: 100vh !important;
      z-index: 9999 !important;
      background: rgba(11, 17, 33, 0.98) !important;
      backdrop-filter: blur(10px);
      padding: 24px !important;
      display: flex;
      flex-direction: column;
      border-radius: 0 !important;
      border: none !important;
  }}
  .fullscreen-chart #symcmp {{
      flex: 1;
      height: 100%% !important;
  }}
  
  /* Responsive Design */
  @media (max-width: 1024px) {{
      .dashboard-layout {{ flex-direction: column; }}
      .sidebar {{ position: static; max-height: none; width: 100%%; flex: none; padding-right: 0; }}
      .content-grid {{ grid-template-columns: 1fr; }}
  }}
</style></head>
<body>
  <div class="dashboard-layout">
    <aside class="sidebar">
      <div class="sidebar-header">
        <h1>{names}</h1>
        <p>SYMBOL: {symbol}<br>{bars} BARS<br>{span}</p>
      </div>
      <div class="lookup-bar">
        <input type="search" id="rptlookup" placeholder="Search report metrics, params..." autocomplete="off">
        <p class="lookup-hint" id="rptlookuphint"></p>
      </div>
      {toc}
    </aside>
    
    <main class="content-grid">
      {body}
    </main>
  </div>
  
<script>
(function() {{
  const input = document.getElementById('rptlookup');
  const hint = document.getElementById('rptlookuphint');
  const secs = Array.from(document.querySelectorAll('.rpt-sec'));
  function run() {{
    const q = input.value.trim().toLowerCase();
    if (!q) {{ secs.forEach(s => s.classList.remove('rpt-dim')); hint.textContent = ''; return; }}
    let hitSecs = 0, firstHit = null;
    secs.forEach(sec => {{
      const heading = sec.querySelector('h2,h3');
      const text = sec.textContent.toLowerCase();
      const match = text.includes(q);
      sec.classList.toggle('rpt-dim', !match);
      if (match) {{
        hitSecs++;
        if (heading && heading.textContent.toLowerCase().includes(q) && !firstHit) firstHit = sec;
        if (!firstHit) firstHit = sec;
      }}
    }});
    hint.textContent = hitSecs ? `${{hitSecs}} matching section(s)` : 'no matches';
    if (firstHit) firstHit.scrollIntoView({{behavior: 'smooth', block: 'start'}});
  }}
  let t;
  input.addEventListener('input', () => {{ clearTimeout(t); t = setTimeout(run, 150); }});
}})();
</script>
</body></html>"""


if __name__ == '__main__':
    build()