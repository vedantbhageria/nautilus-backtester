"""Extended (nonlinear) EM on the drag-EKF model.

Model (3-state, from ExtendedKalmanFilter.py, with a CONSTANT drag coeff d):

    x = [position, velocity, accel]
    f(x) = F3 @ x - d * [0, 0, v|v| / z]^T      (quadratic drag on accel,
    z_t   = x[0] + noise                          normalised by last price z)

Plain EM needs a closed-form E-step, which only exists for linear-Gaussian
models. The standard workaround ("EKS-EM") linearises:

  E-step:  extended Kalman filter + extended RTS smoother + lag-one
           covariance, all using the per-step Jacobian F_t = df/dx.
  M-step:  expectations of the nonlinear residual (x_t - f(x_{t-1})) are
           taken under a first-order expansion of f around the smoothed
           x_s[t-1].  Given d, the ML updates of Q and R stay closed-form.
           The drag coefficient d has no closed form in general -- BUT the
           expected complete-data log-likelihood is exactly QUADRATIC in d
           (f is linear in d), so the gradient equation dJ/dd = 0 solves in
           one Newton step: d* = -beta / (2*alpha).  Gradient descent would
           crawl to the same point; we jump there directly.  (If the drag
           model ever becomes non-quadratic in its parameters, swap the
           one-step solve in _extended_em_fit for a real GD loop -- the
           gradient is assembled from the same M0/M1/M2 statistics.)

Because of the linearisation, the monotone-likelihood guarantee of exact EM
is LOST: loglik should trend up but may wiggle.  Validated instead by (a)
exact reduction to the linear 3-state EM machinery at d=0, and (b) recovery
of a known drag coefficient from synthetic data.
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os.path
import sys
from collections import deque

import bt_compat as bt
import numpy as np

from strategy_base import PortfolioStrategy

_F3 = np.array([[1.0, 1.0, 0.5],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0]])          # constant-acceleration transition
_H3 = np.array([[1.0, 0.0, 0.0]])          # measure position only

_LOG2PI = float(np.log(2.0 * np.pi))

try:
    from numba import njit
    _NUMBA = True
except ImportError:                     # graceful fallback: no-op decorator
    _NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


@njit(cache=True)
def _inv3(A):
    """Explicit 3x3 inverse via the adjugate (np.linalg.inv routes tiny
    matrices through LAPACK -- pure overhead at this size)."""
    a, b, c = A[0, 0], A[0, 1], A[0, 2]
    d, e, f = A[1, 0], A[1, 1], A[1, 2]
    g, h, i = A[2, 0], A[2, 1], A[2, 2]
    A11 = e*i - f*h; A12 = c*h - b*i; A13 = b*f - c*e
    A21 = f*g - d*i; A22 = a*i - c*g; A23 = c*d - a*f
    A31 = d*h - e*g; A32 = b*g - a*h; A33 = a*e - b*d
    det = a*A11 + b*A21 + c*A31
    out = np.empty((3, 3))
    out[0, 0] = A11/det; out[0, 1] = A12/det; out[0, 2] = A13/det
    out[1, 0] = A21/det; out[1, 1] = A22/det; out[1, 2] = A23/det
    out[2, 0] = A31/det; out[2, 1] = A32/det; out[2, 2] = A33/det
    return out


@njit(cache=True)
def _ekf_forward(Z, W, x0, P0, Q, r, d):
    """Extended Kalman forward pass.  H = [1,0,0], scalar R = r.
    W[t] = 1/normaliser for step t (the previous price), so the drag term is
    d * v|v| * W[t].  Also returns the per-step Jacobians (the smoother and
    lag-one recursions need them)."""
    T = Z.shape[0]
    X_f = np.empty((T, 3, 1)); P_f = np.empty((T, 3, 3))
    X_p = np.empty((T, 3, 1)); P_p = np.empty((T, 3, 3))
    Fj = np.empty((T, 3, 3))
    x, P = x0.copy(), P0.copy()
    K = np.zeros((3, 1))
    loglik = 0.0
    for t in range(T):
        v = x[1, 0]
        x_pred = _F3 @ x
        x_pred[2, 0] -= d * v * abs(v) * W[t]          # quadratic drag on accel
        F = _F3.copy()
        F[2, 1] -= d * 2.0 * abs(v) * W[t]             # d(drag)/dv
        P_pred = F @ P @ F.T + Q
        S = P_pred[0, 0] + r                           # H P_pred H^T + R
        innov = Z[t] - x_pred[0, 0]
        K = P_pred[:, :1] / S                          # P_pred H^T S^-1
        x = x_pred + K * innov
        P = P_pred - K * P_pred[0, :]                  # (I - K H) P_pred
        P = 0.5 * (P + P.T)
        X_p[t] = x_pred; P_p[t] = P_pred
        X_f[t] = x; P_f[t] = P
        Fj[t] = F
        loglik += -0.5 * (np.log(S) + innov*innov/S + _LOG2PI)
    return X_f, P_f, X_p, P_p, Fj, K, loglik


@njit(cache=True)
def _ekf_smoother(X_f, P_f, X_p, P_p, Fj):
    """Extended RTS backward pass -- the linear recursion with the per-step
    Jacobian standing in for F (Fj[t+1] is the transition INTO t+1)."""
    T = X_f.shape[0]
    X_s = X_f.copy(); P_s = P_f.copy()
    J = np.zeros((T, 3, 3))
    for t in range(T - 2, -1, -1):
        J[t] = P_f[t] @ Fj[t + 1].T @ _inv3(P_p[t + 1])
        X_s[t] = X_f[t] + J[t] @ (X_s[t + 1] - X_p[t + 1])
        Pt = P_f[t] + J[t] @ (P_s[t + 1] - P_p[t + 1]) @ J[t].T
        P_s[t] = 0.5 * (Pt + Pt.T)
    return X_s, P_s, J


@njit(cache=True)
def _ekf_lag(P_f, J, K_last, Fj, P0, P_p0):
    """Lag-one covariance smoother, Jacobian version of the exact linear
    recursion (Fj[t+1] = transition t -> t+1)."""
    T = P_f.shape[0]
    P_lag = np.zeros((T, 3, 3))
    J_prior = P0 @ Fj[0].T @ _inv3(P_p0)
    IKH = np.eye(3)
    IKH[:, 0] -= K_last[:, 0]                          # I - K H, H = [1,0,0]
    if T >= 2:
        P_lag[T - 1] = IKH @ Fj[T - 1] @ P_f[T - 2]
        for t in range(T - 2, 0, -1):
            P_lag[t] = P_f[t] @ J[t - 1].T \
                + J[t] @ (P_lag[t + 1] - Fj[t + 1] @ P_f[t]) @ J[t - 1].T
        P_lag[0] = P_f[0] @ J_prior.T \
            + J[0] @ (P_lag[1] - Fj[1] @ P_f[0]) @ J_prior.T
    else:
        P_lag[0] = IKH @ Fj[0] @ P0
    return P_lag, J_prior


@njit(cache=True)
def _ekf_mstats(Z, W, X_s, P_s, P_lag, x0_s, P0_s):
    """M-step sufficient statistics.  The transition residual, linearised
    around the smoothed x_s[t-1], is affine in the drag coefficient:

        x_t - f_d(x_{t-1})  ~  a_t + d*g_t   with   Ftil(d) = F3 - d*G_t

    so its expected outer-product sum is a QUADRATIC matrix polynomial
    M(d) = M0 + d*M1 + d^2*M2.  Everything the (Q, d) coordinate updates and
    the analytic d-gradient need is in (M0, M1, M2); r_sum feeds R."""
    T = Z.shape[0]
    M0 = np.zeros((3, 3)); M1 = np.zeros((3, 3)); M2 = np.zeros((3, 3))
    r_sum = 0.0
    for t in range(T):
        if t == 0:
            xp, Pp = x0_s, P0_s
        else:
            xp, Pp = X_s[t - 1], P_s[t - 1]
        v = xp[1, 0]
        g = np.zeros((3, 1)); g[2, 0] = v * abs(v) * W[t]      # +d*g in residual
        G = np.zeros((3, 3)); G[2, 1] = 2.0 * abs(v) * W[t]    # Ftil = F3 - d*G
        a = X_s[t] - _F3 @ xp
        Plag = P_lag[t]
        # E[(x_t - f)(x_t - f)^T] = (a + d g)(a + d g)^T + P_s[t]
        #   - Ftil Plag^T - Plag Ftil^T + Ftil Pp Ftil^T
        M0 += a @ a.T + P_s[t] \
            - _F3 @ Plag.T - Plag @ _F3.T + _F3 @ Pp @ _F3.T
        M1 += a @ g.T + g @ a.T \
            + G @ Plag.T + Plag @ G.T - G @ Pp @ _F3.T - _F3 @ Pp @ G.T
        M2 += g @ g.T + G @ Pp @ G.T
        innov = Z[t] - X_s[t][0, 0]
        r_sum += innov * innov + P_s[t][0, 0]
    return M0, M1, M2, r_sum


def _extended_em_fit(prices, x0, P0, Q0, r0, d0, n_iter=5, r_floor=1e-12,
                     d_min=0.0, d_max=None, coord_iters=3):
    """EKS-EM: learn Q (3x3), R (scalar) and the drag coefficient d.

    Per outer iteration: one extended E-step, then coordinate updates on the
    SAME expected complete-data log-likelihood:
      d | Q :  J(d) = tr(Q^-1 (M0 + d M1 + d^2 M2)) is quadratic, so the
               gradient equation  dJ/dd = tr(Q^-1 M1) + 2d tr(Q^-1 M2) = 0
               is solved exactly in one Newton step (see module docstring --
               gradient descent would converge to exactly this point).
      Q | d :  Q = M(d)/T, then eigenvalue-floored.
      R      :  closed form, as in the linear EM.
    Returns (Q, r, d, x_last, P_last, loglik_hist).
    """
    Z = np.ascontiguousarray(prices, dtype=float)
    T = len(Z)
    # W[t] = 1/(price used to normalise the drag at step t) = previous close
    W = np.empty(T)
    W[0] = 1.0 / Z[0]
    W[1:] = 1.0 / Z[:-1]

    x0 = np.ascontiguousarray(x0, dtype=float)
    P0 = np.ascontiguousarray(P0, dtype=float)
    Q = np.ascontiguousarray(Q0, dtype=float)
    r = float(r0)
    d = float(d0)
    loglik_hist = []

    for _ in range(n_iter):
        X_f, P_f, X_p, P_p, Fj, K_last, ll = _ekf_forward(Z, W, x0, P0, Q, r, d)
        if not np.isfinite(ll):
            # diverged (degenerate window / unstable linearisation): bail out
            # with the last finite parameters rather than crashing the refit.
            break
        loglik_hist.append(ll)
        X_s, P_s, J = _ekf_smoother(X_f, P_f, X_p, P_p, Fj)
        P_lag, J_prior = _ekf_lag(P_f, J, K_last, Fj, P0, P_p[0])

        # prior's smoothed posterior (same boundary treatment as linear EM)
        x0_s = x0 + J_prior @ (X_s[0] - X_p[0])
        P0_s = P0 + J_prior @ (P_s[0] - P_p[0]) @ J_prior.T
        P0_s = 0.5 * (P0_s + P0_s.T)

        M0, M1, M2, r_sum = _ekf_mstats(Z, W, X_s, P_s, P_lag, x0_s, P0_s)

        # coordinate ascent on (d, Q); M0/M1/M2 are fixed within this E-step
        for _c in range(coord_iters):
            Qi = np.linalg.inv(Q)
            alpha = float(np.trace(Qi @ M2))          # J(d) = alpha d^2 + beta d + const
            beta = float(np.trace(Qi @ M1))
            if alpha > 0.0:
                d = -beta / (2.0 * alpha)             # exact gradient root
            d = min(max(d, d_min), d_max) if d_max is not None else max(d, d_min)
            Qn = (M0 + d * M1 + d * d * M2) / T
            Qn = 0.5 * (Qn + Qn.T)
            w, v = np.linalg.eigh(Qn)
            Q = np.ascontiguousarray((v * np.clip(w, r_floor, None)) @ v.T)

        r = max(float(r_sum / T), r_floor)

    X_f, P_f, _, _, _, _, ll = _ekf_forward(Z, W, x0, P0, Q, r, d)
    loglik_hist.append(ll)
    return Q, r, d, X_f[-1].copy(), P_f[-1].copy(), loglik_hist


@njit(cache=True)
def _ekf_step_one(x, P, z, z_prev, Q, r, d):
    """One live EKF predict+update (same math as one _ekf_forward step).
    Returns (x_new, P_new, x_pred_level, innov, S)."""
    w = 1.0 / z_prev
    v = x[1, 0]
    x_pred = _F3 @ x
    x_pred[2, 0] -= d * v * abs(v) * w
    F = _F3.copy()
    F[2, 1] -= d * 2.0 * abs(v) * w
    P_pred = F @ P @ F.T + Q
    S = P_pred[0, 0] + r
    innov = z - x_pred[0, 0]
    K = P_pred[:, :1] / S
    x_new = x_pred + K * innov
    P_new = P_pred - K * P_pred[0, :]
    P_new = 0.5 * (P_new + P_new.T)
    return x_new, P_new, x_pred[0, 0], innov, S


class EMNonLinScaled(PortfolioStrategy):
    """Drag-EKF band strategy (Q, R and drag learned by extended EM) with an
    optional SCALE-IN / averaging overlay: open a small first tranche on the
    band-break signal, then add progressively larger tranches as price keeps
    moving against the entry. See on_bar for the position semantics."""

    params = (
        ('k', 2.0),             # threshold in std devs of the innovation
        ('warmup', 180),        # bars used for the initial polyfit seed
        ('q_level', 0.1e-3),    # INITIAL Q seeds (relative to est. R) --
        ('q_vel', 0.1e-6),      #   EM re-learns Q, R, d after warmup
        ('q_acc', 0.1e-9),
        ('d0', 0.0),            # initial drag coefficient
        ('d_max', 0.5),         # clamp for the learned drag (>=0 enforced).
                                 # Drag linearizes to complex eigenvalues
                                 # 1±i·sqrt(2d|v|/z): any d>0 makes the pred
                                 # RING around price, amplitude ~sqrt(d).
                                 # 1h sweep: 0.5 -> Sharpe 2.34 vs 1.69 at 5;
                                 # d=0 kills the ringing AND the edge (+6k).
        ('reversion', True),    # True: fade the deviation (default — the
                                 # prediction LAGS price, so band breaks mark
                                 # overextension); False: follow it
        ('em_window', 720),     # bars of rolling history the EM refit uses
        ('em_interval', 1440),  # refit every this many bars
        ('em_iters', 2),        # outer EM iterations per refit. Sweep: 2 keeps
                                 # the worst prediction bounded (~18x) vs 1e15+
                                 # at 3 -- the 3rd iter overfits drag unstable.
        ('pred_max_dev', 0.5),  # reject a one-step prediction that deviates
                                 # more than this fraction from the last price
                                 # (the finite -20M spike the S>0 check misses).
                                 # 0 disables. Triggers a self-heal re-warm.
        ('dead', 0),            # bars AFTER warmup to keep filtering but not
                                 # trade -- gives the EKF/drag time to settle
                                 # before signals are trusted. 0 = disabled.
        ('max_hold', 0),        # time stop: flatten any position still open
                                 # after this many bars. A reversion that hasn't
                                 # reverted by then is a failed trade (replay on
                                 # 1h: 336 bars/14d -> +42%% pnl, kills the
                                 # multi-week zombie trades). 0 = disabled.
        ('long_only', False),   # suppress short ENTRIES (fade-the-pump shorts
                                 # earned ~0 over 918 trades at 2x the risk);
                                 # a short signal still CLOSES an open long.
        # ---- scale-in / averaging (opt-in) ------------------------------
        # OFF by default => one fixed tranche per entry, i.e. behaves exactly
        # like the non-scaled strategy. When ON, the band signal only opens the
        # FIRST tranche; further tranches are added purely on price crossing
        # the drop levels below (long) / rise levels (short) — one averaged
        # position, not separate trades (see on_bar docstring).
        ('scale_in', False),
        ('scale_levels', 4),      # max ADD-ON tranches beyond the first entry
        ('scale_step_pct', 0.03), # add the next tranche each time price is this
                                  # fraction further against the entry, measured
                                  # from the INITIAL entry price (long: -3%, -6%,
                                  # -9%…; short: +3%, +6%…)
        ('scale_size_mult', 1.5), # each add-on's notional = previous * this.
                                  # >1 = "buy more and more" (aggressive average-
                                  # down, martingale-ish); 1.0 = equal tranches
                                  # (flat DCA); <1 = tapering adds
        ('scale_max_usd', 0.0),   # TOTAL notional budget for one scaled position.
                                  # >0 => BUDGET MODE: the whole ramp sums to
                                  # exactly this, and the first tranche is derived
                                  # from it (trade_usd is then ignored for sizing).
                                  # 0 => first tranche = trade_usd and the total is
                                  # whatever the ramp adds up to.
    )

    def setup(self):
        self.kf = {}
        self._chart = {}
        for d in self.datas:
            self.kf[d] = {'ready': False, 'warm': [],
                          'X': None, 'P': None, 'Xf': None,
                          'Q': None, 'R': None, 'd': float(self.params.d0),
                          'z_prev': None,
                          'pbuf': deque(maxlen=self.params.em_window),
                          'bars_since_em': 0, 'bars_since_ready': 0,
                          'bars_since_reseed': 0, 'bars_in_pos': 0,
                          'pos_sign': 0,
                          # scale-in bookkeeping (per averaged position)
                          'sc_dir': 0,       # 0 flat, +1 scaling long, -1 short
                          'sc_anchor': None,  # entry price the levels measure from
                          'sc_level': 0,     # add-on tranches fired so far
                          'sc_units': 0.0,   # cumulative TARGET magnitude (units)
                          'sc_usd': 0.0,     # cumulative notional deployed
                          'sc_base_usd': 0.0}  # first-tranche notional (fixed at open)
            self._chart[d._name] = []

    def _scale_reset(self, st):
        st['sc_dir'] = 0
        st['sc_anchor'] = None
        st['sc_level'] = 0
        st['sc_units'] = 0.0
        st['sc_usd'] = 0.0
        st['sc_base_usd'] = 0.0

    def _base_tranche_usd(self):
        """Notional of the FIRST tranche (the rest are mult**j of it).

        Two ways to control sizing:
        • Budget mode (scale_max_usd > 0): scale_max_usd is the TOTAL you're
          willing to deploy across the whole ramp, so the first tranche is
          backed out of the geometric series 1 + m + m^2 + … + m^levels — the
          full ramp then sums to exactly scale_max_usd.
        • Forward mode (scale_max_usd == 0): the first tranche IS trade_usd and
          the total is whatever that series adds up to.
        Either way we NEVER set the entry price — that's just the market price
        on the signal bar; only the size is ours to choose."""
        cap = self.params.scale_max_usd
        if cap and cap > 0:
            m = self.params.scale_size_mult
            n = self.params.scale_levels
            factor = (n + 1) if abs(m - 1.0) < 1e-9 else (m ** (n + 1) - 1.0) / (m - 1.0)
            return cap / factor
        return self.params.trade_usd

    def _scale_open(self, d, st, price, direction):
        """First tranche of a fresh scaled position. Returns the base-class
        (signal, size) with size = positive unit magnitude."""
        base_usd = self._base_tranche_usd()
        units = base_usd / price
        st['sc_dir'] = direction
        st['sc_anchor'] = price
        st['sc_level'] = 0
        st['sc_units'] = units
        st['sc_usd'] = base_usd
        st['sc_base_usd'] = base_usd
        self.log('%s scale-in OPEN %s: tranche 0 $%.0f @ %.6f (ramp budget $%s)'
                 % (d._name, 'LONG' if direction > 0 else 'SHORT', base_usd, price,
                    ('%.0f' % self.params.scale_max_usd) if self.params.scale_max_usd
                    else 'derived'))
        return (direction, units)

    def _scale_check_adds(self, d, st, price):
        """Add every tranche whose price level has been crossed THIS bar (a gap
        can cross several at once). Grows the cumulative target; the base class
        buys only the increment via order_target_size. Returns the new target
        magnitude (units) if anything was added, else None (hold)."""
        added = False
        step, mult = self.params.scale_step_pct, self.params.scale_size_mult
        cap = self.params.scale_max_usd
        while st['sc_level'] < self.params.scale_levels:
            nxt = st['sc_level'] + 1
            if st['sc_dir'] > 0:                       # long: add as price DROPS
                level_price = st['sc_anchor'] * (1.0 - nxt * step)
                crossed = price <= level_price
            else:                                       # short: add as price RISES
                level_price = st['sc_anchor'] * (1.0 + nxt * step)
                crossed = price >= level_price
            if not crossed:
                break
            tranche_usd = st['sc_base_usd'] * (mult ** nxt)
            if cap and st['sc_usd'] + tranche_usd > cap:
                tranche_usd = cap - st['sc_usd']       # clamp to the remaining cap
                if tranche_usd <= 0:
                    break
            st['sc_units'] += tranche_usd / price
            st['sc_usd'] += tranche_usd
            st['sc_level'] = nxt
            added = True
            self.log('%s scale-in ADD level %d $%.0f @ %.6f (total $%.0f)'
                     % (d._name, nxt, tranche_usd, price, st['sc_usd']))
        return st['sc_units'] if added else None

    def _init_filter(self, st):
        buf = np.array(st['warm'], dtype=float)
        idx = np.arange(len(buf))
        # quadratic fit -> position, velocity AND acceleration seeds
        c2, c1, c0 = np.polyfit(idx, buf, 2)
        resid = buf - (c2*idx**2 + c1*idx + c0)
        R = float(np.var(resid))
        R = max(R, (buf[-1] * 1e-6) ** 2, 1e-12)
        st['R'] = R
        st['Q'] = np.diag([R * self.params.q_level,
                           R * self.params.q_vel,
                           R * self.params.q_acc])
        n = len(buf) - 1
        X0 = np.array([[buf[-1]], [2.0*c2*n + c1], [2.0*c2]])
        P0 = np.diag([R, R, R]).astype(float)
        st['X'], st['P'] = X0, P0
        st['z_prev'] = float(buf[-1])
        st['Xf'] = None                     # no forecast until the first step
        st['pbuf'].extend(buf.tolist())
        st['ready'] = True

    def _refit_em(self, d, st):
        prices = list(st['pbuf'])
        x0 = np.array([[prices[0]], [0.0], [0.0]])
        P0 = st['P']                        # current uncertainty as the prior
        Q, r, drag, x_last, P_last, ll = _extended_em_fit(
            prices, x0, P0, st['Q'], st['R'], st['d'],
            n_iter=self.params.em_iters, d_max=self.params.d_max)
        # Never hand a sick fit to the live filter: the drag model is
        # generatively unstable, so a bad window can produce non-finite or
        # indefinite results. Keep the old parameters in that case, and
        # PSD-floor the handed-off covariance either way.
        healthy = (np.isfinite(Q).all() and np.isfinite(P_last).all()
                   and np.isfinite(x_last).all()
                   and np.isfinite(r) and np.isfinite(drag)
                   and (not ll or np.isfinite(ll[-1])))
        
        recent_range = max(prices) - min(prices)
        max_plausible_v = recent_range  # velocity shouldn't exceed the window's whole price range per bar, generously
        plausible = (abs(x_last[1, 0]) < max_plausible_v * 10       # generous slack, tune this
                    and abs(x_last[2, 0]) < max_plausible_v * 10
                    and abs(drag) < self.params.d_max)
        


        healthy = healthy and plausible

        if not healthy:
            self.log('%s EEM refit DISCARDED (non-finite result) — keeping '
                     'previous Q/R/d' % d._name, doprint=True)
            return
        w, v = np.linalg.eigh(0.5 * (P_last + P_last.T))
        P_last = (v * np.clip(w, 1e-12, None)) @ v.T
        self.log('%s EEM refit: Q=diag(%.3g,%.3g,%.3g) R=%.3g d=%.4g  '
                 'loglik %.1f -> %.1f'
                 % (d._name, Q[0, 0], Q[1, 1], Q[2, 2], r, drag,
                    ll[0], ll[-1]))
        st['Q'], st['R'], st['d'] = Q, r, drag
        st['X'], st['P'] = x_last, P_last
        st['z_prev'] = float(prices[-1])

    def _reseed_state(self, st):
        """State-only self-heal: re-init X/P from the recent price buffer while
        KEEPING the learned Q/R/d (a divergence corrupts the state, not the
        parameters). Returns False when there isn't enough recent history —
        the caller then falls back to a full re-warm."""
        buf = np.array(list(st['pbuf'])[-self.params.warmup:], dtype=float)
        if len(buf) < self.params.warmup:
            return False
        idx = np.arange(len(buf))
        c2, c1, _c0 = np.polyfit(idx, buf, 2)
        n = len(buf) - 1
        st['X'] = np.array([[buf[-1]], [2.0 * c2 * n + c1], [2.0 * c2]])
        R = float(st['R'])
        st['P'] = np.diag([R, R, R]).astype(float)
        st['z_prev'] = float(buf[-1])
        return True

    def _step(self, st, price):
        x, P, y_pred, innov, S = _ekf_step_one(
            st['X'], st['P'], price, st['z_prev'],
            st['Q'], st['R'], st['d'])
        # Health check: S <= 0 or non-finite means the covariance lost
        # positive-definiteness (numerical blow-up of the unstable drag
        # dynamics). Signal the caller to self-heal instead of trading on a
        # complex-valued band.
        if not (np.isfinite(S) and S > 0.0
                and np.isfinite(innov) and np.isfinite(x).all()):
            return None
        # Hard plausibility guard: the drag dynamics can stay finite/PSD yet
        # emit a wildly-off one-step forecast (the -20M spike). A prediction
        # more than pred_max_dev away from the last price one bar ahead is
        # never real -- treat it like a divergence and self-heal.
        pmd = self.params.pred_max_dev
        if pmd and price and abs(y_pred / price - 1.0) > pmd:
            return None
        st['X'], st['P'] = x, P
        st['z_prev'] = float(price)
        return float(innov), float(S), float(y_pred)

    def on_bar(self, d, price):
        """Signal + (optional) scale-in.

        POSITIONS — how scaling behaves: each feed has ONE aggregate
        backtrader position. Adding tranches AVERAGES into that same position
        (its entry price becomes the volume-weighted average and its size
        grows); it does NOT open a second position. So the whole ramp — first
        tranche plus every add — is a SINGLE trade that opened when the first
        unit was bought. It closes (one row in the trade log, with the averaged
        entry, the total size and the P&L over the whole ramp) only when the
        position returns fully to 0. A NEW position starts only after that full
        exit: a later entry signal begins a fresh ramp with a new anchor.

        Adds are driven purely by price crossing the drop/rise levels (grid
        style), not by the band signal repeating — the signal only opens the
        first tranche and (opposite signal) closes the whole thing.
        """
        st = self.kf[d]

        # ---- time stop: count bars in the CURRENT trade regardless of filter
        # state (a position can sit open through a re-warm). The clock resets
        # when the position flips sign — a reversal is a new trade, otherwise
        # chained flips would accumulate and flatten a young position (bug
        # found 2026-07-16: a 72-bar short got flattened because exposure had
        # been continuous for 336 bars across many flips). Once overdue, every
        # early-return path below flattens instead of holding.
        overdue = False
        pos = self.getposition(d).size
        sign = 1 if pos > 0 else (-1 if pos < 0 else 0)
        if sign != st.get('pos_sign', 0):
            st['bars_in_pos'] = 0            # flat->open, open->flat, or flip
        st['pos_sign'] = sign
        # Whenever the position is actually flat, drop any scale-in bookkeeping
        # so the NEXT entry starts a fresh averaged position. This catches every
        # exit path — reversal, max_hold flatten, and the base class's trailing
        # stop (which flattens without telling on_bar) — on the following bar.
        if sign == 0 and st['sc_dir'] != 0:
            self._scale_reset(st)
        if sign != 0:
            st['bars_in_pos'] += 1
            if self.params.max_hold and st['bars_in_pos'] >= self.params.max_hold:
                overdue = True
                if st['bars_in_pos'] == self.params.max_hold:
                    self.log('%s max_hold %d bars reached -> flatten'
                             % (d._name, self.params.max_hold))

        if not st['ready']:
            st['warm'].append(price)
            if len(st['warm']) >= self.params.warmup:
                self._init_filter(st)
            return (0, 0) if overdue else 0

        st['pbuf'].append(price)
        st['bars_since_em'] += 1
        st['bars_since_reseed'] += 1
        if (st['bars_since_em'] >= self.params.em_interval
                and len(st['pbuf']) >= self.params.em_window):
            self._refit_em(d, st)
            st['bars_since_em'] = 0

        step = self._step(st, price)
        if step is None:
            # Filter state corrupted (indefinite covariance / implausible
            # prediction). The learned Q/R/d are usually still fine — it's the
            # state X/P that broke — so first try a state-only re-seed from the
            # recent price buffer and resume next bar (no 180-bar blackout).
            # If we re-diverge within `warmup` bars of a re-seed the PARAMETERS
            # themselves are sick: fall back to the full re-warm, which resets
            # the drag to d0 and lets EM relearn everything.
            if (st['bars_since_reseed'] >= self.params.warmup
                    and self._reseed_state(st)):
                st['bars_since_reseed'] = 0
                self.log('%s filter diverged — state re-seeded (kept Q/R/d)'
                         % d._name, doprint=True)
            else:
                self.log('%s filter diverged — re-warming from scratch'
                         % d._name, doprint=True)
                st.update(ready=False, warm=[], d=float(self.params.d0),
                          bars_since_em=0, bars_since_ready=0,
                          bars_since_reseed=0)
            return (0, 0) if overdue else 0
        innov, S, y_pred = step
        band = self.params.k * (S ** 0.5)
        self._chart[d._name].append((self.bar_epoch(d), round(y_pred, 8),
                                     round(y_pred + band, 8),
                                     round(y_pred - band, 8)))
        self._diag_record(d, price, y_pred, innov, S, band, st)

        # "dead" period: keep filtering/EM-fitting (already done above) but
        # suppress trade signals for the first `dead` bars after warmup ends,
        # so the EKF/drag estimate has time to settle before it's trusted.
        st['bars_since_ready'] += 1
        if st['bars_since_ready'] <= self.params.dead:
            return (0, 0) if overdue else 0

        if overdue:
            return (0, 0)

        if band <= 0:
            return 0

        # innov = price - prediction. FOLLOW (reversion=False): trade in the
        # direction of the surprise (price broke ABOVE the band -> LONG).
        # REVERSION (reversion=True): fade it (price above prediction -> SHORT,
        # betting it comes back). NOTE: before 2026-07-16 these were swapped —
        # reversion=False actually faded — so old results used the OTHER mode.
        long_sig = innov > band
        short_sig = innov < -band

        if self.params.reversion:
            long_sig, short_sig = short_sig, long_sig

        pos_size = self.getposition(d).size

        # ---- scale-in mode: signal opens tranche 1; adds are price-driven ----
        if self.params.scale_in:
            if st['sc_dir'] != 0 and pos_size != 0:
                # in a ramp: an OPPOSITE band break FLIPS the position — one
                # order closes the whole averaged ramp and opens the OTHER
                # side's first tranche (order_target_size to a small opposite
                # target does both in a single fill), which then re-ramps. No
                # cash gap. (If shorts are suppressed, a long can only flatten.)
                opposite = short_sig if st['sc_dir'] > 0 else long_sig
                if opposite:
                    new_dir = -st['sc_dir']
                    if new_dir < 0 and self.params.long_only:
                        self.log('%s opposite signal -> flatten scaled long '
                                 '(long_only, no short flip)' % d._name)
                        self._scale_reset(st)
                        return (0, 0)
                    self.log('%s opposite signal -> FLIP scaled %s to %s '
                             '(reopen at tranche 0)'
                             % (d._name, 'long' if st['sc_dir'] > 0 else 'short',
                                'short' if new_dir < 0 else 'long'))
                    return self._scale_open(d, st, price, new_dir)
                # otherwise add any tranches whose level price was crossed
                target = self._scale_check_adds(d, st, price)
                if target is not None:
                    return (st['sc_dir'], target)
                return 0                         # hold the current target
            # flat: a band break opens the first tranche of a new ramp
            if long_sig:
                return self._scale_open(d, st, price, +1)
            if short_sig and not self.params.long_only:
                return self._scale_open(d, st, price, -1)
            return 0

        # ---- non-scale mode (unchanged single-tranche behavior) ----
        if long_sig:
            if pos_size > 0:
                return 0
            self.log('%s innov ABOVE +band (innov %.6f, band %.6f) -> LONG'
                     % (d._name, innov, band))
            return 1
        if short_sig:
            if self.params.long_only:
                # No short ENTRIES — but the opposite signal still closes an
                # open long (otherwise longs would only exit via max_hold).
                if pos_size > 0:
                    self.log('%s short signal -> flatten long (long_only)'
                             % d._name)
                    return (0, 0)
                return 0
            if pos_size < 0:
                return 0
            self.log('%s innov BELOW -band (innov %.6f, band %.6f) -> SHORT'
                     % (d._name, innov, band))
            return -1

        return 0

    def build_chart_lines(self):
        k = self.params.k
        out = {}
        for d in self.datas:
            rows = self._chart[d._name]
            out[d._name] = [
                {'name': 'EEM-Kalman', 'color': '#58a6ff',
                 'points': [{'time': t, 'value': e} for t, e, u, l in rows]},
                {'name': '+%g sigma' % k, 'color': '#8b949e',
                 'points': [{'time': t, 'value': u} for t, e, u, l in rows]},
                {'name': '-%g sigma' % k, 'color': '#8b949e',
                 'points': [{'time': t, 'value': l} for t, e, u, l in rows]},
            ]
        return out

    def stop(self):
        self.log('(k=%g warmup=%d em_window=%d) Ending Value %.2f'
                 % (self.params.k, self.params.warmup, self.params.em_window,
                    self.broker.getvalue()), doprint=True)
