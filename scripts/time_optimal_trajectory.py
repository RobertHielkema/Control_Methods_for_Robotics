"""
Time-optimal trajectory generator for a single DOF, based on:
Ramos, Gajamohan, Huebel, D'Andrea (2013), "Time-Optimal Online
Trajectory Generator for Robotic Manipulators".

Give it (pi, vi) -> (pf, vf) plus limits (aM, vM) and it spits out the
switching times for a bang-bang acceleration profile, plus a function
to evaluate p(t), v(t), a(t) anywhere along it.
"""

import math
import numpy as np


def compute_profile(pi, pf, vi, vf, aM, vM, eps=1e-9):
    """Time-optimal profile from (pi, vi) to (pf, vf)."""
    dp = pf - pi
    dv = vf - vi

    # reference: how far you'd travel just ramping vi -> vf at aM
    if abs(dv) < eps:
        t_crit, dp_crit = 0.0, 0.0
    else:
        sv = 1.0 if dv > 0 else -1.0
        t_crit = sv * dv / aM
        dp_crit = sv * (vf**2 - vi**2) / (2 * aM)

    # exactly matches that reference -> one segment, done
    if abs(dp - dp_crit) < eps:
        return {
            "shape": "critical",
            "pieces": [("accel", 0.0, t_crit, (1.0 if dv >= 0 else -1.0) * aM, vi, pi)],
            "tf": t_crit,
        }

    # otherwise work out which way and how fast the peak is
    s = 1.0 if (dp - dp_crit) > 0 else -1.0
    vp_sq = 0.5 * (vf**2 + vi**2) + s * dp * aM
    if vp_sq < 0:
        raise ValueError("target isn't reachable with these aM/vM limits")
    vp = math.sqrt(vp_sq)

    if vp <= vM:
        # never hits vM, just accel then decel
        T1 = (s * vp - vi) / (s * aM)
        tf = T1 + (vf - s * vp) / (-s * aM)
        p_T1 = 0.5 * s * aM * T1**2 + vi * T1 + pi
        pieces = [
            ("accel", 0.0, T1, s * aM, vi, pi),
            ("decel", T1, tf, -s * aM, s * vp, p_T1),
        ]
        return {"shape": "triangular", "pieces": pieces, "tf": tf}

    else:
        # clips at vM, so accel / coast / decel
        T1 = (s * vM - vi) / (s * aM)
        T2 = (1 / vM) * ((vf**2 + vi**2 - 2 * s * vM * vi) / (2 * aM) + s * dp)
        tf = T2 + (vf - s * vM) / (-s * aM)

        p_T1 = 0.5 * s * aM * T1**2 + vi * T1 + pi
        p_T2 = s * vM * (T2 - T1) + p_T1

        pieces = [
            ("accel", 0.0, T1, s * aM, vi, pi),
            ("coast", T1, T2, 0.0, s * vM, p_T1),
            ("decel", T2, tf, -s * aM, s * vM, p_T2),
        ]
        return {"shape": "trapezoidal", "pieces": pieces, "tf": tf}


def evaluate(profile, t):
    """p, v, a at time t (holds the final state past tf)."""
    pieces = profile["pieces"]
    for _, t_start, t_end, a, v0, p0 in pieces:
        if t_start - 1e-9 <= t <= t_end + 1e-9:
            dt = t - t_start
            return 0.5 * a * dt**2 + v0 * dt + p0, a * dt + v0, a
    _, t_start, t_end, a, v0, p0 = pieces[-1]
    dt = t_end - t_start
    return 0.5 * a * dt**2 + v0 * dt + p0, a * dt + v0, a


class MultiJointTrajectory:
    """One compute_profile() per joint, all starting together. Each
    joint takes however long it needs and just holds once it's done --
    they're not synced to a common tf here (use compute_profile_synced
    below for that)."""

    def __init__(self, q0, qf, qd0, qdf, aM, vM):
        n = len(q0)
        aM = np.full(n, aM) if np.isscalar(aM) else np.asarray(aM)
        vM = np.full(n, vM) if np.isscalar(vM) else np.asarray(vM)
        self.profiles = [
            compute_profile(q0[j], qf[j], qd0[j], qdf[j], aM[j], vM[j])
            for j in range(n)
        ]
        self.n = n
        self.tf = max(prof["tf"] for prof in self.profiles)

    def evaluate(self, t):
        q = np.zeros(self.n)
        qd = np.zeros(self.n)
        qdd = np.zeros(self.n)
        for j, prof in enumerate(self.profiles):
            t_j = min(t, prof["tf"])  # hold once this joint's done
            q[j], qd[j], qdd[j] = evaluate(prof, t_j)
        return q, qd, qdd

    def done(self, t, tol=1e-3):
        return t >= self.tf - tol


def compute_profile_synced(pi, pf, aM, vM, t_sync, eps=1e-9):
    """Same as compute_profile but stretched to land exactly at
    t_sync instead of finishing as fast as possible. Only handles
    rest-to-rest moves (vi = vf = 0), which is all we need for
    waypoint-to-waypoint segments. Keeps acceleration at aM, just
    lowers the coast speed so the timing works out (paper Eq. 30,
    vi = 0 case):

        b = aM * t_sync
        vc = 0.5 * (b - sqrt(b^2 - 4 aM |dp|))

    If t_sync is already this DOF's own fastest time, nothing to
    stretch -- just returns the normal profile.
    """
    dp = pf - pi
    if abs(dp) < eps:
        return {"shape": "stationary", "pieces": [("hold", 0.0, t_sync, 0.0, 0.0, pi)], "tf": t_sync}

    s = 1.0 if dp > 0 else -1.0
    dp_abs = abs(dp)

    base = compute_profile(pi, pf, 0.0, 0.0, aM, vM)
    if t_sync <= base["tf"] + eps:
        return base

    b = aM * t_sync
    disc = max(b * b - 4 * aM * dp_abs, 0.0)
    vc = 0.5 * (b - math.sqrt(disc))
    vc = min(vc, vM)

    T1 = vc / aM
    T2 = t_sync - T1
    p_T1 = pi + 0.5 * s * aM * T1**2
    p_T2 = p_T1 + s * vc * (T2 - T1)

    pieces = [
        ("accel", 0.0, T1, s * aM, 0.0, pi),
        ("coast", T1, T2, 0.0, s * vc, p_T1),
        ("decel", T2, t_sync, -s * aM, s * vc, p_T2),
    ]
    return {"shape": "synced-trapezoidal", "pieces": pieces, "tf": t_sync}


if __name__ == "__main__":
    # quick sanity check
    prof = compute_profile(pi=0.0, pf=2.0, vi=0.0, vf=0.0, aM=2.0, vM=1.0)
    print(prof["shape"], "tf =", prof["tf"])
    for t in [0, 0.5, 1.0, 1.5, prof["tf"]]:
        print(f"t={t:.2f}  ->", evaluate(prof, t))
