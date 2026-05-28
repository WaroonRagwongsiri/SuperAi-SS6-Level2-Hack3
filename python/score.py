import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

G = 9.81
MAX_SCORE = 999.0

DEFAULT_REFERENCE = {
    'straightness':   (0.40, 0.90), 
    'peak_a_g':       (1.50, 5.00), 
    'snap':           (0.50, 0.85), 
    'wrist_turn_deg': (5.0,  60.0), 
    'peak_speed':     (3.0,  7.0),  
}

METRIC_WEIGHTS = {
    'straightness':   1.5,
    'peak_a_g':       1.5,
    'snap':           0.8,
    'wrist_turn_deg': 1.5,
    'peak_speed':     1.0,
}

def _scale(value, lo, hi):
    if hi == lo: return MAX_SCORE / 2
    s = MAX_SCORE * (value - lo) / (hi - lo)
    return float(np.clip(s, 0, MAX_SCORE))

def _gaussian(value, ideal, sigma):
    s = MAX_SCORE * np.exp(-((value - ideal) ** 2) / (2 * sigma ** 2))
    return float(np.clip(s, 0, MAX_SCORE))

def score_punch(rep_df, reference=None, weights=None, return_trajectory=False):
    ref = {**DEFAULT_REFERENCE, **(reference or {})}
    w = {**METRIC_WEIGHTS, **(weights or {})}

    if 'phase' in rep_df.columns:
        rep_df = rep_df[rep_df['phase'] == 'punch']

    if len(rep_df) < 5:
        return dict(overall=0.0, metrics={}, scores={}, max_score=MAX_SCORE, error='too few samples')

    rep_df = rep_df.rename(columns={'gx_dps.1': 'gz_dps'})
    rep_df = rep_df.sort_values('phase_elapsed_ms').reset_index(drop=True)

    t_ms = rep_df['phase_elapsed_ms'].to_numpy(float)
    t = (t_ms - t_ms[0]) / 1000.0
    dt = np.diff(t, prepend=t[0])
    dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 0.02

    acc = rep_df[['ax_g', 'ay_g', 'az_g']].to_numpy() * G
    gyro = np.deg2rad(rep_df[['gx_dps', 'gy_dps', 'gz_dps']].to_numpy())

    stat = (t - t[0]) <= 0.08
    if stat.sum() < 3: stat[:3] = True
    gyro -= gyro[stat].mean(axis=0)
    gravity = acc[stat].mean(axis=0)

    N = len(t)
    orient = [R.identity()]
    for i in range(1, N): orient.append(orient[-1] * R.from_rotvec(gyro[i] * dt[i]))

    acc_world = np.array([orient[i].apply(acc[i]) for i in range(N)])
    lin_acc = acc_world - gravity

    vel = np.zeros_like(lin_acc)
    for i in range(1, N): vel[i] = vel[i - 1] + lin_acc[i] * dt[i]
    vel -= np.linspace(0, 1, N)[:, None] * vel[-1]

    pos = np.zeros_like(vel)
    for i in range(1, N): pos[i] = pos[i - 1] + vel[i] * dt[i]
    pos -= np.linspace(0, 1, N)[:, None] * pos[-1]

    reach = np.linalg.norm(pos, axis=1)
    ext_idx = int(np.argmax(reach))
    reach_max = max(reach[ext_idx], 1e-6)

    pos_ext = pos[:ext_idx + 1]
    chord = np.linalg.norm(pos_ext[-1] - pos_ext[0])
    path = np.linalg.norm(np.diff(pos_ext, axis=0), axis=1).sum() if len(pos_ext) > 1 else 1e-9
    straightness = float(chord / max(path, 1e-9))

    a_mag = np.linalg.norm(acc / G, axis=1)
    peak_a = float(np.abs(a_mag[:ext_idx + 1] - 1).max())

    near_peak = reach >= 0.8 * reach_max
    snap = float(1.0 - near_peak.sum() / len(reach))

    punch_axis = pos[ext_idx] / reach_max
    gyro_world = np.array([orient[i].apply(gyro[i]) for i in range(N)])
    twist_rate = gyro_world @ punch_axis
    late_start = max(int(ext_idx * 0.7), 1)
    _trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    twist_rad = _trapz(twist_rate[late_start:ext_idx + 1], t[late_start:ext_idx + 1])
    wrist_turn = float(abs(np.degrees(twist_rad)))

    peak_speed = float(np.linalg.norm(vel, axis=1).max())

    metrics = dict(straightness=straightness, peak_a_g=peak_a, snap=snap, wrist_turn_deg=wrist_turn, peak_speed=peak_speed)
    scores = {
        'straightness':   _scale(straightness, *ref['straightness']),
        'peak_a_g':       _scale(peak_a, *ref['peak_a_g']),
        'snap':           _scale(snap, *ref['snap']),
        'wrist_turn_deg': _scale(wrist_turn, *ref['wrist_turn_deg']),
        'peak_speed':     _scale(peak_speed, *ref['peak_speed']),
    }

    total_w = sum(w.values())
    overall = sum(scores[k] * w.get(k, 0.0) for k in scores) / total_w
    overall = float(np.clip(overall, 0, MAX_SCORE))

    result = dict(overall=round(overall, 1), metrics=metrics, scores=scores, max_score=MAX_SCORE)
    if return_trajectory: result['trajectory'] = dict(t=t, pos=pos, vel=vel, ext_idx=ext_idx)
    return result
