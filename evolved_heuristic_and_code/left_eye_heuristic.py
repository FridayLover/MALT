import math
import statistics

def calculate_etdrs_score_os(data_string):
    ACC_THR = 0.60
    MIN_TRIALS = 2
    A_BASE = 2.4
    B_BASE = 8
    K_SEV = 6
    SEV_CAP = 10
    RT_REF = 0.80
    RT_BONUS = 1
    RT_PENALTY = 1
    W_MAX = 1.0
    SEV_DIV = 5

    if not data_string or not data_string.strip():
        return None

    per_size_stats = {}
    
    # Single O(N) pass for parsing and aggregation
    for line in data_string.strip().split('\n'):
        try:
            parts = line.split()
            size = int(parts[2])
            response = int(parts[4])
            rt = float(parts[5])
            is_correct = int(parts[7])

            if size == -1 or rt == -1:
                continue

            if size not in per_size_stats:
                per_size_stats[size] = {'N_s': 0, 'C_s': 0, 'no_resp_s': False, 'correct_rts':[]}
            
            stats = per_size_stats[size]
            stats['N_s'] += 1
            
            if is_correct == 1:
                stats['C_s'] += 1
                stats['correct_rts'].append(rt)
                
            if response == -1:
                stats['no_resp_s'] = True
                
        except (ValueError, IndexError):
            return None

    if not per_size_stats:
        return None

    # O(1) core logic over bounded unique sizes [5, 100]
    unique_sizes = sorted(per_size_stats.keys())
    s_thr = -1
    max_sev = 0

    for s in unique_sizes:
        stats = per_size_stats[s]
        p_s = stats['C_s'] / stats['N_s']
        
        if s_thr == -1 and stats['N_s'] >= MIN_TRIALS and p_s >= ACC_THR:
            s_thr = s
            
        sev_s = (1 - p_s) * s
        if stats['no_resp_s']:
            sev_s = max(sev_s, 0.5 * s)
        if sev_s > max_sev:
            max_sev = sev_s

    if s_thr == -1:
        s_thr = unique_sizes[-1]

    S_base = A_BASE * s_thr + B_BASE
    sev = min(SEV_CAP, max_sev)
    S_sev = 20 + K_SEV * sev
    w = min(W_MAX, sev / SEV_DIV)

    delta_RT = 0
    correct_rts_thr = per_size_stats[s_thr]['correct_rts']
    if correct_rts_thr:
        rt_med_thr = statistics.median(correct_rts_thr)
        if rt_med_thr < RT_REF:
            delta_RT = -((RT_REF - rt_med_thr) / 0.1) * RT_BONUS
        else:
            delta_RT = ((rt_med_thr - RT_REF) / 0.1) * RT_PENALTY
    
    S = (1 - w) * S_base + w * S_sev + round(delta_RT)
    return int(max(20, min(80, round(S))))