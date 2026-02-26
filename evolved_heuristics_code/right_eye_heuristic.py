import math

def calculate_etdrs_score_od(data_string):
    def _get_median(numbers):
        if not numbers: return None
        sorted_numbers = sorted(numbers)
        n = len(sorted_numbers)
        mid = n // 2
        return sorted_numbers[mid] if n % 2 == 1 else (sorted_numbers[mid - 1] + sorted_numbers[mid]) / 2

    if not data_string or not data_string.strip():
        return None

    per_size_stats = {}
    all_correct_rts =[]

    # Single O(N) pass for parsing and aggregation
    for line in data_string.strip().split('\n'):
        try:
            parts = line.split()
            size = int(parts[2])
            rt = float(parts[5])
            is_correct = int(parts[7])
            
            if size == -1 or is_correct == -1 or rt == -1:
                continue
                
            if size not in per_size_stats:
                per_size_stats[size] = {'N_s': 0, 'C_s': 0, 'L_s': 0, 'current_streak': 0}
                
            stats = per_size_stats[size]
            stats['N_s'] += 1
            
            if is_correct == 1:
                stats['C_s'] += 1
                stats['current_streak'] += 1
                all_correct_rts.append(rt)
            else:
                if stats['current_streak'] > stats['L_s']:
                    stats['L_s'] = stats['current_streak']
                stats['current_streak'] = 0
                
        except (ValueError, IndexError):
            return None

    if not per_size_stats:
        return None

    # O(1) core logic over bounded unique sizes [5, 100]
    unique_sizes = sorted(per_size_stats.keys())
    max_ea = -1.0
    eligible_sizes =[]

    for s in unique_sizes:
        stats = per_size_stats[s]
        # Finalize streak for the size
        if stats['current_streak'] > stats['L_s']:
            stats['L_s'] = stats['current_streak']
            
        Acc_s = stats['C_s'] / stats['N_s']
        EA_s = Acc_s * min(1.0, stats['L_s'] / 3.0)
        stats['EA_s'] = EA_s
        
        if EA_s >= 0.70:
            eligible_sizes.append(s)
        if EA_s > max_ea:
            max_ea = EA_s

    if not all_correct_rts:
        return None

    RT_glob = _get_median(all_correct_rts)
    tau = 0.80
    gamma = 0.30
    TM = max(0.70, min(1.30, 1 + gamma * (RT_glob - tau) / tau))

    S_eff = eligible_sizes[0] if eligible_sizes else min(s for s in unique_sizes if per_size_stats[s]['EA_s'] == max_ea)

    base_score_table = {
        5: 20, 10: 33, 15: 45, 20: 60, 25: 80, 30: 100,
        40: 130, 50: 150, 60: 170, 80: 200, 100: 250
    }
    
    if S_eff in base_score_table:
        base_score = base_score_table[S_eff]
    else:
        sorted_keys = sorted(base_score_table.keys())
        if S_eff < sorted_keys[0]:
            base_score = base_score_table[sorted_keys[0]]
        elif S_eff > sorted_keys[-1]:
            base_score = base_score_table[sorted_keys[-1]]
        else:
            for i in range(len(sorted_keys) - 1):
                if sorted_keys[i] <= S_eff < sorted_keys[i+1]:
                    s_low, s_high = sorted_keys[i], sorted_keys[i+1]
                    base_score = base_score_table[s_low] + (S_eff - s_low) * (base_score_table[s_high] - base_score_table[s_low]) / (s_high - s_low)
                    break

    beta = 2
    return int(max(20, min(250, round(base_score * TM + beta))))