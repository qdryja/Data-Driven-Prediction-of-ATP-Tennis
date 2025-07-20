import pandas as pd
import re
import numpy as np
import unicodedata, re
import math
from sklearn.metrics import accuracy_score, roc_auc_score,log_loss, brier_score_loss

FEATURES = [
    'AGE', 'HEIGHT',
    'ATP_RANK', 'ATP_PTS', 'TOTAL_MATCHES', 'H2H', 'ELO', 'ELO_SURFACE',
    'TB_WINRATE', 'HAND_WINRATE', 'WIN_STREAK', 'SRV_ADV_S', 'SRV_ADV_L',
    'CMPLT_S', 'CMPLT_L', 'MOMENTUM', 'ELO_S', 'ELO_L', 'ACE_S', 'ACE_L',
    'DF_S', 'DF_L', '1ST_IN_S', '1ST_IN_L', '1ST_WON_S', '1ST_WON_L',
    '2ND_WON_S', '2ND_WON_L', 'WINRATE_S', 'WINRATE_L', 'SRV_PTS_WON_S',
    'SRV_PTS_WON_L', 'SRV_GMS_WON_S', 'SRV_GMS_WON_L', 'BP_SAVED_S',
    'BP_SAVED_L', 'BP_CONVERSION_S', 'BP_CONVERSION_L', 'RET_PTS_WON_S',
    'RET_PTS_WON_L', 'RET_GMS_WON_S', 'RET_GMS_WON_L', 
]

def check_nans(df):
    columns_to_check = ['tourney_id', 'tourney_name', 'surface', 'draw_size', 'tourney_level',
       'tourney_date', 'match_num', 'winner_id', 'winner_seed', 'winner_entry',
       'winner_name', 'winner_hand', 'winner_ht', 'winner_ioc', 'winner_age',
       'loser_id', 'loser_seed', 'loser_entry', 'loser_name', 'loser_hand',
       'loser_ht', 'loser_ioc', 'loser_age', 'score', 'best_of', 'round',
       'minutes', 'winner_rank', 'winner_rank_points', 'loser_rank', 'loser_rank_points']

    player_stat_suffixes = ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 'SvGms', 'bpSaved', 'bpFaced']
    player_stat_cols = [f"w_{suffix}" for suffix in player_stat_suffixes] + [f"l_{suffix}" for suffix in player_stat_suffixes]

    for col in columns_to_check:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                print(f"Number of missing entries in '{col}': {missing}")

    missing_in_stats = df[player_stat_cols].isna().any(axis=1).sum()
    print(f"Number of rows with missing player statistics: {missing_in_stats}")
    print(f"Total rows with at least one missing value: {df.isna().any(axis=1).sum()}")


def filter_df_by_ids(df, ids):
    rows_with_nan_and_gs_players = df[
        df['winner_id'].isin(ids) |
        df['loser_id'].isin(ids)
    ]
    
    return rows_with_nan_and_gs_players


def get_gs_df(df, start_year=2010):
    gs_df = df[
    (df['tourney_level'] == 'G') &
    (df['tourney_date'] >= pd.to_datetime(str(start_year) + "-01-01"))
    ]
    return gs_df


def parse_score(score: str):
    if pd.isna(score):
        return {
            'total_games': None,
            'num_tiebreaks': None,
            'has_tiebreak': None,
            'winner_tb_won': None,
            'loser_tb_won': None
        }

    total_games = 0
    num_tiebreaks = 0
    winner_tb_won = 0
    loser_tb_won = 0

    for s in score.strip().split():

        if not re.fullmatch(r'\d+-\d+(?:\(\d+\))?', s):
            continue

        tb_match = re.fullmatch(r'(\d+)-(\d+)\((\d+)\)', s)
        if tb_match:
            g1, g2, _ = map(int, tb_match.groups())
            total_games += g1 + g2
            num_tiebreaks += 1
            if g1 > g2: winner_tb_won += 1
            else: loser_tb_won += 1
            
            continue

        g1, g2 = map(int, s.split('-'))
        total_games += g1 + g2

        if abs(g1 - g2) == 1 and max(g1, g2) == 7:
            num_tiebreaks += 1
            if g1 > g2: winner_tb_won += 1
            else: loser_tb_won += 1

    return {
        'total_games'     : total_games,
        'num_tiebreaks'   : num_tiebreaks,
        'has_tiebreak'    : num_tiebreaks > 0,
        'winner_tb_won'   : winner_tb_won,
        'loser_tb_won'    : loser_tb_won
    }


def determineWinnerLoser(p1_id, p2_id, result):
    if result == 1:
        return p1_id, p2_id
    return p2_id, p1_id


def convertResult(result):
    if result == 1: return 0
    else: return 1
    

def updateWinStreak(streak, result):
    if result == 1: return streak + 1
    else:  return 0
    

def dynamic_k(n_matches, K=250, offset=5, shape=0.4):
    return K / ((n_matches + offset) ** shape)


def shrinkElo(elo, last_game, date, threshold = 90):
    days_idle = (date - last_game).days
    new_elo = elo
    if days_idle > threshold:
        new_elo = 1500 + (elo - 1500) * math.exp(-0.0008 * days_idle)
    return new_elo


def updateElo(p1_dic, p2_dic, result):   
    k_p1 = dynamic_k(p1_dic['total_matches'])
    k_p2 = dynamic_k(p2_dic['total_matches'])

    if p1_dic['elo_boost'] > 0:
        k_p1 *= 1.5
        p1_dic['elo_boost'] -= 1
    if p2_dic['elo_boost'] > 0:
        k_p2 *= 1.5
        p2_dic['elo_boost'] -= 1

    if(result == 1):
        winner_elo = p1_dic['elo']
        loser_elo = p2_dic['elo']
        k_w = k_p1
        k_l = k_p2
    else:
        winner_elo = p2_dic['elo']
        loser_elo = p1_dic['elo']
        k_w = k_p2
        k_l = k_p1

    expected_win = 1 / (1 + 10**((loser_elo - winner_elo)/400))
    expected_loss = 1 - expected_win
    upd_win_elo = winner_elo + k_w * (1 - expected_win)
    upd_los_elo = loser_elo + k_l * (0 - expected_loss)

    if(result == 1):
        return upd_win_elo, upd_los_elo
    else:
        return upd_los_elo, upd_win_elo


def updateEloSimple(p1_elo, p2_elo, result):
    K = 30
    if(result == 1):
        winner_elo = p1_elo
        loser_elo = p2_elo
    else:
        winner_elo = p2_elo
        loser_elo = p1_elo

    expected_win = 1 / (1 + 10**((loser_elo - winner_elo)/400))
    expected_loss = 1 - expected_win
    upd_win_elo = winner_elo + K * (1 - expected_win)
    upd_los_elo = loser_elo + K * (0 - expected_loss)

    if(result == 1):
        return upd_win_elo, upd_los_elo
    else:
        return upd_los_elo, upd_win_elo


def calculateStats(ace, df, svpt, firstIn, firstWon, secondWon, SvGms, bpSaved, bpFaced, oppSvpt, oppSvptWon, oppSvGms, bpWon, bpChances):

    if svpt != 0: 
        ace_rate = ace / svpt
        d_fault_rate = df / svpt 
        first_serve_in = firstIn / svpt
        service_points_won = (firstWon + secondWon) / svpt
    else: 
        ace_rate = 0
        d_fault_rate = 0
        first_serve_in = 0
        service_points_won = 0
    
    if firstIn > 0: first_serve_won = firstWon / firstIn
    else: first_serve_won = 0

    if svpt > firstIn: second_serve_won = secondWon / (svpt - firstIn)
    else: second_serve_won = 0

    if bpFaced > 0: bp_saved =  bpSaved / bpFaced
    else: bp_saved = 0

    if oppSvpt > 0: return_pts_won = (oppSvpt - oppSvptWon) / oppSvpt
    else: return_pts_won = 0

    if oppSvGms > 0: return_gms_won = bpWon / oppSvGms
    else: return_gms_won = 0

    if bpChances > 0: bp_conv = bpWon / bpChances
    else: bp_conv = 0

    if SvGms > 0: service_games_won = (SvGms - (bpFaced - bp_saved)) / SvGms
    else: service_games_won = 0

    return ace_rate, d_fault_rate, first_serve_in, first_serve_won, second_serve_won, service_points_won, service_games_won, bp_saved, return_pts_won, return_gms_won, bp_conv


def merge(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    p1_col="P1_ATP", p2_col="P2_ATP", round1_col="ROUND",
    w_col="WRank", l_col="LRank", round2_col="Round",
    odds_priority=[
        ("EXW",  "EXL"),    
        ("LBW",  "LBL"),     
        ("PSW",  "PSL"),     
        ("SJW",  "SJL"),
        ("B365W","B365L"),     
        ("AvgW", "AvgL")
    ],
    agg="first",            
    keep_helpers=False,     
):
    round_2_to_std = {
        "1st Round":"R128", "2nd Round":"R64", "3rd Round":"R32",
        "4th Round":"R16", "Quarterfinals":"QF", "Semifinals":"SF",
        "The Final":"F",
    }
    round_1_to_std = {r: r for r in
                      ["R128","R64","R32","R16","QF","SF","F"]}

    avail = set(df2.columns)
    real_priority = [(w, l) for w, l in odds_priority
                     if w in avail and l in avail]

    def _to_int(s):
        return pd.to_numeric(s, errors="coerce").round().astype("Int64")

    def _two_keys(r1, r2, rnd):
        if pd.isna(rnd):
            return np.nan, np.nan
        ranks = [r for r in (r1, r2) if not pd.isna(r)]
        if not ranks:
            return np.nan, np.nan
        if len(ranks) == 1:      
            k = f"{ranks[0]}_{rnd}"
            return k, k
        hi, lo = min(ranks), max(ranks)
        return f"{hi}_{rnd}", f"{lo}_{rnd}"

    def _pick_odds(row, pairs=real_priority):
        for w, l in pairs:                
            win, lose = row.get(w), row.get(l)
            if pd.notna(win) and pd.notna(lose):
                return pd.Series({"WinOdds": win, "LoseOdds": lose})
        return pd.Series({"WinOdds": np.nan, "LoseOdds": np.nan})

    df2 = df2.copy()

    for col in (w_col, l_col):
        df2[col] = _to_int(df2[col])

    df2["round_std"] = df2[round2_col].map(round_2_to_std)

    df2[["high_key", "low_key"]] = df2.apply(
        lambda r: _two_keys(r[w_col], r[l_col], r["round_std"]),
        axis=1, result_type="expand"
    )

    df2[["WinOdds", "LoseOdds"]] = df2.apply(_pick_odds, axis=1)

    lookup = (
        pd.concat(
            [
                df2[["high_key", "WinOdds", "LoseOdds"]].rename(columns={"high_key":"key"}),
                df2[["low_key",  "WinOdds", "LoseOdds"]].rename(columns={"low_key":"key"}),
            ],
            ignore_index=True
        )
        .dropna(subset=["key"])
        .groupby("key", dropna=False)
        .agg(agg)
    )

    df1 = df1.copy()

    for col in (p1_col, p2_col):
        df1[col] = _to_int(df1[col])

    df1["round_std"] = df1[round1_col].map(round_1_to_std)

    df1[["high_key", "low_key"]] = df1.apply(
        lambda r: _two_keys(r[p1_col], r[p2_col], r["round_std"]),
        axis=1, result_type="expand"
    )

    df1["WinOdds"]  = df1["high_key"].map(lookup["WinOdds"])
    df1["LoseOdds"] = df1["high_key"].map(lookup["LoseOdds"])

    missing = df1["WinOdds"].isna() | df1["LoseOdds"].isna()
    if missing.any():
        df1.loc[missing, "WinOdds"]  = df1.loc[missing, "low_key"].map(lookup["WinOdds"])
        df1.loc[missing, "LoseOdds"] = df1.loc[missing, "low_key"].map(lookup["LoseOdds"])

    if not keep_helpers:
        df1 = df1.drop(columns=["round_std", "high_key", "low_key"])

    return df1

def compute_metrics(y_true, p1_prob):
    
    y_pred = (p1_prob >= 0.5).astype(int)
    return {
        "Acc"  : accuracy_score(y_true, y_pred),
        "AUC"  : roc_auc_score(y_true, p1_prob),
        "LogL" : log_loss(y_true, p1_prob),
        "Brier": brier_score_loss(y_true, p1_prob),
    }