import pandas as pd
import numpy as np
import warnings
from collections import defaultdict
from datetime import datetime

from utilities.common import determineWinnerLoser, updateElo, updateEloSimple, calculateStats, convertResult, shrinkElo, updateWinStreak


class StatsAPI:

    def __init__(self):
        self.player_stats = {}
        self.h2h = defaultdict(int) 
        self.alpha_short = 0.18
        self.alpha_long = 0.05
    
    def get_player_stats(self, player_id):
        if player_id not in self.player_stats:
            warnings.warn(f"No player: {player_id}")
            return None
        
        return self.player_stats[player_id]

    def get_ewma(self, player_id, metric, horizon):
        if horizon == 'short':  return self.player_stats.get(player_id, {}).get('ewma_short', {}).get(metric, 0)
        else:                   return self.player_stats.get(player_id, {}).get('ewma_long', {}).get(metric, 0)

    def get_h2h(self, p1, p2):
        return self.h2h.get((p1, p2), 0)
    
    def get_momentum(self, player_id):
        p_dic = self._get_player_dict(player_id)
        s_dict = p_dic['ewma_short']
        l_dict = p_dic['ewma_long']
        if not s_dict or not l_dict: return 0

        diffs = [s_dict[k] - l_dict[k] for k in s_dict if k in l_dict]
        if not diffs: return 0

        arr        = np.asarray(diffs, dtype=float)
        direction  = np.sign(arr).mean()         
        magnitude  = np.sqrt((arr ** 2).mean())

        return float(direction * magnitude)

    def decay_elo(self, player_id, date):
        p_dic = self._get_player_dict(player_id)
        decayedElo = shrinkElo(p_dic['elo'], p_dic['last_game_date'], date)
        if decayedElo != p_dic['elo']:
            p_dic['elo'] = decayedElo
            p_dic['elo_boost'] = 5
    
    def update_stats(self, player1, player2, match_info):
        surface = match_info['SURFACE']
        result = match_info['RESULT']
        p1_dic = self._get_player_dict(player1["ID"])
        p2_dic = self._get_player_dict(player2["ID"])

        self._record_base_stats(p1_dic, match_info, player1['tb_won'], player2['HAND'], 1)
        self._record_base_stats(p2_dic, match_info, player2['tb_won'], player1['HAND'], 0)
        self._update_h2h(player1["ID"], player2["ID"], result)
        self._update_elo(p1_dic, p2_dic, player1["ID"], player2["ID"], surface, result)
        self._record_match_stats(player1, result)
        self._record_match_stats(player2, convertResult(result))

    def _get_player_dict(self, player_id):
        if player_id not in self.player_stats:
            self.player_stats[player_id] = {
                'total_matches': 0,             'wins': 0,
                'total_tbs': 0,                 'tbs_won': 0,
                'elo': 1500,                    'elo_Carpet': 1500, 
                'elo_Clay': 1500,               'elo_Grass': 1500, 
                'elo_Hard': 1500,               'elo_boost': 0,
                'ewma_short': {},               'ewma_long': {},
                'total_vs_R_handers': 0,        'wins_vs_R_handers': 0,
                'total_vs_L_handers': 0,        'wins_vs_L_handers': 0,
                'win_streak': 0,                
                'last_game_date': pd.NaT,       
            }
        return self.player_stats[player_id]

    def _record_base_stats(self, p_dic, match_info, tbs_won, opp_hand, flag):
        if(flag == 1): result = match_info['RESULT']
        else: result = convertResult(match_info['RESULT'])

        p_dic['total_matches']                  += 1
        p_dic['wins']                           += result
        p_dic['total_tbs']                      += match_info['NUM_TIEBREAKS']
        p_dic['tbs_won']                        += tbs_won
        p_dic['win_streak']                     = updateWinStreak(p_dic['win_streak'], result)
        p_dic[f'total_vs_{opp_hand}_handers']   += 1
        p_dic[f'wins_vs_{opp_hand}_handers']    += result
        p_dic['last_game_date']                 = pd.to_datetime(match_info['MATCH_DATE'])
    
    def _record_match_stats(self, player, win_flag):
        (ace_rate, d_fault_rate, first_serve_in, first_serve_won, 
        second_serve_won, service_points_won, service_games_won, bp_saved,
        return_pts_won, return_gms_won, bp_conv) = calculateStats(player['ace'], player['df'], player['svpt'], player['1stIn'], player['1stWon'], 
                                                                        player['2ndWon'], player['SvGms'], player['bpSaved'], player['bpFaced'], player['oppSvpt'],
                                                                        player['oppSvptWon'], player['oppSvGms'], player['bpWon'], player['bpChances'])
        stats = {
            'ACE'           : ace_rate,
            'DF'            : d_fault_rate,
            '1ST_IN'        : first_serve_in,
            '1ST_WON'       : first_serve_won,
            '2ND_WON'       : second_serve_won,
            'SRV_PTS_WON'   : service_points_won,
            'SRV_GMS_WON'   : service_games_won,
            'BP_SAVED'      : bp_saved,
            'RET_PTS_WON'   : return_pts_won,
            'RET_GMS_WON'   : return_gms_won,
            'BP_CONVERSION' : bp_conv,
            'WINRATE'       : win_flag
        }

        for k, v in stats.items():
            self._update_ewma(player["ID"], k, v, self.alpha_short)
            self._update_ewma(player["ID"], k, v, self.alpha_long)

    def _update_h2h(self, p1_id, p2_id, result):
        w_id, l_id = determineWinnerLoser(p1_id, p2_id, result)
        key = (w_id, l_id)
        self.h2h[key] = self.h2h.get(key, 0) + 1

    def _update_elo(self, p1_dic, p2_dic, p1_id, p2_id, surface, result):
        p1_dic[f'elo_{surface}'], p2_dic[f'elo_{surface}'] = updateEloSimple(p1_dic[f'elo_{surface}'], p2_dic[f'elo_{surface}'], result)
        p1_dic['elo'], p2_dic['elo'] = updateElo(p1_dic, p2_dic, result)

        for p_id, p_dic in [(p1_id, p1_dic), (p2_id, p2_dic)]:
            for a in (self.alpha_short, self.alpha_long):
                self._update_ewma(p_id, 'ELO', p_dic['elo'], a)

    def _update_ewma(self, player_id, metric, new_value, alpha):
        if alpha == self.alpha_short:
            ewma_dict = self.player_stats[player_id].setdefault('ewma_short', {})
        else: 
            ewma_dict = self.player_stats[player_id].setdefault('ewma_long', {})

        if metric not in ewma_dict:
            ewma_dict[metric] = new_value
        else:
            ewma_dict[metric] = (1 - alpha) * ewma_dict[metric] + alpha * new_value
    

class FeatureGenerator:

    def __init__(self, stats_api):
        self.stats_api = stats_api
        self.SHORT = 'short'
        self.LONG = 'long'
        self.EWMA_STATS = [
        'ELO',
        'ACE',          'DF',
        '1ST_IN',       '1ST_WON',
        '2ND_WON',      'WINRATE',
        'SRV_PTS_WON',  'SRV_GMS_WON',
        'BP_SAVED',     'BP_CONVERSION',
        'RET_PTS_WON',  'RET_GMS_WON'   
    ]

    def generate_features_for_match(self, p1, p2, match):
        self._decay_elo(p1['ID'], p2['ID'], match['MATCH_DATE'])

        p1_stats = self.stats_api.get_player_stats(p1['ID'])
        p2_stats = self.stats_api.get_player_stats(p2['ID'])
        p1_id = p1['ID']
        p2_id = p2['ID']
        surface = match['SURFACE']

        base_feats = {
            'TOURNEY_NAME': match['TOURNEY_NAME'],
            'TOURNEY_LEVEL': match['TOURNEY_LEVEL'],
            'SURFACE': surface,
            'DRAW_SIZE': match['DRAW_SIZE'],
            'BEST_OF': match['BEST_OF'],
            'ROUND': match['ROUND'],
            'P1_ID': p1_id,
            'P2_ID': p2_id,
            'P1_ATP': p1['ATP_RANK'],
            'P2_ATP': p2['ATP_RANK'],
            'P1_ELO': p1_stats['elo'],
            'P2_ELO': p2_stats['elo'],
            'AGE': p1['AGE'] - p2['AGE'],
            'HEIGHT': p1['HEIGHT'] - p2['HEIGHT'],
            'ATP_RANK': -(p1['ATP_RANK'] - p2['ATP_RANK']),
            'ATP_PTS': p1['ATP_RANK_POINTS'] - p2['ATP_RANK_POINTS'],
            'TOTAL_MATCHES': p1_stats['total_matches'] - p2_stats['total_matches'],
            'H2H': self._h2h_diff(p1, p2),
            'ELO': p1_stats['elo'] - p2_stats['elo'],
            'ELO_SURFACE': p1_stats[f'elo_{surface}'] - p2_stats[f'elo_{surface}'],
            'TB_WINRATE': self._tb_diff(p1_stats, p2_stats),
            'HAND_WINRATE': self._hand_winrate_diff(p1, p2, p1_stats, p2_stats),
            'WIN_STREAK': p1_stats['win_streak'] - p2_stats['win_streak'],
            'SRV_ADV_S': self._srv_adv(p1_id, p2_id, self.SHORT) - self._srv_adv(p2_id, p1_id, self.SHORT),
            'SRV_ADV_L': self._srv_adv(p1_id, p2_id, self.LONG) - self._srv_adv(p2_id, p1_id, self.LONG),
            'CMPLT_S': self._completeness(p1_id, self.SHORT) - self._completeness(p2_id, self.SHORT),
            'CMPLT_L': self._completeness(p1_id, self.LONG) - self._completeness(p2_id, self.LONG),
            'MOMENTUM': self.stats_api.get_momentum(p1_id) - self.stats_api.get_momentum(p2_id),
        }

        ewma_feats = self._ewma_features(p1_id, p2_id)

        features_dict = {**base_feats, **ewma_feats}
        return features_dict

    def _decay_elo(self, p1_id, p2_id, match_date):
        for pid in (p1_id, p2_id):
            self.stats_api.decay_elo(pid, match_date)

    def _h2h_diff(self, p1, p2):
        h2h_diff = (self.stats_api.get_h2h(p1['ID'], p2['ID']) - self.stats_api.get_h2h(p2['ID'], p1['ID']))
        return h2h_diff

    def _tb_diff(self, p1, p2):
        if p1['total_tbs'] == 0 or p2['total_tbs'] == 0:
            return 0
        return (p1['tbs_won'] / p1['total_tbs']) - (p2['tbs_won'] / p2['total_tbs'])

    def _hand_winrate_diff(self, p1, p2, s1, s2):
        def hand_wr(stats, opponent_hand):
            key_wins = 'wins_vs_R_handers' if opponent_hand == 'R' else 'wins_vs_L_handers'
            key_total = 'total_vs_R_handers' if opponent_hand == 'R' else 'total_vs_L_handers'
            return stats[key_wins] / stats[key_total] if stats[key_total] else 0

        p1_wr = hand_wr(s1, p2['HAND'])
        p2_wr = hand_wr(s2, p1['HAND'])
        return p1_wr - p2_wr

    def _ewma_features(self, p1_id, p2_id):
        p1 = self._collect_ewma(p1_id)
        p2 = self._collect_ewma(p2_id)
        return {k: p1[k] - p2[k] for k in p1}

    def _srv_adv(self, pid1, pid2, horizon):
        return (self.stats_api.get_ewma(pid1, 'SRV_PTS_WON', horizon) - self.stats_api.get_ewma(pid2, 'RET_PTS_WON', horizon))

    def _completeness(self, pid, window):
            return (self.stats_api.get_ewma(pid, 'SRV_PTS_WON', window) * self.stats_api.get_ewma(pid, 'RET_PTS_WON', window))
    
    def _collect_ewma(self, player_id):
        ewma = {}
        for stat in self.EWMA_STATS:
            s_val = self.stats_api.get_ewma(player_id, stat, self.SHORT)
            l_val = self.stats_api.get_ewma(player_id, stat, self.LONG)
            ewma[f'{stat}_S'] = s_val                
            ewma[f'{stat}_L'] = l_val
        return ewma


def create_training_dataset(df):

    stats_api = StatsAPI()
    feat_gen = FeatureGenerator(stats_api)

    rows_for_model = []

    for idx, match_row in df.iterrows():

        player1 = {
            'ID'              : match_row['p1_id'],
            'NAME'            : match_row['p1_name'],
            'ATP_RANK_POINTS' : match_row['p1_rank_points'],
            'ATP_RANK'        : match_row['p1_rank'],
            'AGE'             : match_row['p1_age'],
            'HEIGHT'          : match_row['p1_ht'],
            'HAND'            : match_row['p1_hand']
        }

        player2 = {
            'ID'              : match_row['p2_id'],
            'NAME'            : match_row['p2_name'],
            'ATP_RANK_POINTS' : match_row['p2_rank_points'],
            'ATP_RANK'        : match_row['p2_rank'],
            'AGE'             : match_row['p2_age'],
            'HEIGHT'          : match_row['p2_ht'],
            'HAND'            : match_row['p2_hand']
        }

        match_info = {
            'TOURNEY_NAME'    : match_row['tourney_name'],
            'TOURNEY_LEVEL'   : match_row['tourney_level'],
            'ROUND'           : match_row['round'],
            'BEST_OF'         : match_row['best_of'],
            'DRAW_SIZE'       : match_row['draw_size'],
            'SURFACE'         : match_row['surface'],
            'MATCH_DATE'      : match_row['match_date']
        }
   
        feats = feat_gen.generate_features_for_match(player1, player2, match_info)

        post_match_info = {
            **match_info,
            'RESULT'           : match_row['result'],
            'TOTAL_GAMES'      : match_row['total_games'],
            'MINUTES'          : match_row['minutes'],
            'NUM_TIEBREAKS'    : match_row['num_tiebreaks'],
        }

        p1_post_match = {
            **player1,
            'tb_won'       : match_row['p1_tb_won'],
            'ace'          : match_row['p1_ace'],        'df'       : match_row['p1_df'], 
            'svpt'         : match_row['p1_svpt'],       '1stIn'    : match_row['p1_1stIn'], 
            '1stWon'       : match_row['p1_1stWon'],     '2ndWon'   : match_row['p1_2ndWon'],  
            'SvGms'        : match_row['p1_SvGms'],      'bpSaved'  : match_row['p1_bpSaved'], 
            'bpFaced'      : match_row['p1_bpFaced'],    'oppSvpt'  : match_row['p2_svpt'],
            'oppSvGms'     : match_row['p2_SvGms'],
            'oppSvptWon'   : match_row['p2_1stWon'] + match_row['p2_2ndWon'],
            'bpWon'        : match_row['p2_bpFaced'] - match_row['p2_bpSaved'],
            'bpChances'    : match_row['p2_bpFaced']
        }

        p2_post_match = {
            **player2,
            'tb_won'       : match_row['p2_tb_won'],
            'ace'          : match_row['p2_ace'],        'df'       : match_row['p2_df'], 
            'svpt'         : match_row['p2_svpt'],       '1stIn'    : match_row['p2_1stIn'], 
            '1stWon'       : match_row['p2_1stWon'],     '2ndWon'   : match_row['p2_2ndWon'],  
            'SvGms'        : match_row['p2_SvGms'],      'bpSaved'  : match_row['p2_bpSaved'], 
            'bpFaced'      : match_row['p2_bpFaced'],    'oppSvpt'  : match_row['p1_svpt'],
            'oppSvGms'     : match_row['p1_SvGms'],
            'oppSvptWon'   : match_row['p1_1stWon'] + match_row['p1_2ndWon'],
            'bpWon'        : match_row['p1_bpFaced'] - match_row['p1_bpSaved'],
            'bpChances'    : match_row['p1_bpFaced']
        }

        row_dict = feats.copy()
        row_dict['MATCH_DATE']  = match_row['match_date']
        row_dict['RESULT']      = match_row['result']
        row_dict['p1_name']     = match_row['p1_name']
        row_dict['p2_name']     = match_row['p2_name']
        
        rows_for_model.append(row_dict)

        stats_api.update_stats(p1_post_match, p2_post_match, post_match_info)

    model_df = pd.DataFrame(rows_for_model)
    return model_df