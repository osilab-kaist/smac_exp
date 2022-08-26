from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.maps import lib


class SMAC_Plus_Map(lib.Map):
    directory = "SMAC_Plus_Maps"
    # download = "https://github.com/oxwhirl/smac#smac-maps"
    players = 2
    step_mul = 8
    game_steps_per_episode = 0


map_param_registry = {
    # SMAC Plus
    # registry of created maps
    # Attack maps
    #easy
    "hill_mTMt_att_both_1": {
        "n_agents": 15,
        "n_enemies": 9,
        "n_neutrals": 8,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 8,
        "map_type": "plain_neut_tank",
    },
    #superhard
    "hill_mTMt_att_both_6": {
        "n_agents": 15,
        "n_enemies": 9,
        "n_neutrals": 8,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 8,
        "map_type": "plain_neut_tank",
    }, 
    #ultrahard
    "hill_mTMt_att_both_6A": {
        "n_agents": 15,
        "n_enemies": 9,
        "n_neutrals": 16,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 8,
        "map_type": "plain_neut_tank",
    },
    #hard
    "hill_mTMt_att_left_1": { 
        "n_agents": 15,
        "n_enemies": 9,
        "n_neutrals": 8,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 8,
        "map_type": "plain_neut_tank",
        #########여기까지가 공격 시나이로에서 우리가 현재 쓰는 맵#######
    },
    "offense_supersuperultrahard": {
        "n_agents": 9,
        "n_enemies": 9,
        "n_neutrals": 16,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 8,
        "map_type": "plain_neut_tank",
    },

    "offense_superultrahard": {
        "n_agents": 9,
        "n_enemies": 9,
        "n_neutrals": 16,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 8,
        "map_type": "plain_neut_tank",
    },
    
    "hill_mTMt_att_left_2": {
        "n_agents": 15,
        "n_enemies": 9,
        "n_neutrals": 8,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 8,
        "map_type": "plain_neut_tank",
    },
    "hill_mTMt_att_left_3": {
        "n_agents": 15,
        "n_enemies": 9,
        "n_neutrals": 8,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 8,
        "map_type": "plain_neut_tank",
    },
    "hill_mTMt_att_alternate_1": {
        "n_agents": 15,
        "n_enemies": 9,
        "n_neutrals": 8,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 8,
        "map_type": "plain_neut_tank",
    },
    "hill_mTMt_att_alternate_2": {
        "n_agents": 15,
        "n_enemies": 9,
        "n_neutrals": 8,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 8,
        "map_type": "plain_neut_tank",
    },
    "hill_mTMt_att_alternate_3": {
        "n_agents": 15,
        "n_enemies": 9,
        "n_neutrals": 8,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 8,
        "map_type": "plain_neut_tank",
    },
    # Defense Maps
    "offense_near": {
        "n_agents": 15,
        "n_enemies": 9,
        "n_neutrals": 8,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 8,
        "map_type": "plain_neut_tank",
    },
    "offense_distant": {
        "n_agents": 15,
        "n_enemies": 9,
        "n_neutrals": 8,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 8,
        "map_type": "plain_neut_tank",
    }, 
    "offense_complicated": {
        "n_agents": 15,
        "n_enemies": 9,
        "n_neutrals": 16,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 8,
        "map_type": "plain_neut_tank",
    },
    "offense_hard": {
        "n_agents": 9,
        "n_enemies": 9,
        "n_neutrals": 16,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 8,
        "map_type": "plain_neut_tank",
    },
    "offense_superhard": {
        "n_agents": 9,
        "n_enemies": 9,
        "n_neutrals": 16,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 8,
        "map_type": "plain_neut_tank",
    },
    
    #easy
    "hill_mTMt_def_1": {
        "n_agents": 5,
        "n_enemies": 7,
        "n_neutrals": 6,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 8,
        "map_type": "plain_neut_tank",
    },
    #Hard
    "hill_mTMt_def_3_split_jh": {
        "n_agents": 8,
        "n_enemies": 13,
        "n_neutrals": 6,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 8,
        "map_type": "plain_neut_tank",
    },
    #Super Hard
    "hill_mTMt_def_3_split_jh_M1M1": {
        "n_agents": 8,
        "n_enemies": 15,
        "n_neutrals": 6,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 8,
        "map_type": "plain_neut_tank",
    },
    "defense_outnumbered_random_each_1morehill": {
        "n_agents": 8,
        "n_enemies": 15,
        "n_neutrals": 6,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 8,
        "map_type": "plain_neut_tank",
    },
    "defense_outnumbered_1Marauder": {
        "n_agents": 8,
        "n_enemies": 16,
        "n_neutrals": 6,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 8,
        "map_type": "plain_neut_tank",
    },
    "defense_outnumbered_1Marauder_random_each": {
        "n_agents": 8,
        "n_enemies": 16,
        "n_neutrals": 6,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 8,
        "map_type": "plain_neut_tank",
    },
    "hill_mTMt_def_split_jh_1T": {
        "n_agents": 8,
        "n_enemies": 14,
        "n_neutrals": 6,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 8,
        "map_type": "plain_neut_tank",
    },
}


def get_smac_map_registry():
    return map_param_registry


for name in map_param_registry.keys():
    globals()[name] = type(name, (SMAC_Plus_Map,), dict(filename=name))
