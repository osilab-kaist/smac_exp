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
    # Offense Maps
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
    # Defense Maps
    "defense_infantry": {
        "n_agents": 5,
        "n_enemies": 7,
        "n_neutrals": 6,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 8,
        "map_type": "plain_neut_tank",
    },
    "defense_armored": {
        "n_agents": 8,
        "n_enemies": 13,
        "n_neutrals": 6,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 8,
        "map_type": "plain_neut_tank",
    },
    "defense_outnumbered": {
        "n_agents": 8,
        "n_enemies": 15,
        "n_neutrals": 6,
        "limit": 120,
        "a_race": "T",
        "b_race": "T",
        "unit_type_bits": 8,
        "map_type": "plain_neut_tank",
    },
    "defense_superhard": {
        "n_agents": 8,
        "n_enemies": 16,
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
