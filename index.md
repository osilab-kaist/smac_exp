# SMAC+

### Abstract
 In this paper, we propose a novel benchmark called SMAC+, where agents learn to perform multi-stage tasks and to use environmental benefits without precise reward functions. 
 The StarCraft Multi-Agent Challenges (SMAC) recognized as a standard benchmark of Multi-Agent Reinforcement Learning (MARL) is mainly concerned with ensuring that all agents cooperatively eliminate approaching adversaries only through fine manipulation with obvious reward functions. 
 SMAC+, on the other hand, is interested in exploration capability of MARL algorithms to efficiently learn intrinsic tasks and advantages as well as micro-control. This study presents two types scenarios. 
 In the offensive scenarios, agents must learn to find opponents first and then eliminate them, whereas the defensive scenarios need agents to use topographic features such as placing behind structures to lower the possibility of being attacked by enemies. 
 In those scenarios, MARL algorithms must learn indirectly how they perform usage of benefits or multi-stage tasks without direct incentives relying on their exploration. 
 We investigate MARL algorithms under SMAC+ and observe that recent approaches work well in similar settings to the previous challenges but misbehave in offensive scenarios, even when training time is significantly extended. 
 We also discover that risk-based extra exploration approach has a positive effect on performance through the completion of sub-tasks.
### Demo


**Put some GIF things**


### Installation Guide
**Git clone SMAC_PLUS**:
```shell
git clone https://github.com/osilab-kaist/smac_plus.git
```

**Download StarCraft II**
Set up StarCraft II:
```shell
bash install_sc2.sh
```

This will download SC2 into the `pymarl/3rdparty` folder.

The `requirements.txt` file can be used to install the necessary packages into a virtual environment (not recommended).

**Move map directoryes to StarCraftII map directory**

Move `SMAC_Maps` / `SMAC_Plus_Maps` directories to `StarCraftII/Maps/`

'''shell
mv SMAC_Plus_Maps ./pymarl/3rdparty/StarCraftII/Maps/
mv SMAC_Maps ./pymarl/3rdparty/StarCraftII/Maps/
'''
### Run

**Episode mode**
```shell
cd ./pymarl
python src/main.py --alg=qmix --env-config=smac_plus with env_args.map_name=offense_hard
python src/main.py --alg=qmix --env-config=smac with env_args.map_name=2s3z
```

**Parallel mode**
```shell
cd ./pymarl
python src/main.py --alg=qmix --env-config=smac_plus with env_args.map_name=offense_hard runner=parallel batch_size_run=20
python src/main.py --alg=qmix --env-config=smac with env_args.map_name=2s3z runner=parallel batch_size_run=20
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`