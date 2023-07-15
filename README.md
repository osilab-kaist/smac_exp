<div align="center">
    
# The StarCraft Multi-Agent Exploration Challenges
    
**Bechmarks for Efficient Exploration of Multi-stage Tasks Completion and Utilization of Environmental Factors**

|<img src="/images/SMAC_plus_off.GIF" width="100%"> | <img src="/images/SMAC_plus_def.GIF" width="100%"> |
|:--------:|:-------:|
|*SMAC-Exp Offense*|*SMAC-Exp Defense*|

    
    
<!-- ![gif_off](/images/SMAC_plus_off.GIF){: width="45%"} ![gif_def](/images/SMAC_plus_def.GIF){: width="45%"}  -->

    
---
</div>


## 🧚 Preview

**[The StarCraft Multi-Agent Exploration Challenges : Learning Multi-Stage Tasks and Environmental Factors without Precise Reward Functions](https://arxiv.org/abs/2207.02007)**.   
[Mingyu Kim\*](https://mingyukim87.github.io), Jihwan Oh\*, Yongsik Lee, Joonkee Kim, Seonghwan Kim, Song Chong, Se-Young Yun.  
(\*:equal contribution)

* Paper : [https://ieeexplore.ieee.org/document/10099458](https://ieeexplore.ieee.org/document/10099458)
* Project page : [https://osilab-kaist.github.io/smac_exp/](https://osilab-kaist.github.io/smac_exp/)
* Paperwithcodes : [https://paperswithcode.com/paper/the-starcraft-multi-agent-challenges-learning](https://paperswithcode.com/paper/the-starcraft-multi-agent-challenges-learning)
* Learning curves : [https://url.kr/mak6gq](https://url.kr/mak6gq)
* Tensorboard logs and checkpoints : [https://url.kr/92bp83](https://url.kr/92bp83)

</br>

## ✍️ Notice
* [PyMARL](https://github.com/oxwhirl/pymarl) : The framework for deep multi-agent reinforcement learning with PyTorch.  
* [SMAC](https://github.com/oxwhirl/smac) : The environments for research in the field of collaborative multi-agent reinforcement learning (MARL) based on Blizzard's StarCraft II RTS game. We work on based on these.  
* [SMAC_Exp](https://github.com/osilab-kaist/smac_exp) : It provides its environment and exceutes training and testing RL algorithms based on PyMARL with both SMAC and SMAC_Exp.  

</br>

## 🆚 SMAC vs SMAC_Exp 
* [SMAC](https://github.com/oxwhirl/smac) is the standard benchmark of Multi-Agent Reinforcement Learning. It is mainly concerned with ensuring that all agents cooperatively eliminate approaching adversaries only through fine manipulation with obvious reward functions.  

* [SMAC_Exp](https://github.com/osilab-kaist/smac_exp) contains total 8 maps and categorizes into three types like `defense`, `offense`, `challenging`. It is interested in the exploration capability of MARL algorithms to efficiently learn implicitly multi-stage tasks and environmental factors as well as micro-control.  

| Main Issues           | SMAC   | SMAC_Exp |
|:-----------------------:|:--------:|:-------:|
| Agents micro-control | O     |  O     |
| Multi-stage tasks     | ▵ |  O     |
| Environmental factors  | ▵        |  O     |

</br>

## 🌏 Maps
* SG, Mar and M refer each Siege Tank, Marauder, Marine units.  

### **Defense**
<div align="center">

<img src="./images/defense_infantry.png" alt="drawing" width="200"/>
<img src="./images/defense_armored.png" alt="drawing" width="200"/>
<img src="./images/defense_outnumbered.png" alt="drawing" width="200"/>


</div>

| Name | Ally Units | Enemy Units | Opponents approach |
|:---:|:---:|:---:|:---:|
| `defense_infantry` | 1 Mar & 4 M | 1 Mar & 6 M | One-sided |
| `defense_armored` | 1 SG Tank, 1 Tank, 1 Mar & 5 M | 2 Tank, 2 Mar & 9 M | Two-sided |
| `defense_outnumbered` | 1 SG Tank, 1 Tank, 1 Mar & 5 M  | 2 Tank, 3 Mar & 10 M | Two-sided |


</br>

### **Offense**
<div align="center">

<img src="./images/offense_near.png" alt="drawing" width="200"/>
<img src="./images/offense_distant.png" alt="drawing" width="200"/>
<img src="./images/offense_complicated.png" alt="drawing" width="200"/>

</div>

| Name | Ally Units | Enemy Units | Distance & formation |
|:---:|:---:|:---:|:---:|
| `offense_near` |  3 SG Tank, 3 Tank, 3 Mar & 4 M| 1 SG Tank, 2 Tank, 2 Mar & 4 M | Near & Spread |
| `offense_distant` |  3 SG Tank, 3 Tank, 3 Mar & 4 M| 1 SG Tank, 2 Tank, 2 Mar & 4 M | Distant & Spread |
| `offense_complicated` |  3 SG Tank, 3 Tank, 3 Mar & 4 M| 1 SG Tank, 2 Tank, 2 Mar & 4 M | Complicated & Spread |

</br>

### **Challenging**
<div align="center">

<img src="./images/offense_hard.png" alt="drawing" width="200"/>
<img src="./images/offense_superhard.png" alt="drawing" width="200"/>

</div>

| Name | Ally Units | Enemy Units | Opponents approach |
|:---:|:---:|:---:|:---:|
| `defense_superhard` | 1 SG Tank, 1 Tank, 1 Mar & 5 M  | 2 Tank, 3 Mar & 10 M | Two-sided |


| Name | Ally Units | Enemy Units | Distance & formation |
|:---:|:---:|:---:|:---:|
| `offense_superhard` |  1 SG Tank, 2 Tank, 2 Mar & 4 M| 1 SG Tank, 2 Tank, 2 Mar & 4 M | Complicated & Gathered |

</br>

## 🎮 Implemented Algorithms
| Algorithm | Category | Paper Links|
|:---:|:---:|:---:|
| `IQL` | Value based | [paper](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.55.8066)|
| `VDN` | Value based |  [paper](https://arxiv.org/abs/1706.05296)|
| `QTRAN` | Value based |  [paper](https://arxiv.org/abs/1905.05408)|
| `QMIX` | Value based |  [paper](https://arxiv.org/abs/1803.11485)|
| `DIQL` | Distributional Value based | [paper](https://arxiv.org/abs/2102.07936)|
| `DDN` | Distributional Value based | [paper](https://arxiv.org/abs/2102.07936)|
| `DMIX` | Distributional Value based | [paper](https://arxiv.org/abs/2102.07936) |
| `DRIMA` | Distributional Value based | [paper](https://openreview.net/forum?id=5qwA7LLbgP0) |
| `COMA` | Policy Gradient based | [paper](https://arxiv.org/abs/1705.08926) |
| `MASAC` | Policy Gradient based | [paper](https://arxiv.org/abs/2104.06655) |
| `MADDPG` | Policy Gradient based | [paper](https://arxiv.org/abs/1706.02275) |
| `MAPPO` | Policy Gradient based | [paper](https://arxiv.org/abs/2103.01955) |

</br>

## ⚙️ Installation instructions
```diff
- Please pay attention to the version of SC2 you are using for your experiments. 
- Performance is *not* always comparable between versions. 
- The results in SMAC (https://arxiv.org/abs/1902.04043) use SC2.4.6.2.69232 not SC2.4.10.
- wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.6.2.69232.zip
```

</br>

1️⃣ **Cloning SMAC_Exp**
```shell
git clone https://github.com/osilab-kaist/smac_plus.git
```

</br>

2️⃣ **Download and set up StarCraft II**

```shell
bash install_sc2.sh
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - This will download SC2 into the `pymarl/3rdparty` folder, or using symbolic link to use SC2.  

</br>

3️⃣ **Install required packages**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- The `requirements.txt` file can be used to install the necessary packages into a virtual environment (not recommended).  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- After install requirements, install `torch` suitable for the environment.  
</br>

4️⃣ **Move map directoryes to StarCraftII map directory**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Move `SMAC_Maps` / `SMAC_Plus_Maps` directories to `StarCraftII/Maps/`. 

```shell
mv SMAC_Plus_Maps ./pymarl/3rdparty/StarCraftII/Maps/
mv SMAC_Maps ./pymarl/3rdparty/StarCraftII/Maps/
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - You should have a structure like these:


```
smac_exp
├── pymarl
│   ├── docker
│   ├── 3rdparty
│   │   └── StarCraftII 
│   │       ├── Maps
│   │       │   ├── SMAC_Maps
│   │       │   └── SMAC_Plus_Maps
│   │       └── ...
│   ├── src
│   └── results
├── smac_plus
├── requirements.txt
└── install_sc2.sh

```

</br>

## 🏃Run an experiment 
* Episode experience buffer
```shell
cd ./pymarl
python src/main.py --alg=qmix --env-config=smac_plus with env_args.map_name=offense_hard
python src/main.py --alg=qmix --env-config=smac with env_args.map_name=2s3z
```

* Parallel experience beffer
```shell
cd ./pymarl
python src/main.py --alg=qmix --env-config=smac_plus with env_args.map_name=offense_hard runner=parallel batch_size_run=20
python src/main.py --alg=qmix --env-config=smac with env_args.map_name=2s3z runner=parallel batch_size_run=20
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- The config files act as defaults for an algorithm or environment.   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- They are all located in `src/config`.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- `--config` refers to the config files in `src/config/algs`.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- `--env-config` refers to the config files in `src/config/envs`.   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- All results will be stored in the `results` folder.   

</br>

## 🏃Run an test 
```shell
cd ./pymarl
python src/main.py --alg=qmix --env-config=smac_plus with env_args.map_name=offense_hard save_replay=True checkpoint_path={checkpoint_dir_path} load_step={n_steps} test_nepisode={n_test}
python src/main.py --alg=qmix --env-config=smac with env_args.map_name=2s3z save_replay=True checkpoint_path={checkpoint_dir_path} load_step={n_steps} test_nepisode={n_test}
```

</br>

## Logs and Checkpoints

* While we are developing a benchmark from Feb., 2022, we encountered an unexpectable difficulty like deleting stored result files owing to malfunction of computation resources. 
* Therefore, despite of restroing these files, we provides partial information of pretrained checkpoints and tensorboard logs. Please refer to this URL([https://url.kr/92bp83](https://url.kr/92bp83))
* Instead, we completely provide training curves of all algorithms and its test scores. Please see the provided csv files ([https://url.kr/mak6gq](https://url.kr/mak6gq)).

```
Par : Parallel experience buffer
Seq : Sequential experience buffer
```

```
O : Exsistence of tensorboard logs and checkpoints.   
▵ : Exsitence of either tensorboard logs or checkpoints.   
X : Absence of tensorboard logs and checkpoints.   
```

#### Defense scenarios

|        | (Par)Def_infantry |   |   | (Par)Def_armored |   |   | (Par)Def_outnumbered |   |   |
|:------:|:-----------------:|:-:|:-:|:----------------:|:-:|:-:|:--------------------:|:-:|:-:|
|        |         1         | 2 | 3 |         1        | 2 | 3 |           1          | 2 | 3 |
|   IQL  |         O         | O | O |         O        | O | O |           O          | O | O |
|   VDN  |         -         | - | - |         -        | - | - |           -          | - | - |
|  QMIX  |         -         | - | - |         -        | - | - |           -          | - | - |
|  QTRAN |         -         | - | - |         -        | - | - |           -          | - | - |
|  COMA  |         O         | O | O |         O        | O | O |           O          | O | O |
|  MASAC |         O         | O | O |         O        | O | O |           O          | O | O |
|  MAPPO |         -         | - | - |         -        | - | - |           -          | - | - |
|  DIQL  |         O         | O | O |         O        | O | O |           O          | O | O |
|   DDN  |         -         | - | - |         -        | - | - |           -          | - | - |
|  DMIX  |         -         | - | - |         -        | - | - |           -          | - | - |
|  DRIMA |         -         | - | - |         -        | - | - |           -          | - | - |

|        | (Seq)Def_infantry |   |   | (Seq)Def_armored |   |   | (Seq)Def_outnumbered |   |   |
|:------:|:-----------------:|:-:|:-:|:----------------:|:-:|:-:|:--------------------:|:-:|:-:|
|        |         1         | 2 | 3 |         1        | 2 | 3 |           1          | 2 | 3 |
|   IQL  |         O         | O | O |         O        | O | O |           O          | O | O |
|   VDN  |         O         | O | O |         O        | O | O |           -          | - | - |
|  QMIX  |         O         | O | O |         O        | O | - |           O          | O | - |
|  QTRAN |         O         | O | O |         O        | O | O |           -          | - | - |
|  COMA  |         O         | O | - |         O        | O | O |           O          | O | O |
| MADDPG |         O         | O | - |         O        | O | - |           O          | O | O |
|  MASAC |         O         | O | O |         O        | O | O |           O          | O | O |
|  MAPPO |         -         | - | - |         -        | - | - |           -          | - | - |
|  DIQL  |         O         | O | O |         O        | O | O |           O          | O | O |
|   DDN  |         O         | O | O |         O        | O | O |           O          | O | O |
|  DMIX  |         O         | O | O |         O        | O | O |           O          | O | O |
|  DRIMA |         O         | O | - |         O        | ▵(Model) | - |           O          | ▵(Model) | - |

</br>

#### Offensive scenarios

|        | (Par)Off_near |   |   | (Par)Off_distant |   |   | (Par)Off_complicated |   |   |
|:------:|:-----------------:|:-:|:-:|:----------------:|:-:|:-:|:--------------------:|:-:|:-:|
|        |         1         | 2 | 3 |         1        | 2 | 3 |           1          | 2 | 3 |
|   IQL  |         O         | O | O |         O        | O | O |           O          | O | O |
|   VDN  |         -         | - | - |         -        | - | - |           -          | - | - |
|  QMIX  |         -         | - | - |         -        | - | - |           -          | - | - |
|  QTRAN |         -         | - | - |         -        | - | - |           -          | - | - |
|  COMA  |         O         | O | O |         O        | O | O |           O          | O | O |
|  MASAC |         O         | O | O |         O        | O | O |           O          | O | O |
|  MAPPO |         -         | - | - |         -        | - | - |           -          | - | - |
|  DIQL  |         O         | O | O |         O        | O | O |           O          | O | O |
|   DDN  |         -         | - | - |         -        | - | - |           -          | - | - |
|  DMIX  |         -         | - | - |         -        | - | - |           -          | - | - |
|  DRIMA |         -         | - | - |         -        | - | - |           -          | - | - |

|        | (Seq)Off_near |   |   | (Seq)Off_distant |   |   | (Seq)Off_complicated |   |   |
|:------:|:-----------------:|:-:|:-:|:----------------:|:-:|:-:|:--------------------:|:-:|:-:|
|        |         1         | 2 | 3 |         1        | 2 | 3 |           1          | 2 | 3 |
|   IQL  |         -         | - | - |         -        | - | - |           -          | - | - |
|   VDN  |         -         | - | - |         -        | - | - |           -          | - | - |
|  QMIX  |         ▵(Model)         | ▵(Model) | - |         O        | - | O |           O          | O | - |
|  QTRAN |         -         | - | - |         -        | - | - |           -          | - | - |
|  COMA  |         O         | O | O |         O        | O | O |           -          | - | - |
| MADDPG |         O         | O | O |         -        | - | - |           O          | O | - |
|  MASAC |         -         | - | - |         -        | - | - |           -          | - | - |
|  MAPPO |         -         | - | - |         -        | - | - |           -          | - | - |
|  DIQL  |         -         | - | - |         -        | - | - |           -          | - | - |
|   DDN  |         -         | - | - |         -        | - | - |           -          | - | - |
|  DMIX  |         -         | - | - |         -        | - | - |           -          | - | - |
|  DRIMA |         -         | ▵(Model) | O |         O        | O | - |           O          | O | O |

</br>

#### Challenging scenarios

|        | (Par)Off_hard |   |   | (Par)Off_superhard |   |   |
|:------:|:-------------:|:-:|:-:|:------------------:|:-:|:-:|
|        |       1       | 2 | 3 |          1         | 2 | 3 |
|   IQL  |       O       | O | O |          O         | O | O |
|   VDN  |       O       | O | O |          ▵(Model)         | O | O |
|  QMIX  |       O       | O | O |          O         | O | O |
|  QTRAN |       O       | O | O |          -         | - | - |
|  COMA  |       O       | O | O |          O         | O | O |
|  MASAC |       O       | O | O |          O         | O | O |
|  MAPPO |       -       | - | - |          -         | - | - |
|  DIQL  |       O       | O | O |          O         | O | O |
|   DDN  |       O       | O | O |          -         | - | - |
|  DMIX  |       O       | O | O |          O         | O | O |
|  DRIMA |       -       | - | - |          -         | - | - |


|        | (Seq)Off_hard |   |   | (Seq)Off_superhard |   |   |
|:------:|:-------------:|:-:|:-:|:------------------:|:-:|:-:|
|        |       1       | 2 | 3 |          1         | 2 | 3 |
|   IQL  |       -       | - | - |          -         | - | - |
|   VDN  |       -       | - | - |          -         | - | - |
|  QMIX  |       O       | O | O |          O         | O | O |
|  QTRAN |       -       | - | - |          -         | - | - |
|  COMA  |       O       | O | O |          O         | O | O |
| MADDPG |       -       | - | - |         O          | O | O |           
|  MASAC |       -       | - | - |          -         | - | - |
|  MAPPO |       -       | - | - |          -         | - | - |
|  DIQL  |       -       | - | - |         O          | O | O |           
|   DDN  |       -       | - | - |         O          | O | O |           
|  DMIX  |       -       | - | - |         O          | O | O |           
|  DRIMA |       ▵(Model)       | ▵(Model) | ▵(Model) |          O         | O | O |

</br>
  
## 🤝 License

* The original SMAC environment and PyMARL code follow the MIT license and Apache 2.0 license respectively. The proposed SMAC-Exp environment and the modified PyMARL code are also released under the MIT license and Apache 2.0 license each.

</br>

## 📌 Citation
```bibtext
@article{kim2022smacplus,
  title={The StarCraft Multi-Agent Exploration Challenges : Learning Multi-Stage Tasks and Environmental Factors without Precise Reward Functions},
  author={Mingyu Kim, Jihwan Oh, Yongsik Lee, Joonkee Kim, Seonghwan Kim, Song Chong, Se-Young Yun},
  journal={arXiv preprint arXiv:2207.02007},
  year={2022}
}
```
<!-- ## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

## Watching StarCraft II replays

`save_replay` option allows saving replays of models which are loaded using `checkpoint_path`. Once the model is successfully loaded, `test_nepisode` number of episodes are run on the test mode and a .SC2Replay file is saved in the Replay directory of StarCraft II. Please make sure to use the episode runner if you wish to save a replay, i.e., `runner=episode`. The name of the saved replay file starts with the given `env_args.save_replay_prefix` (map_name if empty), followed by the current timestamp. 

The saved replays can be watched by double-clicking on them or using the following command:

```shell
python -m pysc2.bin.play --norender --rgb_minimap_size 0 --replay NAME.SC2Replay
```

**Note:** Replays cannot be watched using the Linux version of StarCraft II. Please use either the Mac or Windows version of the StarCraft II client.

## Documentation/Support

Documentation is a little sparse at the moment (but will improve!). Please raise an issue in this repo, or email [Tabish](mailto:tabish.rashid@cs.ox.ac.uk) -->

<!-- ## Citing PyMARL 

If you use PyMARL in your research, please cite the [SMAC Plus]().

*M. Samvelyan, T. Rashid, C. Schroeder de Witt, G. Farquhar, N. Nardelli, T.G.J. Rudner, C.-M. Hung, P.H.S. Torr, J. Foerster, S. Whiteson. The StarCraft Multi-Agent Challenge, CoRR abs/1902.04043, 2019.*

In BibTeX format:

```tex
@article{samvelyan19smac,
  title = {{The} {StarCraft} {Multi}-{Agent} {Challenge}},
  author = {Mikayel Samvelyan and Tabish Rashid and Christian Schroeder de Witt and Gregory Farquhar and Nantas Nardelli and Tim G. J. Rudner and Chia-Man Hung and Philiph H. S. Torr and Jakob Foerster and Shimon Whiteson},
  journal = {CoRR},
  volume = {abs/1902.04043},
  year = {2019},
}
```

</br>

## License

Code licensed under the Apache License v2.0 -->
