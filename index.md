---
layout: page
---

# The StarCraft Multi-Agent Challenges+

The StarCraft Multi-Agent Challenges requires agents to learn completion of multi-stage tasks and use of environmental factors without precise reward functions. The previous challenges (SMAC) recognized as a standard benchmark of Multi-Agent Reinforcement Learning are mainly concerned with ensuring that all agents cooperatively eliminate approaching adversaries only through fine manipulation with obvious reward functions. This challenge, on the other hand, is interested in the exploration capability of MARL algorithms to efficiently learn implicit multi-stage tasks and environmental factors as well as micro-control. This study covers both offensive and defensive scenarios. In the offensive scenarios, agents must learn to first find opponents and then eliminate them. The defensive scenarios require agents to use topographic features. For example, agents need to position themselves behind protective structures to make it harder for enemies to attack. We investigate MARL algorithms under SMAC+ and observe that recent approaches work well in similar settings to the previous challenges, but misbehave in offensive scenarios. Additionally, we observe that an enhanced exploration approach has a positive effect on performance but is not able to completely solve all scenarios. This study proposes a new axis of future research. 

| ![gif_off](/assets/gif/SMAC_plus_off.GIF){: width="550" } | ![gif_def](/assets/gif/SMAC_plus_def.GIF){: width="550"} |  
|:--:| |:--:| 
| *<span style="font-family:Raleway; font-size:0.9em;"> SMAC+_Offense </span>*  | *<span style="font-family:Raleway; font-size:0.9em;"> SMAC+_Defense </span>* |

<br/>

## Paper and Source Codes
Here is the [Paper] and [Code] for the benchmarks and implemented baselines.  

<br/>  

## General Description of SMAC+
<hr>

This challenge offers advanced environmental factors such as destructible structures that can be used to conceal enemies and terrain features, such as a hill, that may be used to mitigate damages. Also, we newly introduce offensive scenarios that demand sequential completion of multi-stage tasks requiring finding adversaries initially and then eliminating them. Like in SMAC, both defensive and offensive scenarios in SMAC+ employ the reward function proportional to the number of enemies removed. 

### Comparison between [SMAC] and [SMAC+]

| Main Issues           | SMAC   | SMAC+ |
|:-----------------------:|:--------:|:-------:|
| Agents micro-control | O     |  O     |
| Multi-stage tasks     | ▵ |  O     |
| Environmental factors  | ▵        |  O     |

<br/>  

### List of environmental factors and multi-stage tasks for both [SMAC] and [SMAC+]

In SMAC, some difficult scenarios, such as *2c\_vs\_64zg* and *corridor*, require agents to indirectly learn environmental factors, such as exploiting different levels of terrains or discover multi-stage tasks like avoiding rushing enemies first and then eliminating individuals without a specific reward for them. However, those scenarios do not allow quantitative assessment of the algorithm's exploration capabilities, as they do not accurately reflect the difficulty of the task, which depends on the complexity of multi-stage tasks and the significance of environmental factors.  

To address this issue, we propose a new class of the StarCraft Multi-Agent Challenges+ that encompasses advanced and sophisticated multi-stage tasks, and involves environmental factors agents must learn to accomplish, as seen in the table as follows.


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; • **[SMAC]**

|                       |        2c_vs_64zg        |               Corridor               |
|:-----------------------:|:-------------------------------:|:-------------------------------------------:|
| Environmental factors | Different levels of the terrain |        Limited sight range of enemies       | 
| Multi-stage tasks     |                -                | Avoid enemies first, eliminate individually | 

<br/>  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; • **[SMAC+]**

|                       |                  Defense         |                                 Offese                                |
|:-----------------------:|:-------------------------------:|:-------------------------------------------:|
| Environmental factors |  Destroy obstacles hiding enemies |   Approach enemies strategically<br/> Discover a detour<br/> Destory moving impediments |
| Multi-stage tasks     |                  -                |             Identify where enemies place, then exterminate enemies            |

<br/>  
<br/>  

## Defensive Scenarios in SMAC+
<hr>
- Graphical explanation of defensive scenarios

![Def_infantry](/assets/imgs/def/defense_infantry.png){: width="33%" }
![Def_armored](/assets/imgs/def/defense_armored.png){: width="33%" }
![Def_outnumbered](/assets/imgs/def/defense_outnumbered.png){: width="33%" }

<br/>  

In defensive scenarios, we place allies on the hill and adversaries on the plain. We emphasize the importance of agents defeating adversaries utilizing topographical factors. The defensive scenarios in SMAC$^{+}$ are almost identical to those in SMAC. However, our environment expands the exploration range of allies to scout for the direction of offense by allowing enemies to attack in several directions and adding topographical changes. We control the difficulties of the defensive settings as follows.

|     Scenario    | Supply difference  | Opponents approach |
|:---------------:|:------------------:|:------------------:|
|   Def_infantry  |         -2         |      One-sided     |
|   Def_armored   |         -6         |      Two-sided     |
| Def_outnumbered |         -9         |      Two-sided     |

<br/>  

## Offensive Scenarios in SMAC+
<hr>

- Graphical explanation of offensive scenarios

![Off_near](/assets/imgs/off/offense_near.png){: width="33%" }
![Off_distant](/assets/imgs/off/offense_distant.png){: width="33%" }
![Off_complicated](/assets/imgs/off/offense_complicated.png){: width="33%" }

<br/>  
                         
Offensive scenarios provide learning of multi-stage tasks without direct incentives in MARL challenges. We suggest that agents should accomplish goals incrementally, such as eliminating adversaries after locating them. To observe a clear multi-stage structure, we allocate thirteen supplies to the allies more than the enemies. Hence, as soon as enemies are located, the agents rapidly learn to destroy enemies. As detailed in the previous table, in SMAC+, agents will not have a chance to get a reward if they do not encounter adversaries. This is because there are only three circumstances in which agents can get rewards: when agents defeat an adversary, kill an adversary, or inflict harm on an adversary. As a result, the main challenges necessitate not only micro-management, but also exploration to locate enemies.

|     Scenario      | Distance from opponents   |
|:---------------:  |:------------------:       |
|   Off_near        |         Near              |
|   Off_distance    |         Distant           |
| Off_complicated   |         Complicated       |

<br/>  

## Challenging Scenarios in SMAC+
<hr>

- Graphical explanation of challenging scenarios

![Off_hard](/assets/imgs/cha/offense_hard.png){: width="40%" }
![Off_superhard](/assets/imgs/cha/offense_superhard.png){: width="40%" }

As a result, the main challenges necessitate not only micro-management, but also exploration to locate enemies. For instance, the agents learn to separate the allied troops, locate the enemies, and effectively use armored troops like a long-ranged siege Tank. We measure the exploration strategy of effectively finding the enemy through this scenario. In this study, we examine the efficiency with which MARL algorithms explore to identify enemies by altering distance from them. In addition, to create more challenging scenarios, we show how enemy formation affects difficulty. 

|    Scenario   | Supply difference | Opponents formation |
|:-------------:|:-----------------:|:-------------------:|
|    Off_hard   |         0         |        Spread       |
| Off_superhard |         0         |        Gather       |

<br/>  
<br/>  

## Benchmarks 
<hr>

To demonstrate the need for assessment of exploration capabilities, we choose eleven algorithms of MARL algorithms classified into three categories; policy gradient algorithms, typical value-based algorithms, and distributional value-based algorithms. First, as an initial study of the MARL domain, policy gradient algorithms such as [COMA](https://arxiv.org/abs/1705.08926), [MASAC](https://arxiv.org/abs/2104.06655), [MADDPG](https://arxiv.org/abs/1706.02275) are considered. The typical value-based algorithm including [IQL](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.55.8066), [VDN](https://arxiv.org/abs/1706.05296), [QMIX](https://arxiv.org/abs/1803.11485) and [QTRAN](https://arxiv.org/abs/1905.05408) are chosen as baselines. Last but not least, we choose [DIQL, DDN, DMIX](https://arxiv.org/abs/2102.07936) and [DRIMA](https://openreview.net/forum?id=5qwA7LLbgP0) as distributional value-based algorithms that recently reported high performance owing to the effective exploration of difficult scenarios in SMAC.

### Defensive Sceanrios

We first look into defensive scenario experiments on SMAC+ to test whether MARL algorithms not only adequately employ environmental factors but also learn micro-controls. In terms of algorithmic performance, we observe COMA and QMIX drastically degrade, but MADDPG gradually degrades. This fact reveals that MADDPG enables agents to effectively learn micro-control. However, among baselines, DRIMA achieves the highest score and retains performance even when the supply difference significantly increases.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; • **<span style="font-family:Raleway; font-size:1.0em;"> Sequential Episodic Buffer </span>**  
![Def_infantry](/assets/results/def_infantry_sequential.png){: width="33%" }
![Def_armored](/assets/results/def_armored_sequential.png){: width="33%" }
![Def_outnumbered](/assets/results/def_outnumbered_sequential.png){: width="33%" }

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; • **<span style="font-family:Raleway; font-size:1.0em;"> Parallel Episodic Buffer </span>**  
![Def_infantry](/assets/results/def_infantry_parallel.png){: width="33%" }
![Def_armored](/assets/results/def_armored_parallel.png){: width="33%" }
![Def_outnumbered](/assets/results/def_outnumbered_parallel.png){: width="33%" }

<br/>

<iframe width="370" height="205" src="https://www.youtube.com/embed/slZviX1SARE" title="SMAC+: Def Infantry" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<iframe width="370" height="205" src="https://www.youtube.com/embed/rsVDBjB83fY" title="SMAC+:  Def Armored" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<iframe width="370" height="205" src="https://www.youtube.com/embed/Ys4kF-cSdYQ" title="SMAC+: Def Outnumbered" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<br/>

### Offensive Sceanrios

Regarding to offensive scenarios, we notice considerable performance differences of each baseline. Overall, even if an algorithm attains high scores at a trial, with exception of DRIMA, it is not guaranteed to train reliably in other trials. As mentioned, offensive scenarios do not require as much high micro-control as defensive scenarios, instead, it is important to locate enemies without direct incentives, such that when agents find enemies during training, the win-rate metric immediately goes to a high score. However, the finding enemies during training is decided by random actions drawn by epsilon-greedy or probabilistic policy, resulting in considerable variance in test outcome. In contrast, we see a perfect convergence of DRIMA in all offensive scenarios by employing its efficient exploration. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; •**<span style="font-family:Raleway; font-size:1.0em;"> Sequential Episodic Buffer </span>**  
![Off_near](/assets/results/off_near_sequential.png){: width="33%" }
![Off_distant](/assets/results/off_distant_sequential.png){: width="33%" }
![Off_complicated](/assets/results/off_complicated_sequential.png){: width="33%" }

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; •**<span style="font-family:Raleway; font-size:1.0em;"> Parallel Episodic Buffer </span>**  
![Off_near](/assets/results/off_near_parallel.png){: width="33%" }
![Off_distant](/assets/results/off_distant_parallel.png){: width="33%" }
![Off_complicated](/assets/results/off_complicated_parallel.png){: width="33%" }

<br/>

<iframe width="370" height="205" src="https://www.youtube.com/embed/zOFwSsI26Sc" title="SMAC+: off near (random policy)" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<iframe width="370" height="205" src="https://www.youtube.com/embed/4KVZboNel_0" title="SMAC+: Off near" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<iframe width="370" height="205" src="https://www.youtube.com/embed/r6pfkoOlYHw" title="SMAC+: Off Distant" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<iframe width="370" height="205" src="https://www.youtube.com/embed/1LjmT88iNJw" title="SMAC+: Off Complicated" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


<!-- <iframe width="400" height="444" src="https://www.youtube.com/embed/4KVZboNel_0" title="SMAC+: Off near" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<iframe width="400" height="444" src="https://www.youtube.com/embed/rLIdx_JIA30" title="SMAC+: Off Distant" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<iframe width="400" height="444" src="https://www.youtube.com/embed/1LjmT88iNJw" title="SMAC+: Off Complicated" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> -->

<br/>

### Challenging

To provide open-ended problems for the MARL domain, we suggest more challenging scenarios. In these scenarios, the agents are required to simultaneously learn completion of multi-stage tasks and micro-control during training. We argue that this scenario requires more sophisticated fine manipulation compared to other offensive scenarios. This is due to the fact that not only the strength of allies is identical to that of opponents, but also *Gathered* enables opponents to intensively strike allies at once. This indicates the necessity of more efficient exploration strategies for the completion of multi-stage tasks and micro-control. 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; •**<span style="font-family:Raleway; font-size:1.0em;"> Sequential Episodic Buffer </span>**  
<!-- ![Off_hard](/assets/results/off_hard_sequential.png){: width="40%" } -->
![Off_superhard](/assets/results/off_hard_parallel.png){: width="40%" }

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; •**<span style="font-family:Raleway; font-size:1.0em;"> Parallel Episodic Buffer </span>**  
<!-- ![Off_hard](/assets/results/off_superhard_sequential.png){: width="40%" } -->
![Off_superhard](/assets/results/off_superhard_parallel.png){: width="40%" }

<br/>

<iframe width="370" height="205" src="https://www.youtube.com/embed/o4__OnkYukM" title="SMAC+: Def Superhard" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<iframe width="370" height="205" src="https://www.youtube.com/embed/yNsphTrXwY4" title="SMAC+: Off Superhard" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<br/>
<br/>  


## Ablation Study : Heatmaps of Agents Movement
<hr>
Through the heat-maps of movements, we conduct qualitative analysis of each baseline. As illustrated in the heatmaps, at early stage of COMA only few agents find enemies. When it comes to end stage, agents do not face the opponents. Whereas although the early stage of DRIMA makes agents move around their starting points, its exploration capability makes agents get closer to enemies and finally confront them after training.  

![Off_near](/assets/heatmaps/off_near.png){: width="600" }
![Off_distant](/assets/heatmaps/off_distant.png){: width="600" }
![Off_complicated](/assets/heatmaps/off_complicated.png){: width="600" }
![Off_hard](/assets/heatmaps/off_hard.png){: width="600" }
![Off_superhard](/assets/heatmaps/off_superhard.png){: width="600" }


<br/>
<br/>  

## Future Direction
<hr>
This study takes a look at multi-stage tasks in MARL. We mostly work on two-stage tasks. Two-stage tasks appear somewhat simplistic, but as early stage works, this work provides meaningful direction for exploration capability in MARL domains. We will develop MARL environments with multiple stage tasks in the future. We hope this work serves as a valuable benchmark to evaluate the exploration capabilities of MARL algorithms and give guidance for future research.

<br/>
<br/>  

## Acknowledgments
<hr>
This work was conducted by Center for Applied Research in Artificial Intelligence (CARAI) grant funded by DAPA and ADD (UD190031RD).

<br/>
<br/>  

[Code]: https://github.com/osilab-kaist/smac_plus
[SMAC]: https://arxiv.org/abs/1902.04043
[SMAC+]: https://arxiv.org/abs/2207.02007
[Paper]: https://arxiv.org/abs/2207.02007
