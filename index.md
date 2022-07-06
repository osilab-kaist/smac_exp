---
layout: page
---

# The StarCraft Multi-Agent Challenges+

The StarCraft Multi-Agent Challenges requires agents to learn completion of multi-stage tasks and use of environmental factors without precise reward functions. The previous challenges (SMAC) recognized as a standard benchmark of Multi-Agent Reinforcement Learning are mainly concerned with ensuring that all agents cooperatively eliminate approaching adversaries only through fine manipulation with obvious reward functions. This challenge, on the other hand, is interested in the exploration capability of MARL algorithms to efficiently learn implicit multi-stage tasks and environmental factors as well as micro-control. This study covers both offensive and defensive scenarios. In the offensive scenarios, agents must learn to first find opponents and then eliminate them. The defensive scenarios require agents to use topographic features. For example, agents need to position themselves behind protective structures to make it harder for enemies to attack. We investigate MARL algorithms under SMAC+ and observe that recent approaches work well in similar settings to the previous challenges, but misbehave in offensive scenarios. Additionally, we observe that an enhanced exploration approach has a positive effect on performance but is not able to completely solve all scenarios. This study proposes a new axis of future research. 

<br/>

## Source Codes
Here is the [Code] for the benchmark and implemented baselines.  
<br/>  

## SMAC+ Descriptions
<hr>

**<span style="font-family:Raleway; font-size:1.0em;"> We need defensive / offensive scenarios overview </span>**  

<br/>  
<br/>  

## SMAC+ Benchmarks
<hr>

### Defensive Sceanrios
**<span style="font-family:Raleway; font-size:1.0em;"> Sequential Episodic Buffer </span>**  
TBD  

**<span style="font-family:Raleway; font-size:1.0em;"> Parallel Episodic Buffer </span>**  
TBD  
<br/>

### Offensive Sceanrios
**<span style="font-family:Raleway; font-size:1.0em;"> Sequential Episodic Buffer </span>**  
TBD  

**<span style="font-family:Raleway; font-size:1.0em;"> Parallel Episodic Buffer </span>**  
TBD  
<br/>

### Challenging
**<span style="font-family:Raleway; font-size:1.0em;"> Sequential Episodic Buffer </span>**  
TBD  

**<span style="font-family:Raleway; font-size:1.0em;"> Parallel Episodic Buffer </span>**  
TBD  
<br/>
<br/>  


## Ablation Study
<hr>

**<span style="font-family:Raleway; font-size:1.0em;"> Reward Engineering </span>**  
TBD  

**<span style="font-family:Raleway; font-size:1.0em;"> Heatmaps </span>**  
TBD 

<br/>
<br/>  

## SMAC+ Future Direction
<hr>
TBD   

<br/>
<br/>  

## Acknowledgments
<hr>
This work was conducted by Center for Applied Research in Artificial Intelligence (CARAI) grant funded by DAPA and ADD [UD190031RD] and supported by Institute of Information \& communications Technology Planning \& Evaluation (IITP) grant funded by the Korea government (MSIT) [No.2019-0-00075, Artificial Intelligence Graduate School Program (KAIST)].  

<br/>
<br/>  

[Code]: https://github.com/osilab-kaist/smac_plus