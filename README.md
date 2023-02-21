# Mod-Cog tasks for multitask learning 

This repository contains code to implement the Mod-Cog tasks as described in <a href ='https://arxiv.org/abs/2207.03523'> Khona*, Chandra* et al (2022)</a>. It is built on the <a href= 'https://github.com/neurogym/neurogym'> neurogym </a>framework created and maintained by <a href ='https://www.metaconscious.org/author/guangyu-robert-yang/'> Prof. Guangyu Robert Yang</a> and others.

## What are these tasks?
These tasks are designed to be modular and built on the original <a href = 'https://www.nature.com/articles/s41593-018-0310-2'> 20 cognitive tasks </a>. They consist of 2 main additions: "int" and "seq" specifically designed so that RNNs with diagonal weight matrices cannot solve them and are inspired by sequence production and interval estimation.

The "int" extension asks for a delay-dependent shift in the location of the bump on the ring, so the RNN has to use the rule input to shift the location of the bump it receives in the appropriate stimulus modality during the delay period of the task. The "seq" extension asks for a time-varying output that moves on the ring. These extensions are further augmented by "r" or "l" which determine whether the movement is clockwise or counter-clockwise. These computations fundamentally involve rotational dynamics and thus rely on the lateral recurrent connectivity of the RNN to be implemented.


## Citation statement
Please cite the following publication if you use these tasks:

Winning the lottery with neurobiology: faster learning on many cognitive tasks with fixed sparse RNNs

Mikail Khona*, Sarthak Chandra*, Joy J. Ma, Ila Fiete

https://arxiv.org/abs/2207.03523

BibTex:
