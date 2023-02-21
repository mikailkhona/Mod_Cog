# Mod-Cog tasks for multitask learning 

This repository contains code to implement the Mod-Cog tasks as described in <a href ='https://arxiv.org/abs/2207.03523'> Khona*, Chandra* et al (2022)</a>. It is built on the <a href= 'https://github.com/neurogym/neurogym'> neurogym </a>framework created by <a href ='https://www.metaconscious.org/author/guangyu-robert-yang/'> Guangyu Robert Yang</a>.

These tasks are designed to be modular and built on the original 20 cognitive tasks developed by Guangyu Robert Yang. They consist of 2 main additions: "int" and "seq" specifically designed so that RNNs with diagonal weight matrices cannot solve them and are inspired by sequence production and interval estimation. The "int" extension asks for a delay-dependent shift in the location of the bump on the ring and the "seq" extension asks for a time-varying output that moves on the ring.

Please cite the following publication if you use these tasks:

Winning the lottery with neurobiology: faster learning on many cognitive tasks with fixed sparse RNNs

Mikail Khona*, Sarthak Chandra*, Joy J. Ma, Ila Fiete

https://arxiv.org/abs/2207.03523

BibTex:
