# Mod-Cog tasks for multitask learning 

This repository contains code to implement the Mod-Cog tasks as described in <a href ='https://arxiv.org/abs/2207.03523'> Khona*, Chandra* et al (2022)</a>. It is built on the <a href= 'https://github.com/neurogym/neurogym'> neurogym </a>framework created and maintained by <a href ='https://www.metaconscious.org/author/guangyu-robert-yang/'> Prof. Guangyu Robert Yang</a> and others.

## What are these tasks?
These tasks are designed to be modular and built on the original <a href = 'https://www.nature.com/articles/s41593-018-0310-2'> 20 cognitive tasks</a>. They consist of 2 main additions: "int" and "seq" specifically designed so that RNNs with diagonal weight matrices cannot solve them and are inspired by sequence production and interval estimation.

The "int" extension asks for a delay-dependent shift in the location of the bump on the ring, so the RNN has to use the rule input to shift the location of the bump it receives in the appropriate stimulus modality during the delay period of the task. The "seq" extension asks for a time-varying output that moves on the ring. These extensions are further augmented by "r" or "l" which determine whether the movement is clockwise or counter-clockwise. These computations fundamentally involve rotational dynamics and thus rely on the lateral recurrent connectivity of the RNN to be implemented.

The "seq" extension can be added to any task, with either "r" or "l". This triples the number of tasks to 60.
<img width="1027" alt="image" src="https://user-images.githubusercontent.com/49488315/220438786-8c95d34f-f86e-4762-a198-4af4f9863f51.png">

The "int" extension can be added to tasks with a delay period such as dms, dnms, dmc, dnmc, dlygo, dlyanti, ctxdlydm1, ctxdlydm2, again with either "r" or "l".
<img width="1027" alt="image" src="https://user-images.githubusercontent.com/49488315/220438845-1eea5a1f-e0cb-444c-b586-f6c22b5b93bb.png">

In principle, both these extensions can be added to a task to increase its complexity. 
<img width="713" alt="image" src="https://user-images.githubusercontent.com/49488315/220438977-90110654-6cdd-4fd5-9e11-cba3143d75aa.png">

## Explore
To have a playground where you can customize, visualize and possibly build on these tasks, I have created a minimal Colab notebook: https://colab.research.google.com/drive/1nzfOmJuVQ-GzbzP5oNarBk0o3XuOetMQ?usp=sharing

## Installation

First, install <a href= 'https://github.com/neurogym/neurogym#Installation'>neurogym</a>:
```python
pip install -U gym==0.18.3
git clone https://github.com/neurogym/neurogym.git
cd neurogym
pip install -e .
```
Next install Mod-Cog and import the tasks:

```python
git clone https://github.com/mikailkhona/Mod_Cog.git
from Mod_Cog.mod_cog_tasks import *
```

## Contact
If you have questions or suggestions (or find mistakes) please email me at mikailkhona@gmail.com.

## BibTeX Citation
If you use Mod-Cog in a scientific publication, we would appreciate using the following citation:

Winning the lottery with neurobiology: faster learning on many cognitive tasks with fixed sparse RNNs

Mikail Khona*, Sarthak Chandra*, Joy J. Ma, Ila Fiete

https://arxiv.org/abs/2207.03523

```
@article{Khona2022,
    url       = {[https://arxiv.org/abs/2207.03523]},
    year      = {2022},
    author    = {Mikail Khona* and Sarthak Chandra* and Joy J. Ma and Ila Fiete},
    title     = {Winning the lottery with neurobiology: faster learning on many cognitive tasks with fixed sparse RNNs},
    journal   = {arXiv}
}
```
