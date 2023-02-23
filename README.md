# Mod-Cog tasks for multitask learning 

This repository contains code to implement the Mod-Cog tasks as described in <a href ='https://arxiv.org/abs/2207.03523'> Khona*, Chandra* et al (2022)</a>. It is built on the <a href= 'https://github.com/neurogym/neurogym'> neurogym </a>framework created and maintained by <a href ='https://www.metaconscious.org/author/guangyu-robert-yang/'> Prof. Guangyu Robert Yang</a> and <a href ='https://manuelmolano.wixsite.com/home'> Dr. Manuel Molano-Mazon</a> and others.

## What are these tasks?
The Mod-Cog task set is modular, creating tasks from compositions of a set of subtasks. It is built on a similar framework as an earlier <a href = 'https://www.nature.com/articles/s41593-018-0310-2'> 20 cognitive task set</a>. It includes two additional subtasks, involving integration for interval integration ("int") and sequence production ("seq"). "Int" and "seq" are specifically designed so that RNNs with diagonal weight matrices (pools of unconnected autapses) cannot solve them. 

The "int" extension asks for a delay-dependent shift in the location of the bump on the ring, so the RNN has to use the rule input to shift the location of the bump it receives in the appropriate stimulus modality during the delay period of the task. The "seq" extension asks for a time-varying output that moves on the ring. These extensions are further augmented by "r" or "l" which determine whether the movement is clockwise or counter-clockwise. These computations fundamentally involve rotational dynamics and thus rely on the lateral recurrent connectivity of the RNN to be implemented.

The "seq" extension can be added to any task, with either "r" or "l". This triples the number of tasks to 60.
<img width="1027" alt="image" src="https://user-images.githubusercontent.com/49488315/220438786-8c95d34f-f86e-4762-a198-4af4f9863f51.png">

The "int" extension can be added to tasks with a delay period such as dms, dnms, dmc, dnmc, dlygo, dlyanti, dlydm1, dlydm2, ctxdlydm1, ctxdlydm2, multidlydm again with either "r" or "l".
<img width="1027" alt="image" src="https://user-images.githubusercontent.com/49488315/220438845-1eea5a1f-e0cb-444c-b586-f6c22b5b93bb.png">

In principle, both these extensions can be added to a task to increase its complexity. This increases the number of tasks to 126. 

<img width="713" alt="image" src="https://user-images.githubusercontent.com/49488315/220438977-90110654-6cdd-4fd5-9e11-cba3143d75aa.png">

## Explore
To have a playground where you can customize, visualize and possibly build on these tasks, I have created a minimal Colab notebook: https://colab.research.google.com/drive/1nzfOmJuVQ-GzbzP5oNarBk0o3XuOetMQ?usp=sharing

## Installation

First, install <a href= 'https://github.com/neurogym/neurogym#Installation'>neurogym</a>:
```python
pip install gym
git clone https://github.com/neurogym/neurogym.git
cd neurogym
pip install -e .
```
Next install Mod-Cog. Note that Mod-Cog creates wrappers around some of neurogym's core functions.

```python
git clone https://github.com/mikailkhona/Mod_Cog.git

```

You can create a neurogym dataset with 82 tasks using the following code:

```python
from neurogym.wrappers import ScheduleEnvs
from neurogym.utils.scheduler import RandomSchedule
from neurogym.wrappers.block import MultiEnvs
from neurogym import Dataset
from Mod_Cog.mod_cog_tasks import *

envs = [go(), rtgo(), dlygo(), anti(), rtanti(), dlyanti(),
        dm1(), dm2(), ctxdm1(), ctxdm2(), multidm(), dlydm1(), dlydm2(),
        ctxdlydm1(), ctxdlydm2(), multidlydm(), dms(), dnms(), dmc(), dnmc(),
        dlygointr(),dlygointl(),dlyantiintr(),dlyantiintl(),dlydm1intr(),dlydm1intl(),
        dlydm2intr(),dlydm2intl(),ctxdlydm1intr(),ctxdlydm1intl(),ctxdlydm2intr(),ctxdlydm2intl(),
        multidlydmintr(),multidlydmintl(),dmsintr(),dmsintl(),dnmsintr(),
        dnmsintl(),dmcintr(),dmcintl(),dnmcintr(),dnmcintl(), goseqr(), rtgoseqr(), dlygoseqr(), 
        antiseqr(), rtantiseqr(), dlyantiseqr(), dm1seqr(), dm2seqr(), ctxdm1seqr(), ctxdm2seqr(), 
        multidmseqr(), dlydm1seqr(),dlydm2seqr(),ctxdlydm1seqr(), ctxdlydm2seqr(), multidlydmseqr(),
        dmsseqr(), dnmsseqr(), dmcseqr(), dnmcseqr(), goseql(), rtgoseql(), dlygoseql(), antiseql(),
        rtantiseql(), dlyantiseql(), dm1seql(), dm2seql(), ctxdm1seql(), ctxdm2seql(), multidmseql(), dlydm1seql(),
        dlydm2seql(),ctxdlydm1seql(), ctxdlydm2seql(), multidlydmseql(), dmsseql(), dnmsseql(), dmcseql(), dnmcseql()]

env_analysis = MultiEnvs(envs, env_input = True)
schedule = RandomSchedule(len(envs))
env = ScheduleEnvs(envs, schedule=schedule, env_input=True)
dataset = Dataset(env, batch_size=4, seq_len=350)
env = dataset.env
ob_size = env.observation_space.shape[0]
act_size = env.action_space.n


# To draw samples, use neurogym's dataset class:

inputs, labels = dataset()
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
