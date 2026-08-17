This repository contains the code I have used to generate the results for my thesis. The environment can be set up using the following steps. I have set the environment up in a conda environment. When reproducing this the user may want to change the environment name or directory path in the following steps.

First an environment is set up with python version 3.10.
```
conda create -n test_environment python=3.10
```

After activating the environment, we assure that pip is at an appropriate version to not cause an issue with incompatibility when installing the requirements.

```
conda activate test_environment
python.exe -m pip install pip==23.3.2
```

Then the requirements may be installed using the file in this repository.
```
pip install -r requirements.txt
```

Finally the github repository of the benchmarking environment cited in my thesis (Mai et al., 2023) is downloaded.
```
git clone https://github.com/huda-lab/RL-Epidemic-Benchmark
```

This setup should allow the code uploaded to this repository to be run. Before running it, the user should however take care to appropriately adapt the relevant directory paths in the script, the model for which the experiment should be run and the number of workers to be used on their machine for parallelization. They may also skim the initial part of the code in order to set any variables they may want to differ compared to my experiments

In order to generate the data, the user may first run the script `run_epidemiology_parallel.py` in order to run the QD algorithm and generate the archive of results. This may take quite some time, on my machine with the setup for my thesis this took 2-3 hours for the SIR_A and SIRV_A models and about 20 hours for the COVID_A (C15 in the paper and my thesis) model. Because I was manually reconstructing the costs in this script rather than extracting them directly from the environment, which turned out to result in noticeable differences due to various reasons, the next step is to run the script `rerun_archive_elites.py` in order to extract the correct costs for the elites. I set it up this way instead of updating the original script because I noticed the relevance of this issue only late in my writing process and could not run the full QD optimization again due to time constraints, the second script luckily finishes much faster. After running both scripts for one model the ipython notebook `Plot_Generation_Thesis.ipynb` may be used to generate the plots found in my thesis.

When using this code you may cite my Master's thesis in the following way:

```
@mastersthesis{Drozd2026,
  title        = {Exploring Epidemic Control Policy Spaces with MAP-Elites Quality-Diversity Optimization},
  author       = {Maximilian Roy Drozd},
  year         = 2026,
  month        = {August},
  address      = {Zurich, Switzerland},
  school       = {Imperial College London},
  type         = {Master's thesis}
}
```
