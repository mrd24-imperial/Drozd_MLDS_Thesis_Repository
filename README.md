This repository contains the code I have used to generate the results for my thesis. The environment can be set up using the following steps. I have set the environment up in a conda environment. In order to reproduce this the user may want to change the environment name or directory path.

First an environment is set up with python version 3.10.
```
conda create -n test_environment python=3.10
```

After activating the environment, we assure that pip is at an appropriate version not to cause an issue with the requirements.

```
conda activate test_environment
python.exe -m pip install pip==23.3.2
```

Then the requirements may be installed using the file in this repository.
```
pip install -r "C:/Users/mrdxy/Documents/MLDS/Final Project/requirements.txt"
```

Finally the github repository of the benchmarking environment cited in the thesis (Mai et al., 2023) is downloaded.
```
git clone https://github.com/huda-lab/RL-Epidemic-Benchmark
```

This setup should allow the code uploaded to this repository to be run. Before running it, the user should however take care to appropriately adapt the relevant directory paths in the script and set the model for which the experiment should be run.
