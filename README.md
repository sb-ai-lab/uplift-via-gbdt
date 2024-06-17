## Experimental code for "Uplift modeling via Gradient Boosting paper"

This repository is created to reproduce the experiments of the paper. It contains the scripts to run the experiments and Jupyter notebooks to launch and analyze the results. 


### Requirements

The proposed method is based on the frameworks that requires Nvidia GPU. We used 24 Cores CPU, 512 GB RAM, and 2xTesla V100 to obtain the experimental results, but it is possible to execute with less hardware.  

To setup the environment you need to have `conda` installed. After that, please execute the following to install all the dependencies (it is assumed that you run everything from the repository root directory): 

```bash
bash ./setup_env.sh

```

After that `rapids-24.04` conda env will be created in the repository root dir. You need to run all the experiments under this env. You can activate it by executing `conda activate -p ./rapids-24.04`

### Data

To download and preprocess all the data please execute

```bash
python datasets/get_data.py

```

### Synthetic experiment

Please run `Synthetic.ipynb` notebook to obtain the results provided in **Section 4.1**

### Main experiment

The main experiment provided in **Section 4.2** is separated on the two parts:

* GPU based experiments contain the proposed method together with Neural Network baselines. You can execute it by running `RunGPUTasks.ipynb`. Please, adjust the `Params` cell according to your hardware. The proposed results were obtained with the listed above hardware

* CPU based experiments Meta Learners based algorithms and CausalForest. You can execute it by running `RunCPUTasks.ipynb`. Please, adjust the `Params` cell according to your hardware. The proposed results were obtained with the listed above hardware

### Analyze the results

After evaluations are finalized, you can obtain the contents by running `ResultsMain.ipynb` notebook
