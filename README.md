# lrl-asr-experiments

To install this GitHub repo, run
`git clone 
# Automatic Speech Recognition (ASR) Tutorial: Setting up your coding environment

## asr-tutorial-fleurs-[nlp-gpu-02].ipynb
To understand the ASR workflow, you can follow this tutorial. We are assuming you are using the nlp-gpu-02 machine (NVIDIA A100 80 GB). Before starting this tutorial, you will need to create a virtual environment for this project so you can download all the required packages without affecting your other projects. We recommend using miniconda (conda) to create a virtual environment. In your terminal (in a folder of your choosing) run these commands:

1. **Install miniconda** ([tutorial](https://educe-ubc.github.io/conda.html))
```
curl -sL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > "Miniconda3.sh"
```
```
bash Miniconda3.sh
```
Follow all prompts, and accept all defaults. Next, close terminal and restart, then run:
```
conda update conda
```
2. **Create Conda Virtual Environment**

Create a virtual environment with Python 3.10
```
conda create -n asr python=3.10
```
Activate this environment
```
conda activate asr
```
Now, every package you install will be installed only in this virtual environment.

3. **Install Jupyter Notebook:**
```
pip install notebook
```

4. **Run Jupyter Notebook**
In the nlp-gpu-02 terminal, in the environment run:
```
jupyter notebook --no-browser --port=8080
```
You should see some output, then 
Now, open a terminal in your *local machine* and run
``


## Google Colab Pro