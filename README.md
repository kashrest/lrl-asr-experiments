# lrl-asr-experiments

To download this GitHub repo, run the following command on your terminal. 
```
git clone https://github.com/kashrest/lrl-asr-experiments.git
```
If you do not have `git`, install using [this tutorial](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

# Automatic Speech Recognition (ASR) Tutorial: Setting up your coding environment
We have provided a tutorial (runnable on Google Colab and nlp-gpu-02) that will help you understand the ASR workflow.

## asr-tutorial-fleurs-[nlp-gpu-02].ipynb
 We are assuming you are using the nlp-gpu-02 machine (NVIDIA A100 80 GB) for this tutorial. Before starting, you will need to create a virtual environment for this project so you can download all the required packages without affecting your other projects. We recommend using miniconda (conda) to create a virtual environment. In your terminal run these commands to set up your environment to be able to run this tutorial:

1. **Install miniconda** ([tutorial](https://educe-ubc.github.io/conda.html))

Download the installer
```
curl -sL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > "Miniconda3.sh"
```
Run the installer
```
bash Miniconda3.sh
```
Follow all prompts, and accept all defaults. Next, close terminal and restart, then run:
```
conda update conda
```
2. **Create Conda Virtual Environment**

Create a virtual environment with Python 3.10 using `conda`
```
conda create -n asr python=3.10
```
Initialize conda

```
source /path/to/.bashrc
```

Activate this environment
```
conda activate asr
```
Now, every package you install will be installed in this virtual environment.

3. **Install Jupyter Notebook**

Run this command in the nlp-gpu-02 terminal to install 
```
pip install notebook
```

4. **Run Jupyter Notebook**

In the nlp-gpu-02 terminal, with the environment activated, and in the directory that contains this GitHub repository, run:
```
jupyter notebook --no-browser --port=8080 (any open port is okay to use 80**)
```
You should see some output, and a url with instructions to copy paste into your browser. **Copy that url**

Now, open a terminal in your *local machine* and run
```
ssh -L 8080:localhost:8080 <USER_ID>@nlp-gpu-02.soe.ucsc.edu
```
The remote nlp-gpu-02 is now connected with your local machine.

Open up a web browser and paste the url from before. You should see your directory contents in nlp-gpu-02. You can now open the tutorial and run the lines!

## asr-tutorial-fleurs-[colab_pro].ipynb
For this tutorial, we recommend using Google Colab Pro, so you have access to an NVIDIA A100 40 GB GPU. 

For saving model training checkpoints, you can specify an output directory on the virtual machine. Do not save on Google Drive by mounting because some checkpoints may take a few GBs. However, your output directory on the virtual machine will disappear once the runtime is disconnected. So, after training, make sure to download the checkpoints to your local machine or wherever you have some space (such as Hugging Face Hub, etc).

**Important note about saving files/data on Google Colab**

Once the runtime disconnects/terminates, all your data will be lost and once you reconnect, you will be connected to a new runtime. The runtime will disconnect if it has remained idle for 90 minutes, or if it has been in use for 12 hours. Make sure you save any model checkpoints/outputs if you would like to investigate further, by downloading the data to your local machine or some other form of storage.
