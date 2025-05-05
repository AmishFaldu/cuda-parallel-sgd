# Parallelize SGD in CUDA

## Setup

1. Make sure Python is installed in your system
2. Create a virtual environment using `python3 -m venv .venv`
3. Activate your virtual environemtn using `source .venv/bin/activate`
4. Install all the packages by running `pip install -r requirements.txt`
5. Run following command to download the dataset `python setup.py`

## Compile the CUDA program

1. Make sure the CUDA and other Nvidia GPU tools are installed
2. Go to the proejct directory using `cd project`
3. If you're running GPU in Wulver, make sure to load the module `module load CUDA/12.4.0`
4. To compile the code, use `nvcc -lcublas -Iinclude -o ./code/cuda/main ./coda/cuda/main.cu`

## Run the benchmarks

1. Make sure you're in `project` directory.
2. Submit jobs to worker nodes using sbatch command
   1. `sbatch jobs/cuda-job.submit`
   2. `sbatch jobs/python-single-core-job.submit`
3. Inspect the logs in the `.out` files to get the runtimes of the algorithms

## Data preparation

There is a data preprocessing notebook which generates processed version of this [train data](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data). If you want to run the notebook and reproduce the training data follow the below steps.

1. Download the `train.csv` from the dataset link.
2. Create a directory in the `project` direcotry named `data`.
3. Place the `train.csv` in the `data` directory.
4. Run the `data-preprocessing.ipynb` notebook to generate the final traning data.
