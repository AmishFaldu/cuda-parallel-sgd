# Parallelize SGD in CUDA

## Setup

1. Make sure Python is installed in your system
2. Create a virtual environment using `python3 -m venv .venv`
3. Activate your virtual environemtn using `source .venv/bin/activate`
4. Install all the packages by running `pip install -r requirements.txt`
5. Run following command to download the dataset `python setup.py`

## Compile the CUDA program

1. Make sure the CUDA and other Nvidia GPU tools are installed
2. If you're running GPU in Wulver, make sure to load the module `module load CUDA/12.4.0`
3. To compile the code, use `nvcc -lcublas -Iinclude -o main main.cu`
4. To run the code, you can either use `./main` or create a job file
