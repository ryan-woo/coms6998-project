# coms6998-project
Computation and the Brain Project

# Installation
Clone this repository and run

```
cd coms6998-project
pip install .
```

You may optionally install into a virtual environment by cloning the repository, creating the virtual environment, 
and then installing.
```
cd coms6998-project
python3 -m venv .venv  # Create a virtual environment named .venv
source .venv/bin/activate
pip install .
```

# Running

To run a scoring script, simply run
```
cd candbproj
python gpt2_trained.py
```

In order to run the scoring function `n` times, each with a different random seed, do
```
python gpt2_trained.py -n <n>
```
