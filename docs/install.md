# Environment

It is recommended to create a new virtual environment using the provided environment.yml file.

## 1. Create environment from environment.yml

```bash
# Using conda
conda env create -f environment.yml

# Or using mamba (faster)
mamba env create -f environment.yml

# Activate the environment
conda activate mamba
```

This will install all required packages (including pip packages) as specified in environment.yml.

## 2. (Optional) Updated environment
If some packages need to be updated for your specific configuration, please note that this may cause compatibility issues and the project might not work as expected.
```
