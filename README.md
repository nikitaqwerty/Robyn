# Robyn: Continuous & Semi-Automated MMM <img src='R/man/figures/logo.png' align="right" height="139px" />
### The Open Source Marketing Mix Model Package from Meta Marketing Science

[![CRAN\_Status\_Badge](https://www.r-pkg.org/badges/version/Robyn)](https://cran.r-project.org/package=Robyn) [![Downloads](https://cranlogs.r-pkg.org/badges/grand-total/Robyn?color=green)](https://cranlogs.r-pkg.org/badges/grand-total/Robyn?color=green) [![Site](https://img.shields.io/badge/site-Robyn-blue.svg)](https://facebookexperimental.github.io/Robyn/) [![Facebook](https://img.shields.io/badge/group-Facebook-blue.svg)](https://www.facebook.com/groups/robynmmm/) [![CodeFactor](https://www.codefactor.io/repository/github/facebookexperimental/robyn/badge)](https://www.codefactor.io/repository/github/facebookexperimental/robyn)
---

## Introduction

  * **What is Robyn?**: Robyn is an experimental, semi-automated and open-sourced Marketing Mix Modeling (MMM) package from Meta Marketing Science. It uses various machine learning techniques (Ridge regression, multi-objective evolutionary algorithm for hyperparameter optimization, time-series decomposition for trend & season, gradient-based optimization for budget allocation, clustering, etc.) to define media channel efficiency and effectivity, explore adstock rates and saturation curves. It's built for granular datasets with many independent variables and therefore especially suitable for digital and direct response advertisers with rich data sources. 
  
  * **Why are we doing this?**: MMM used to be a resource-intensive technique that was only affordable for "big players". As the privacy needs of the measurement landscape evolve, there's a clear trend of increasing demand for modern MMM as a privacy-safe solution. At Meta Marketing Science, our mission is to help all businesses grow by transforming marketing practices grounded in data and science. It's highly aligned with our mission to democratizing MMM and making it accessible for advertisers of all sizes. With Project Robyn, we want to contribute to the measurement landscape, inspire the industry and build a community for exchange and innovation around the future of MMM and Marketing Science in general.
  
Robyn is available in R and Python. For installation and usage guide see below. Please note that the current Python version is a LLM-translated Beta version and might encounter bugs. 
  
## Quick start for R

**1. Installing the package**
  
  * Install Robyn latest R package version:
```{r}
## CRAN VERSION
install.packages("Robyn")

## DEV VERSION
# If you don't have remotes installed yet, first run: install.packages("remotes")
remotes::install_github("facebookexperimental/Robyn/R")
```

  * If it's taking too long to download, you have a slow or unstable internet connection, and have [issues](https://github.com/facebookexperimental/Robyn/issues/309) while installing the package, try setting `options(timeout=400)`.
  
  * Robyn requires the Python library [Nevergrad](https://facebookresearch.github.io/nevergrad/). If encountering Python-related 
  error during installation, please check out the [step-by-step guide](https://github.com/facebookexperimental/Robyn/blob/main/demo/demo.R) as well as this [issue](https://github.com/facebookexperimental/Robyn/issues/189) to get more info.
  
  * For Windows, if you get openssl error, please see instructions
  [here](https://stackoverflow.com/questions/54558389/how-to-solve-this-error-while-installing-python-packages-in-rstudio/54566647) and
  [here](https://dev.to/danilovieira/installing-openssl-on-windows-and-adding-to-path-3mbf) to install and update openssl.

**2. Getting started**

  * Use this [demo.R](https://github.com/facebookexperimental/Robyn/tree/main/demo/demo.R) script as step-by-step guide that is
  intended to cover most common use-cases. Test the package using simulated dataset provided in the package. 
  
  * Visit our [website](https://facebookexperimental.github.io/Robyn/) to explore more details about Project Robyn.
  
  * Join our [public group](https://www.facebook.com/groups/robynmmm/) to exchange with other users and interact with team Robyn.
  
  * Take Meta's [official Robyn blueprint course](https://www.facebookblueprint.com/student/path/253121-marketing-mix-models?utm_source=readme) online 
  
## Quick start for Python (Beta)

The Python version of Robyn is rewritten from Robyn's R package version `3.11.1` to Python using object oriented programming principles and modular architecture for a robust solution. It was developed by utilizing various LLMs and AI workflows like [Llama](https://www.llama.com/). As is common with any AI-based solutions, there may be potential challenges in translating code from one language to another. In this case, we anticipate that there could be some issues in the translation from R to Python. However, we believe in the power of community collaboration and open-source contribution. Therefore, we are opening this project to the community to participate and contribute. Together, we can address and resolve any issues that may arise, enhancing the functionality and efficiency of the Python version of Robyn. We look forward to your contributions and to the continuous improvement of this project.

### 1. Prerequisites

- R must be installed on your machine. Download from the [official R Project website](https://www.r-project.org/)
- The glmnet R package is required

#### Installing glmnet on Windows
```bash
# Open R console (run in Command Prompt/PowerShell)
R
# Then in R console, install glmnet
install.packages("glmnet")
# Exit R console
q()
```

#### Installing glmnet on macOS/Linux
```bash
# Open terminal and run R
R
# Then in R console, install glmnet
install.packages("glmnet")
# Exit R console
q()
```

### 2. Setting up Python Environment

#### Windows
```bash
# Create virtual environment
python -m venv robyn-env

# Activate virtual environment
robyn-env\Scripts\activate
```

#### macOS/Linux
```bash
# Create virtual environment
python3 -m venv robyn-env

# Activate virtual environment
source robyn-env/bin/activate
```

### 3. Install Robyn

Choose one of the following installation methods:

```bash
# Install from PyPI (recommended)
pip3 install robynpy

# OR install development version from source
pip3 install -r requirements.txt
```

### 4. Getting Started

The `python/src/robyn/tutorials` directory contains tutorials for most common scenarios using the included simulated dataset.

There are two ways to run Python Robyn:

#### Option 1: Using tutorial1.ipynb (Recommended)
- Provides an end-to-end flow with a one-click solution
- Ideal for users who prefer minimal setup
- Uses APIs from `python/src/robyn/robyn.py`
- Includes feature engineering, model training, clustering, one-pager generation, and budget allocation
- Configurations can be modified directly in the notebook

#### Option 2: Using tutorial1_src.ipynb (Advanced)
- Offers more flexibility and control over individual modules
- Designed for users who want to customize the workflow
- Calls modules directly with parameters
- Allows skipping specific components (clustering/one-pager plots/budget allocation)
- Requires understanding of underlying logic
  
## Quick start Python wrapper (Robyn API for Python beta)

The Robyn API for Python (beta), first released on Nov.22nd 2023, is a plumber-based solution that requires the installation of the Robyn R pacakge first. It serves as a workaround when the Python native version is not yet available or up-to-date. Please see the usage guide [here](https://github.com/facebookexperimental/Robyn/blob/main/robyn_api/robyn_python_notebook.ipynb).


## License

Meta's Robyn is MIT licensed, as found in the LICENSE file.

- Terms of Use - https://opensource.facebook.com/legal/terms 
- Privacy Policy - https://opensource.facebook.com/legal/privacy
- Defensive Publication - https://www.tdcommons.org/dpubs_series/4627/

## Contact

* gufeng@meta.com, Gufeng Zhou, Marketing Science, Robyn creator
* igorskokan@meta.com, Igor Skokan, Marketing Science Director, open source
