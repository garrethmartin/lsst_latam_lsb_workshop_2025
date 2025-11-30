### Install Miniconda

If you do not already have Conda, install Miniconda by following the
official instructions for your platform:

https://www.anaconda.com/docs/getting-started/miniconda/install#windows-command-prompt

After installation, open a new terminal to ensure the shell
initialisation takes effect.

### Where to run Conda

Conda must be run in a terminal that has been initialised for it.

**Windows** - Use *Anaconda Prompt* (installed with Miniconda).

**macOS** - Use Terminal.
- If `conda` is not recognised: `bash   conda init zsh` Then start a new
terminal session.

**Linux** - Any shell works once initialised.
- If needed: `bash   conda init bash` Then open a new shell.

------------------------------------------------------------------------

### Clone the repository and install dependencies

``` bash
git clone https://github.com/garrethmartin/lsst_latam_lsb_workshop_2025.git
cd lsst_latam_lsb_workshop_2025
```
(or just download and from https://github.com/garrethmartin/lsst_latam_lsb_workshop_2025 and unzip)
```
conda env create -f environment_lsst_lsb.yml
conda activate lsst_lsb

python -m ipykernel install --user --name=lsst_lsb --display-name "Python (lsst_lsb)"

jupyter notebook
```

In the menu bar, make sure the kernel is set to the correct environment `(lsst_lsb)`.