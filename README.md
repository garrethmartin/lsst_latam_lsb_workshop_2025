Clone the repository and install dependencies:

```bash
git clone https://github.com/garrethmartin/python_fits_viewer.git
cd python_fits_viewer
conda env create -f environment.yml
conda activate fits_viewer
python -m ipykernel install --user --name=fits_viewer --display-name "Python (fits_viewer)"
jupyter notebook
```
