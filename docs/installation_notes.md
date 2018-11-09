# Installation Notes

---

## Linux installation

1. Install via pip:
    `pip install git+https://bitbucket.org/mrayson/iwaves.git@master`

---

## Windows installation

(assuming you have no python distribution installed)

1. Download and install miniconda3 from here: https://conda.io/miniconda.html (install as local user to avoid requiring admin priveleges)

2. Open an anaconda command prompt from the windows start menu
3. From the anaconda command prompt:

```
conda create --name iwaves
conda activate iwaves
conda install -y pip git
conda install -c conda-forge -y gsw
pip install git+https://bitbucket.org/mrayson/iwaves.git@master
```

4. Test installation from the anaconda prompt
	`python -c "from iwaves.utils import isw;print('done')"`

### Notes

 - `gsw` package requires a c++ compiler to install via pip. This is a pain to install via windows. Visual Studio 2017 did not work for me...
 - Download visual c++ tools https://visualstudio.microsoft.com/visual-cpp-build-tools/

