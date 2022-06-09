# Bayesian Computing & Hierarchical Modeling Labs

#### led by [Prof. Murali Haran](http://personal.psu.edu/muh10/) and [Prof. Joel Leja](http://www.personal.psu.edu/jql6565/)
#### Astroinformatics Summer School 2022 
#### Organized by [Penn State Center for Astrostatistics](https://sites.psu.edu/astrostatistics/)

---

This repository contains several computational notebooks: 
- monte_carlo.jl ([Pluto notebook](https://astroinformatics.github.io/BayesianComputing/monte_carlo.html)):  Compares methods for performing numerical integration
- app_hbm_galaxy_evolution.ipynb ([Jupyter notebook](https://github.com/Astroinformatics/BayesianComputing/blob/main/app_hbm_galaxy_evolution.ipynb)):  Applies hierarchical Bayesian modeling to make inferences about galaxy evolution.  
- ppl_intro.jl ([Pluto notebook](https://astroinformatics.github.io/BayesianComputing/ppl_intro.jl.html)):   Introduces the use of probabilistic programming languages (PPLs) using [Turing.jl](https://turing.ml/stable/)
- ppl_hbm.jl ([Pluto notebook](https://astroinformatics.github.io/BayesianComputing/ppl_hbm.jl.html)):  Demonstrates performing inference with hierarchical Bayesian models using the Turing PPL

monte_carlo.jl is mostly independent of the other notebooks in this lesson and could reasonably be done first or last.
We recommend that students complete ppl_intro.jl before ppl_hbm.jl.  
app_hbm_galaxy_evolution.ipynb and ppl_hbm.jl could be worked in either order.  

Files ending in .jl are Pluto notebooks written in Julia and files ending in .ipynb are Jupyter notebooks written in Python.
Labs do not assume familiarity with either language.  While it can be useful to "read" selected portions of the code, the lab tutorials aim to emphasize understanding how algorithms work, while minimizing need to pay attention to a language's syntax.

---

## Running Labs
Instructions will be provided for students to run labs on AWS severs during the summer school.  Below are instruction for running them outside of the summer school.

### Running Pluto notebooks on your local computer
Summer School participants will be provided instructions for accessing a Pluto server.  Others may install Julia and Pluto on their local computer with the following steps:
1.  Download and install current version of Julia from [julialang.org](https://julialang.org/downloads/).
2.  Run julia
3.  From the Julia REPL (command line), type
```julia
julia> using Pkg
julia> Pkg.add("Pluto")
```
(Steps 1 & 3 only need to be done once per computer.)

4.  Start Pluto
```julia
julia> using Pluto
julia> Pluto.run()
```
5.  Open the Pluto notebook for your lab

### Running Jupter/Python notebooks 
Summer School participants will be provided instructions for accessing JupyterLab server.  
Others may install Python 3 and Jupyter (or JupyterLab) on their local computer or use [Google Colab](https://colab.research.google.com/) to open the Jupyter notebooks.

---
## Additional Links
- [GitHub respository](https://github.com/Astroinformatics/SummerSchool2022) for all of Astroinformatics Summer school
- Astroinformatics Summer school [Website & registration](https://sites.psu.edu/astrostatistics/astroinfo-su22/)

## Contributing
We welcome people filing issues and/or pull requests to improve these labs for future summer schools.
