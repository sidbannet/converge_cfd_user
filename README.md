# Introduction
---
*   Example notebook showing how to analyze and present your [CONVERGE CFD](https://convergecfd.com/) data
*   How to combine `markdown` and `code` to write a live presentable report

## About author

Dr. Banerjee is currently leading CFD modeling at [Mainspring Energy Inc.](https://www.mainspringenergy.com) a clean energy startup based in the San Francisco Bay Area.

Dr. Banerjee has 12 years of experience focused in the areas of computational combustion, propulsion, and clean energy technology. After earning a Ph.D. from the [University of Wisconsin - Madison](https://www.wisc.edu) in 2011, Dr. Banerjee worked in the Corporate R&D division of [Cummins Inc.](https://www.cummins.com) for several years. In 2016, he received Director's award at [Oak Ridge Leadership Computing Facility](https://www.olcf.ornl.gov/) along with DOE funding to work on novel combustion engine technology. Throughout his career, he collaborated with several National Labs and Convergent Science engineers to bring cutting-edge clean energy technology to market using high-performance computer simulation models. He published over [25 technical papers](https://scholar.google.com/citations?user=eTX1dWAAAAAJ&hl=en), authored several patents, and served in several organizing committees at _ASME's IC Engine Division_ over the years.

---
| [Github](https://github.com/sidbannet) | [LinkedIn](https://www.linkedin.com/in/sidban/) | [Email](mailto:sidban@uwalumni.com) |

## Why open-source CONVERGE user community?

* Increasingly *CFD* is used in conjunction with *data-science* to use as powerful predictive tool for analysis-led development and accelerate R&D. Meta-models and optimization methods like **Response Surface**, **Optimization on Manifolds**, **Genetic algorithim** optimization and **Decision Tree / Random Forest** are used by large number of CONVERGE users these days.
* [Python](https://www.python.org/) is one of the most popular scientific computing languages to perform data-science / analysis
* Web-based version control platforms like [github](https://github.com/) provides an oppertunity to collaborate with CONVERGE developers and users in open-source forum and learn from each other and to advance CFD data-analysis further.

### Few more reasons

* Most of CFD reports are static slides / pages. Using Notebook you can create a CFD result dashboard and integrate it with reports / presentations.
> Example: [COVID dashboard](https://gist.github.com/sidbannet/5f344203c1811696a0c8c51500323052) (report orginally published in November 2020) using Jupyter Notebook is still up-to-date with a single click.
* Code lives with your report.
> Embedding code in your presentation is powerful. You can **interact** with the report and draw insights faster.
* Easier to collaborate.
> * **Peers**: You can bring in test data within your CFD analysis easily and compare / validate your model faster and better.
  * **CONVERGE developers**: Easier to share your data and analysis with CONVERGE support

* Live analysis and HPC job management
  * **HPC cloud platform**: You can have your code live alongside the simulation and have live data-anlysis while simulation is running. 
  * You can **manage** your HPC jobs based on pre-set criteria (like emissions, efficiency etc.).

# Loading CONVERGE results

* Import nessesary packages.
* Instantiate an [example high-performance CFD case](https://convergecfd.com/benefits/high-performance-computing) for post processing.
* Load time series data from the example CFD result.

## Code to import packages
> `from post.process import SimpleCase as CfdCase` will give `CfdCase` class in your analysis

## Instantiate an example `cfd_obj` case from a given _case directory_

> `cfd_obj = CfdCase(proj_dir, proj_name)` will instantiate `cfd_obj` with all nessesary properties and methods

## Load CFD data
> `cfd_obj.load_cfd_data()` will loadd all CFD timeseries data in `cfd_obj`

## Get `echo` file metadata
> `cfd_obj._get_echo_file(file_name='engine.echo', eng_info='rpm')` will give RPM information from `engine.echo` file

## Plot pressure trace
> `(cfd_obj.thermo.all.Pressure * 10).plot(title='Pressure trace')` will give you pressure trace

# Quickstart

If you are a [vscode](https://code.visualstudio.com/) user, this project is pre-configured with full dev environment using docker. Make sure to have [docker](https://docs.docker.com/engine/install/) and [remote-containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) installed. Open the project and allow it to open inside container when the pop-up shows, once fully loaded, start playing with sample jupyter notebook in root folder.

# Remarks

**FAQs**
1. How to load data when there are multiple restarts in a particular project?
> `pandas` append takes care of it. You don't need to do anything special.
2. Can I add my own methods / functions to do further analysis?
> Yes, use [python's inheritance](https://www.w3schools.com/python/python_inheritance.asp) to built your own methods / functions.
3. How can I get started with this?
> * Install `git` if you don't have already. Here is a helpful [link](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) to install git
> * Clone the repository using `git clone https://github.com/sidbannet/converge_cfd_user.git`
> * Install popular open-source data science toolkit like [conda](https://www.anaconda.com/products/individual-d)
> * Use your choice of *Integrated Development Environment* like [Pycharm](https://www.jetbrains.com/pycharm/) or [Spyder](https://www.spyder-ide.org/) or [Visual Studio](https://visualstudio.microsoft.com/) to build your version of this code.
4. How can I contribute to this "open-source" code repository?
> Use **Github**'s feature to 
> * Conribute (Helps this community grow)
> * Fork (have your own version of this code repository)
> * Write to [me](mailto:sidban@uwalumni.com) if you have any questions
