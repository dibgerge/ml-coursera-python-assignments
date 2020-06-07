# [Coursera Machine Learning MOOC by Andrew Ng](https://www.coursera.org/learn/machine-learning) 
# Python Programming Assignments

![](machinelearning.jpg)

This repositry contains the python versions of the programming assignments for the [Machine Learning online class](https://www.coursera.org/learn/machine-learning) taught by Professor Andrew Ng. This is perhaps the most popular introductory online machine learning class. In addition to being popular, it is also one of the best Machine learning classes any interested student can take to get started with machine learning. An unfortunate aspect of this class is that the programming assignments are in MATLAB or OCTAVE, probably because this class was made before python became the go-to language in machine learning.

The Python machine learning ecosystem has grown exponentially in the past few years, and is still gaining momentum. I suspect that many students who want to get started with their machine learning journey would like to start it with Python also. It is for those reasons I have decided to re-write all the programming assignments in Python, so students can get acquainted with its ecosystem from the start of their learning journey. 

These assignments work seamlessly with the class and do not require any of the materials published in the MATLAB assignments. Here are some new and useful features for these sets of assignments: 

- The assignments use [Jupyter Notebook](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html), which provides an intuitive flow easier than the original MATLAB/OCTAVE assignments.
- The original assignment instructions have been completely re-written and the parts which used to reference MATLAB/OCTAVE functionality have been changed to reference its `python` counterpart.
- The re-written instructions are now embedded within the Jupyter Notebook along with the `python` starter code. For each assignment, all work is done solely within the notebook.
- The `python` assignments can be submitted for grading. They were tested to work perfectly well with the original Coursera grader that is currently used to grade the MATLAB/OCTAVE versions of the assignments. 
- After each part of a given assignment, the Jupyter Notebook contains a cell which prompts the user for submitting the current part of the assignment for grading.  

 ## Online workspace
 
 You can work on the assignments in an online workspace called [Deepnote](https://www.deepnote.com/). This allows you to play around with the code and access the assignments from your browser. [<img height="22"  src="https://beta.deepnote.com/buttons/launch-in-deepnote.svg">](https://beta.deepnote.com/launch?template=data-science&url=https%3A%2F%2Fgithub.com%2Fdibgerge%2Fml-coursera-python-assignments)

## Downloading the Assignments

To get started, you can start by either downloading a zip file of these assignments by clicking on the `Clone or download` button. If you have `git` installed on your system, you can clone this repository using : 

    clone https://github.com/dibgerge/ml-coursera-python-assignments.git
    
Each assignment is contained in a separate folder. For example, assignment 1 is contained within the folder `Exercise1`. Each folder contains two files: 
 - The assignment `jupyter` notebook, which has a `.ipynb` extension. All the code which you need to write will be written within this notebook.
 - A python module `utils.py` which contains some helper functions needed for the assignment. Functions within the `utils` module are called from the python notebook. You do not need to modify or add any code to this file.

## Requirements 

These assignments has been tested and developed using the following libraries: 

    - python==3.6.4
    - numpy==1.13.3
    - scipy==1.0.0
    - matplotlib==2.1.2
    - jupyter==1.0.0
    - jupyter-client==5.0.1
    
We recommend using at least these versions of the required libraries or later. Python 2 is not supported. 
    
## Python Installation

We highly recommend using anaconda for installing python. [Click here](https://www.anaconda.com/download/) to go to Anaconda's download page. Make sure to download Python 3.6 version.
If you are on a windows machine:
 - Open the executable after download is complete and follow instructions.
 - Once installation is complete, open `Anaconda prompt` from the start menu. This will open a terminal with python enabled.
 
 If you are on a linux machine: 
 
 - Open a terminal and navigate to the directory where Anaconda was downloaded. 
 - Change the permission to the downloaded file so that it can be executed. So if the downloaded file name is `Anaconda3-5.1.0-Linux-x86_64.sh`, then use the following command:
 
      `chmod a+x Anaconda3-5.1.0-Linux-x86_64.sh`
 
 - Now, run the installation script using `./Anaconda3-5.1.0-Linux-x86_64.sh`, and follow installation instructions in the terminal.
 
 
Once you have installed python, create a new python environment will all the requirements using the following command: 

    conda env create -f environment.yml
 
After the new environment is setup, activate it using (windows)

    activate machine_learning
   
or if you are on a linux machine

    source activate machine_learning 

Now we have our python environment all set up, we can start working on the assignments. To do so, navigate to the directory where the assignments were installed, and launch the jupyter notebook from the terminal using the command

    jupyter notebook

This should automatically open a tab in the default browser. To start with assignment 1, open the notebook `./Exercise1/exercise1.ipynb`. 

## Python Tutorials

If you are new to python and to `jupyter` notebooks, no worries! There is a plethora of tutorials and documentation to get you started. Here are a few links which might be of help:

- [Python Programming](https://pythonprogramming.net/introduction-to-python-programming/): A turorial with videos about the basics of python. 

- [Numpy and matplotlib tutorial](http://cs231n.github.io/python-numpy-tutorial/): We will be using numpy extensively for matrix and vector operations. This is great tutorial to get you started with using numpy and matplotlib for plotting.

- [Jupyter notebook](https://medium.com/codingthesmartway-com-blog/getting-started-with-jupyter-notebook-for-python-4e7082bd5d46): Getting started with the jupyter notebook. 

- [Python introduction based on the class's MATLAB tutorial](https://github.com/mstampfer/Coursera-Stanford-ML-Python/blob/master/Coursera%20Stanford%20ML%20Python%20wiki.ipynb): This is the equivalent of class's MATLAB tutorial, in python.


## Caveats and tips

- In many of the exercises, the regularization parameter $\lambda$ is denoted as the variable name `lambda_`, notice the underscore at the end of the name. This is because `lambda` is a reserved python keyword, and should never be used as a variable name.

-  In `numpy`, the function `dot` is used to perform matrix multiplication. The operation '*' only does element-by-element multiplication (unlike MATLAB). If you are using python version 3.5+, the operator '@' is the new matrix multiplication, and it is equivalent to the `dot` function.

## Acknowledgements

- I would like to thank professor Andrew Ng and the crew of the Stanford Machine Learning class on Coursera for such an awesome class. 

- Some of the material used, especially the code for submitting assignments for grading is based on [`mstampfer`'s](https://github.com/mstampfer/Coursera-Stanford-ML-Python) python implementation of the assignments. 
