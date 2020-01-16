# NonlinearRoutine
Iterative routine to implement nonlinear material properties in a linear finite element solver 

### Introduction
The scripts are written for usage in the linear finite element (FE) solver ParOSol. They describe how to apply nonlinear material properties in this linear solver by running iteratively in a loop over ParOSol. Since ParOSol is a voxel-based FE solver, the code only works for voxel-based FE models. ParOSol is an open-source software, which can be downloaded from the following link: https://bitbucket.org/cflaig/parosol/src/default/. 

### Publication
For more information about the use of this code and the example data, we refer to the following journal article:

A. Sas, N. Ohs, E. Tanck, G.H. van Lenthe, Nonlinear voxel-based finite element model for strength assessment of  healthy and metastatic proximal femora, Bone reports 2020, xx.

If you decide to use this code for your research, we kindly request that you cite this paper when publishing. 

### User information
#### 1.	Code is not directly implementable
We should remark that the code in not directly implementable. One part of the code (class ‘ParOSolver'), which enables to call ParOSol from within the Python environment, could not be made open-source. The reason for this is that the code is part of a larger framework, which is a multi-project collaboration. Since not all developers agreed on releasing the code at this stage, we were only able to release our part on the nonlinear routine. The complete framework might become available in the future, but for now the user should self-implement the missing part based on the documentation of ParOSol. Alternatively, the user could simply use the provided code as a basis to develop a similar methodology to implement nonlinear material properties in any linear FE solver.   

#### 2.	Examples
The code is applied on two example datasets in the folder ‘data’: an FE model of a simple beam (Beam_example.h5) and an FE model of a femur (Femur_example.h5). A description of these two models can be found in the publication. The FE simulations for both examples can be run through the respective scripts ‘Load_beam.py’ and ‘Load_femur.py’. 

#### 3.	Material model
The implemented nonlinear material model in the code is based on the following publication:

J.H. Keyak, T.S. Kaneko, J. Tehranzadeh, H.B. Skinner, Predicting proximal femoral strength using structural engineering models, Clin. Orthop. Relat. Res. 437 (2005) 219–228. 

We refer the reader to this publication for more information on the material model.

#### 4.	Additional information within scripts
Additional information regarding the use of the code can be found at the start of the scripts ‘Load_beam.py’ and ‘Load_femur.py’.

#### 5. Python version
The code is compatible with python version 3.5 or higher.
