README:
=======

The current folder contains the code used for simulations in the paper:

Stochastic Variance Reduction Methods for Saddle-point problems. 
Authors: P.Balamurugan and Francis Bach.
To appear in Advances in Neural Information Processing Systems, 2016. 
Preprint available at: https://hal.archives-ouvertes.fr/hal-01319293/document

Please note that the current version of code contains toy data. 
If you need to use other data sets, please make appropriate changes to the code.
  
How to Use:
===========
Please run the function simulations_sagsaddle_nips(ftype, gtype) from MATLAB command window. 

ARGUMENTS:
ftype: Regularizer options for function f(x) 
       2 for L1-Norm
       5 for cluster norm

gtype: Loss options for function g(y)
       1 for Squared Hinge-loss
       6 for AUC Loss


Example :
=========

Please open the MATLAB command prompt and change to current directory, and type: 

simulations_sagsaddle_nips(2,1)

and then press enter. 

Contact:
========

If you have any doubts, bug reports or questions, please mail: balamurugan.palaniappan@inria.fr 

