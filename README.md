# ARC (Adaptive and Reliable Classification)

This package provides some statistical wrappers for machine learning classification tools in order to construct prediction sets for the label of a new test point with valid marginal coverage.

Accompanying paper:

    "Classification with Valid and Adaptive Coverage", Y. Romano, M. Sesia, E. Candès, 2020.
    

## Contents

 - `arc/` Python package implementing our methods and some alternative benchmarks.
 - `third_party/` Third-party Python packages imported by our package.
 - `examples/` Jupyter notebooks with introductory usage examples.
 - `experiments_sim_data` Code for the experiments with simulated data discussed in the accompanying paper.
 - `experiments_real_data` Code for the experiments with real data discussed in the accompanying paper.
  
## Third-party packages

This package builds upon the following non-standard Python packages provided in the "third-party" directory:

 - `nonconformist` https://github.com/donlnz/nonconformist
 - `cqr` https://github.com/yromano/cqr
 - `cqr-comparison` https://github.com/msesia/cqr-comparison
    
## Prerequisites

Prerequisites for the `arc` package:
 - numpy
 - scipy
 - sklearn
 - skgarden
 - torch
 - tqdm
 
Additional prerequisites for example notebooks:
 - pandas
 - matplotlib
 - seaborn

 

## Installation

The development version is available from GitHub:

    git clone https://github.com/msesia/arc.git

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.
