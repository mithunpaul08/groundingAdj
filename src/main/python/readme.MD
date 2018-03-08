# Code for the Neural Network part in : Grounding Gradable Adjectives through Crowdsourcing. Submitted to  LREC 2018.


## Reproducibility

This repository contains the files necessary to reproduce thissubmission.

Rather than providing seed values and requiring the model to be
retrained, the repository also contains relevant scripts and the PyTorch
model trained as part of the submission.

The submission can easily be reproduced by loading this model using the
`initializer.py` script and following instructions on screen.

Alternatively the initializer.py` script can be used to train the model with different architectures.

## Getting started

To get started, simply download the files in this repository to a local
directory.

### Prerequisites
The model was developed, trained and tested using the
following packages on a Mac OSX:10.12.6 Sierra

A detailed list of packages are given at the end of this document.
Following the list of instructions in the `Steps to Install` section should get you all these packages installed.

```
Conda==4.3.30
Python==3.6.3
PyTorch=0.1.12
NumPy==1.13.1
Pandas=0.21.0
Pip=9.0.1
Scipy=0.19.1
Tqdm=4.19.4



```




Please note that compatibility of the saved model with newer versions
of `PyTorch` has not been checked. Accordingly, please use the
`PyTorch` version listed above.
### Installing

Other than ensuring the dependencies are in place, no separate
installation is required.

Simply execute the `initializer.py` file once the repository has been
saved locally.

## Steps to Install


conda create --name grounding python==3.6.3 scipy==0.19.1 pandas==0.21.0
source activate grounding
pip install sklearn
conda install pytorch torchvision -c pytorch

pip install tqdm
pip install torchwordemb


## Steps to run
If you are not already in the home folder:
cd src/main/python
python initializer.py



#  These are the complete list of packages you need to run this code:
#
python 3+

ca-certificates           2017.08.26           h1d4fec5_0
certifi                   2017.7.27.1      py36h8b7b77e_0
cffi                      1.10.0           py36had8d393_1
cudatoolkit               8.0                           3
cudnn                     6.0.21                cuda8.0_0
cycler                    0.10.0                    <pip>
intel-openmp              2018.0.0             h15fc484_7
libedit                   3.1                  heed3624_0
libffi                    3.2.1                hd88cf55_4
libgcc                    7.2.0                h69d50b8_2
libgcc-ng                 7.2.0                h7cc24e2_2
libgfortran               3.0.0                         1
libstdcxx-ng              7.2.0                h7a57d05_2
matplotlib                2.1.0                     <pip>
mkl                       2017.0.4             h4c4d0af_0
nccl                      1.3.4                 cuda8.0_1
ncurses                   6.0                  h9df7e31_2
numpy                     1.13.1                   py36_0
openssl                   1.0.2m               h26d622b_1
pandas                    0.21.0           py36h78bd809_1
pip                       9.0.1            py36h6c6f9ce_4
pycparser                 2.18             py36hf9f622e_1
pyparsing                 2.2.0                     <pip>
python                    3.6.3                h1284df2_4
python-dateutil           2.6.1            py36h88d3b88_1
pytorch                   0.1.12          py36cuda8.0cudnn6.0_1
pytz                      2017.3           py36h63b9c63_0
readline                  7.0                  ha6073c6_4
scipy                     0.19.1              np113py36_0
setuptools                36.5.0           py36he42e2e1_0
six                       1.11.0           py36h372c433_1
sqlite                    3.20.1               hb898158_2
tk                        8.6.7                hc745277_3
tqdm                      4.19.4           py36ha5a5176_0
wheel                     0.29.0           py36he7f4e38_1
xz                        5.2.3                h55aa19d_2
zlib                      1.2.11               ha838bed_2
