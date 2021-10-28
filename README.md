# SolarNet
A repository hosting code for a series of SolarNet papers.

As described in:

**Convolutional Neural Networks for Intra-hour Solar Forecasting Based on Sky Image Sequences**, Cong Feng, Jie Zhang, Wenqi Zhang, and Bri-Mathias Hodge. (under review)


## How to use
Users can implement the SolarNet by running the SolarNet_SolarForecasting.py script. High performance computing resource with GPUs is strongly suggested.

### Dataset
The example dataset in this repository contains processed and partitioned training, validation, and testing datasets for 10-min-ahead solar forecasting.

### Environment
```
module load intel/18.0.2 python3/3.7.0 cuda/10.0 cudnn/7.6.2 impi/18.0.2 git/2.24.1 autotools/1.2 cmake/3.16.1 xalt/2.9.6 phdf5/1.10.4 
```

### Python library
```
import pandas as pd
import os, sys, pickle
from keras import models
from keras import layers
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from keras.utils import multi_gpu_model
import tensorflow as tf
import subprocess, argparse
```


## Publications
**If you use this package in your research, please cite our publications**:

Convolutional Neural Networks for Intra-hour Solar Forecasting Based on Sky Image Sequences, Cong Feng, Jie Zhang, Wenqi Zhang, and Bri-Mathias Hodge. (under review)


**Collaborations are always welcome if more help is needed.**
## License
MIT License, Copyright (c) 2021 Cong Feng

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


## Contact

Cong Feng

joey.fueng@outlook.com


