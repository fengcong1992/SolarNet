# SolarNet
A repository hosting code for a series of SolarNet papers, as described in:

**SolarNet2.0**: Convolutional Neural Networks for Intra-hour Solar Forecasting Based on Sky Image Sequences

**SolarNet1.1**: A sky image-based deep convolutional neural network for intra-hour solar forecasting

**SolarNet1.0**: A sky image-based deep convolutional neural network for intra-hour solar forecasting


## How to use
Users can implement the SolarNet by running the GPU-enabled ```SolarNet_SolarForecasting.py``` script. High performance computing resource with GPUs is strongly suggested for implementations with large datasets. An CPU-enabled implementation could be found in a Jupyter notebook ```SolarNet_Implementation_CPU.ipynb```.

### Dataset
The datasets in this repository contains (1) ```Data_3Dyas.pkl```: a file with 3 days of aligned data (1 day of data for each of the training/validation/testing dataset), (2) ```SkyImage_3Days.zip```: 3 days of processed sky images, and (3) ```Data_6Years.pkl```: a file with 6 years of aligned data. Examples in this repository could be replicated with the Data_3Days.pkl and SkyImage_3Days.zip. To implement results with full datasets, sky images and numerical weather measurements should be downloaded and processed first.

### Environment
```
module load intel/18.0.2 python3/3.7.0 cuda/10.0 cudnn/7.6.2 impi/18.0.2 git/2.24.1 autotools/1.2 cmake/3.16.1 xalt/2.9.6 phdf5/1.10.4 
```

### Python library
```
tensorflow           1.15.0
tensorflow-estimator 1.15.1
tensorflow-gpu       1.15.0
Keras                2.3.1
Keras-Applications   1.0.8
Keras-Preprocessing  1.1.2
numpy                1.20.3
pandas               0.24.1
```


## Publications
**If you use this resource in your research, please cite our publications**:

Feng, C., Zhang, J., Zhang, W., Hodge, B.-M., 2021. Convolutional Neural Networks for Intra-hour Solar Forecasting Based on Sky Image Sequences. (under review)

Feng, C. and Zhang, J., 2020. SolarNet: A sky image-based deep convolutional neural network for intra-hour solar forecasting. Solar Energy, 204, pp.71-78.

Feng, C. and Zhang, J., 2020, February. SolarNet: A Deep Convolutional Neural Network for Solar Forecasting via Sky Images. In 2020 IEEE Power & Energy Society Innovative Smart Grid Technologies Conference (ISGT) (pp. 1-5). IEEE.

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


