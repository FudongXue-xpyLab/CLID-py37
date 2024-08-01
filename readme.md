## Introduction

Our CLID method is a U-net based  denoising neural network, which combine self-supervised and supervised network using clear images as the ground truth. 


##  Installation
This code is tested with python 3.7.  We're using [Anaconda](https://www.anaconda.com/download/) to manage the Python environment.  Please select the correct Pytorch version that matches your CUDA version from [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/). 

    ```bash
    conda create -n CLID python=3.7
    conda activate CLID
    conda install torch==1.8.0+cu111 torchaudio=0.8.0 torchvision==0.9.0+cu111 cuda111 -c pytorch
    conda install cudatoolkit=11.3
    pip install matplotlib
    pip install scikit-image
    pip install pyqt5
    pip install opencv-python
    ```

## Test models

  ```bash
    #test
    python CLID_test.py
    
    #key parameters:
    --pretrain_dir
      the pre-trained model pool
    --data_root
      dir for the noisy images
    --test_data
      filename of the noisy image
    --multi_cells (bool)
      whether search the same key names or not
    --offset
       offset of sCMOS
    --scale
      enlarge times
```

## Open source CLID
This software and corresponding methods can only be used for **non-commercial** use, and they are under Open Data Commons Open Database License v1.0.