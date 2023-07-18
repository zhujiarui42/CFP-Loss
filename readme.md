Feature-oriented Deep Learning Framework for Pulmonary Cone-beam CT (CBCT) Enhancement with Multi-task Customized Perceptual Loss

By Jiarui Zhu,Weixing Chen, Hongfei Sun, Shaohua Zhi, Jing Cai and Ge Ren.

This is a Pytorch implementation of our paper

### 🛠 Requirements
- Python 3.7+
- PyTorch 1.11.0+

![](figure1.jpg)
> Fig. 1.  The overall architecture of our proposed framework.(a)indicates our multi-task feature-selection network (b)indicates our feature extraction network (c)indicates our CBCT-to-PlanCT translation network.

![](media/54ad677f39e072e1df5e16927a81563d.png)
> Fig. 2. The analysis of the 3DBlock in the proposed CycN-Net.

## Dataset

In this study, we utilized four-dimensional thoracic CBCT and PCT image pairs from 100 lung cancer patients who underwent stereotactic radiotherapy on a Varian Medical Systems (VISION 3253) machine between 2017-2019 at Queen Mary’s Hospital in Hong Kong. These 100 patients were randomly split 70/30 into AI-training and AI-testing groups, with the training dataset further split 56/14 for training and validation. 

Due to the hospital confidential agreement, we cannot share the real patient data at this moment. Yet we can provide one demo patient for convient reproduction of work.

## Training
Read the training tutorial for details.

- For the N-net:

Run [main_train_Nnet.py](https://github.com/shaohua-zhi/N-Net_and_CycNet/blob/master/N-Net/main_train_Nnet.py "main_train_Nnet.py") to train the N-Net model.

File [TrainDataset_Nnet.py](https://github.com/shaohua-zhi/N-Net_and_CycNet/blob/master/N-Net/TrainDataset_Nnet.py "TrainDataset_Nnet.py") illustrates the training dataset.

File [model_Nnet.py](https://github.com/shaohua-zhi/N-Net_and_CycNet/blob/master/N-Net/model_Nnet.py "model_Nnet.py") illustrates the architecture model of N-Net.

Please edit some settings in [main_train_Nnet.py](https://github.com/shaohua-zhi/N-Net_and_CycNet/blob/master/N-Net/main_train_Nnet.py "main_train_Nnet.py"):

-- Path of Training Dataset:

![](media/478d38a9f55394ddff57d0de3cfef7a4.png)

-- Trained Model:

![](media/4f0e993601be57a742ca91762a588b7b.png)

Memory requirement:

1 GPU: NVIDIA GeForce RTX 3090 (24GB)

- For CycN-Net:

Run [main_train_CycNnet.py](https://github.com/shaohua-zhi/N-Net_and_CycNet/blob/master/CycN-Net/main_train_CycNnet.py "main_train_CycNnet.py") to train CycN-Net model.

File [TrainDataset_CycNnet.py](https://github.com/shaohua-zhi/N-Net_and_CycNet/blob/master/CycN-Net/TrainDataset_CycNnet.py "TrainDataset_CycNnet.py") illustrates the training dataset.

File [model_CycNnet.py](https://github.com/shaohua-zhi/N-Net_and_CycNet/blob/master/CycN-Net/model_CycNnet.py "model_CycNnet.py") illustrates the architecture model of CycN-Net.

Please edit some settings in [main_train_CycNnet.py](https://github.com/shaohua-zhi/N-Net_and_CycNet/blob/master/CycN-Net/main_train_CycNnet.py "main_train_CycNnet.py"): 

-- Path of Training Dataset:

![](media/6e4af40b9a732c074a089eb24c89b1de.png)

-- Trained Model:

![](media/37af13dfa2a7a4c03e0a1be111155590.png)

Memory requirement:

3 GPUs: NVIDIA GeForce GTX 1080 (8GB)

## Evaluation
- For N-Net:

Run [main_test_Nnet_XCAT.py](https://github.com/shaohua-zhi/N-Net_and_CycNet/blob/master/N-Net/main_test_Nnet_XCAT.py "main_test_Nnet_XCAT.py") to test the trained model.

File [TestingDataset_Nnet_XCAT.py](https://github.com/shaohua-zhi/N-Net_and_CycNet/blob/master/N-Net/TestingDataset_Nnet_XCAT.py "TestingDataset_Nnet_XCAT.py") illustrates the testing dataset.

Please edit some settings in [main_test_Nnet_XCAT.py](https://github.com/shaohua-zhi/N-Net_and_CycNet/blob/master/N-Net/main_test_Nnet_XCAT.py "main_test_Nnet_XCAT.py"):

-- Load trained model:

![](media/6c8a8877cd5b93ae13b35828b68b0f28.png)

-- Set save path:

![](media/4671c2a4dc086f744251c3a406ae624f.png)

-- Load the testing dataset:

![](media/11cd5cd2070b4001e42c88b3ff6ee418.png)

- For CycN-Net:

Run [main_test_CycNnet_XCAT.py](https://github.com/shaohua-zhi/N-Net_and_CycNet/blob/master/CycN-Net/main_test_CycNnet_XCAT.py "main_test_CycNnet_XCAT.py") to test the trained model.

File [TestingDataset_CycNnet_XCAT.py](https://github.com/shaohua-zhi/N-Net_and_CycNet/blob/master/CycN-Net/TestingDataset_CycNnet_XCAT.py "TestingDataset_CycNnet_XCAT.py") illustrates the testing dataset.

Please edit some settings in [main_test_CycNnet_XCAT.py](https://github.com/shaohua-zhi/N-Net_and_CycNet/blob/master/CycN-Net/main_test_CycNnet_XCAT.py "main_test_CycNnet_XCAT.py"):

-- Load trained model:

![](media/352e4594d1236ee64bf583aff49a537f.png)

-- Set save path:

![](media/210a6f92abfdacc23e0854bbf705f614.png)

-- Load the testing dataset:

![](media/184e0cc91d8cf1e8969827d30710e2ab.png)


## Acknowledgement
dual-pyramid registgration network :
cycle-gan model :https://github.com/Lornatang/CycleGAN-PyTorch
