# A Partially Supervised Reinforcement Learning Framework for Visual Active Search
This repository contains implementation of our work titled as __A Partially Supervised Reinforcement Learning Framework for Visual Active Search__. We propose a partially Supervised reinforcement learning framework for Visual Active Search. 

<img src="./figures/framework.png" alt="WAMI_Positives" style="width: 200p;"/>

**PDF**: https://arxiv.org/abs/2310.09689

**Authors**: Anindya Sarkar, Nathan Jacobs, Yevgeniy Vorobeychik.

-------------------------------------------------------------------------------------
## Requirements
**Frameworks**: Our implementation uses **Python3.5** and **PyTorch-v1.4.0** framework.

**Packages**: You should install prerequisites using:
```shell
  pip install -r requirements.txt
```
**Datasets**:



**xView**: You can find the instructions to download images [here](https://challenge.xviewdataset.org/data-format). After downloading the images along with **xView_train.geojson**, you need to run the following script. It will generate a csv file containing the image-path and it's corresponding grid-label sequence. Don't forget to change the directory.

```shell
  python3 Prepare_data.py
```

## Training
**Train the MPS-VAS Policy Network**


To train the policy network on different benchmarks including **xView**, **DOTA** dataset:

```shell
  python3 train-mpsvas.py
```
