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
  python3 train-MPSVAS.py
```


Note that, train-MPSVAS.py script is used to train the MPS-VAS policy with large Vehicle as target class from DOTA and 6 * 6 grid structure.
In order to train MPS-VAS in different settings as reported in the paper, modify the following:
1. Use the **appropriate model class** for each settings as defined in utils.py ( for example, in order to train MPS-VAS with small car target class from xView and with 11 * 9 grid structure, use the model class (Model_search_Arch_Adapt_pred, Model_search_Arch_Adapt_search_meta) defined in line 600 to line 700 in utils.py. MPS-VAS policy architecture for each setting is also defined in utils.py. We mention the setting name just above the model class definition in each settings. MPS-VAS policy architecture for all different settings we consider is defined between line 502 to line 700 in utils.py script inside utils_c folder.
2. Specify the **right train/test csv file path** as input for that particular setting in "get_datasetVIS and "get_datasetVIS_Classwise" function as defined in utils.py. Provide the path of train csv file in line 343 of utils.py and test csv file in line 347 and 359 of utils.py.
3. Provide the **appropriate label file** for that particular settings in dataloader.py script in the dataset folder. Specifically in line 189 and in line 230.
4. Provide the **appropriate value for num_actions** in line 6 of constant.py. For example, in case of 6 * 6 grid structure, num_actions = 36. Also modify the coord function defined in vas_train.py/vas_test.py based on grid structure.


## Evaluate
**Test the MPS-VAS Policy Network**

To test the policy network on different benchmarks including **xView**, **DOTA** dataset:

```shell
  python3 test-MPSVAS.py
```

In order to test MPS-VAS in different settings, follow the exact same modification instructions as mentioned above for the training part.
Note that, the provided code is used to test MPS-VAS in **uniform query cost** setting, where, we assign the cost budget in line 56. In order to test VAS in **distance based query cost** setting, assign the cost budget in line 85 and uncomment the lines from 120 to 130. 

We provide the trained MPS-VAS policy model parameters for each settings in the following Google Drive folder link:

For example, the trained MPS-VAS policy model parameters when trained with large vehicle as target and number of grids as 36 is saved in files named as model_vas_dota36_lv_adapt_F_meta_pred (to store the weights for prediction module) and model_vas_dota36_lv_adapt_F_meta_search (to store the weights for search module).
