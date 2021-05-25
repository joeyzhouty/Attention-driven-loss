# Attention-Driven Loss for Anomaly Detection in Video Surveillance
This repo is the official open source of [Attention-Driven Loss for Anomaly Detection in Video Surveillance)
* Joey Tianyi Zhou, Le Zhang, Zhiwen Fang, Jiawei Du, Xi Peng, Yang Xiao, "Attention-Driven Loss for Anomaly Detection in Video Surveillance", IEEE Transactions on Circuits and Systems for Video Technology (IEEE TCSVT), 2020.

It is implemented in tensorflow. Please follow the instructions to run the code.  
The backbone network in this work is based on “Future Frame Prediction for Anomaly Detection -- A New Baseline”(CVPR-2018). 

If you feel this project helpful to your research, please cite the following paper
```bibtex
@ARTICLE{8778733, 
author={Zhou, Joey Tianyi and Zhang, Le and Fang, Zhiwen and Du, Jiawei and Peng, Xi and Yang Xiao}, 
journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
title={Attention-Driven Loss for Anomaly Detection in Video Surveillance}, 
year={2020}, 
volume={}, 
number={}, 
month={},}
```
## 1. Installation 
* Install 3rd-package dependencies of python (listed in requirements.txt)
```
numpy==1.14.1
scipy==1.0.0
matplotlib==2.1.2
tensorflow-gpu==1.4.1
tensorflow==1.4.1
Pillow==5.0.0
pypng==0.0.18
scikit_learn==0.19.1
opencv-python==3.2.0.6
```

```shell
pip install -r requirements.txt

pip install tensorflow-gpu==1.4.1
```

## 2. Preparing datasets

Please download the following datasets [ped1.tar.gz, ped2.tar.gz,](http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz) [avenue.tar.gz](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/Avenue_Dataset.zip) and [shanghaitech.tar.gz](https://svip-lab.github.io/dataset/campus_dataset.html)
and move them in to **Data** folder.


## 3. Testing on saved models
* Download the trained models (There are the pretrained FlowNet and the trained models of the papers, such as ped1, ped2 and avenue).
Please manually download pretrained models from [pretrains.tar.gz, avenue, ped1, ped2, flownet](https://drive.google.com/drive/folders/1tG_3ioeZk2-nhA2maC4VFgQie4KBTbQ3?usp=sharing)
and tar -xvf pretrains.tar.gz, and move pretrains into **Codes/checkpoints** folder. **[ShanghaiTech pre-trained models](https://onedrive.live.com/?authkey=%21AMlRwbaoQ0sAgqU&id=303FB25922AAD438%217383&cid=303FB25922AAD438)**

* Running the sript (as ped2 and avenue datasets for examples) and cd into **Codes** folder at first.
```shell
python inference.py  --dataset  ped2    \
                    --test_folder  ../Data/ped2/testing/frames      \
                    --gpu  1    \
                    --snapshot_dir    checkpoints/pretrains/ped2
```

```shell
python inference.py  --dataset  avenue    \
                    --test_folder  ../Data/avenue/testing/frames      \
                    --gpu  1    \
                    --snapshot_dir    checkpoints/pretrains/avenue
```


* There is an example **run.sh** in **Code** folder. 


## 4. Generate attention map via dynamic images

* To generate the attention map, you need to download a dataset and put it into **Data** folder. 
*  Run **Codes/utiles.py/get\_universal_di()**, you need to modify the path and image size in **get\_universal\_di()**. 
*  **di.npy** will be saved in **Code**, **Codes/utiles.py/objectness\_rgb\_estimation()** will load that file. 

## 5. Training from scratch (here we use ped2 and avenue datasets for examples)
* Download the pretrained FlowNet at first and see above mentioned step 3.1 
* Set hyper-parameters
The default hyper-parameters, such as $\lambda_{init}$, $\lambda_{gd}$, $\lambda_{op}$, $\lambda_{adv}$ and the learning rate of G, as well as D, are all initialized in **training_hyper_params/hyper_params.ini**. 

* Running script (as ped2 or avenue for instances) and cd into **Codes** folder at first.
```shell
python train.py  --dataset  ped2    \
                 --train_folder  ../Data/ped2/training/frames     \
                 --test_folder  ../Data/ped2/testing/frames       \
                 --gpu  0       \
                 --iters    80000
```
* Model selection while training
In order to do model selection, a popular way is to testing the saved models after a number of iterations or epochs (Since there are no validation set provided on above all datasets, and in order to compare the performance with other methods, we just choose the best model on testing set). Here, we can use another GPU to listen the **snapshot_dir** folder. When a new model.cpkt.xxx has arrived, then load the model and test. Finnaly, we choose the best model. Following is the script.
```shell
python inference.py  --dataset  ped2    \
                     --test_folder  ../Data/ped2/testing/frames       \
                     --gpu  1
```
Run **python train.py -h** to know more about the flag options or see the detials in **constant.py**.
```shell
Options to run the network.

optional arguments:
  -h, --help            show this help message and exit
  -g GPU, --gpu GPU    the device id of gpu.
  -i ITERS, --iters ITERS
                        set the number of iterations, default is 1
  -b BATCH, --batch BATCH
                        set the batch size, default is 4.
  --num_his NUM_HIS    set the time steps, default is 4.
  -d DATASET, --dataset DATASET
                        the name of dataset.
  --train_folder TRAIN_FOLDER
                        set the training folder path.
  --test_folder TEST_FOLDER
                        set the testing folder path.
  --config CONFIG      the path of training_hyper_params, default is
                        training_hyper_params/hyper_params.ini
  --snapshot_dir SNAPSHOT_DIR
                        if it is folder, then it is the directory to save
                        models, if it is a specific model.ckpt-xxx, then the
                        system will load it for testing.
  --summary_dir SUMMARY_DIR
                        the directory to save summaries.
  --psnr_dir PSNR_DIR  the directory to save psnrs results in testing.
  --evaluate EVALUATE  the evaluation metric, default is compute_auc
```
