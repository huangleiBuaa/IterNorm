# IterNorm

Code for reproducing the results in the following paper:

**Iterative Normalization: Beyond Standardization towards Efficient Whitening** 

Lei Huang, Yi Zhou, Fan Zhu, Li Liu, Ling Shao

*IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019 (accepted).*
[arXiv:1904.03441](https://arxiv.org/abs/1904.03441)


This is the torch implementation (results of experimetns are based on this implementation). Other implementation are shown as follows: 

### [1. Pytorch re-implementation](https://github.com/huangleiBuaa/IterNorm-pytorch)
### [2. Tensorflow implementation](https://github.com/bhneo/decorrelated_bn) by Lei Zhao. 
==============================================================================================

## Requirements and Dependency
* Install [Torch](http://torch.ch) with CUDA (for GPU).
* Install [cudnn](http://torch.ch).
* Install the dependency `optnet` by:
```Bash
luarocks install optnet
 ```
 
 ## Experiments
 
 #### 1.  Reproduce the results of VGG-network on Cifar-10 datasets:
 Prepare the data:  download [CIFAR-10](https://yadi.sk/d/eFmOduZyxaBrT) , and put the data files under `./data/`.
 * Run: 
```Bash
bash y_execute_vggE_base.sh               //basic configuration
bash y_execute_vggE_b1024.sh              //batch size of 1024
bash y_execute_vggE_b16.sh                //batch size of 16
bash y_execute_vggE_LargeLR.sh            //10x larger learning rate
bash y_execute_vggE_IterNorm_Iter.sh      //effect of iteration number
bash y_execute_vggE_IterNorm_Group.sh     //effect of group size
```
Note that the scripts don't inculde the setups of [Decorrelated Batch Noarmalizaiton (DBN)](https://arxiv.org/abs/1804.08450). To reproduce the results of DBN please follow the instructions of the [DBN project](https://github.com/princeton-vl/DecorrelatedBN), and the corresponding hyper-parameters described in the paper. 


#### 2.  Reproduce the results of Wide-Residual-Networks on Cifar-10 datasets:
 Prepare the data: same as in VGG-network on Cifar-10 experiments.
  * Run: 
```Bash
bash y_execute_wr.sh               
```

#### 3. Reproduce the ImageNet experiments. 
 *  Download ImageNet and put it in: `/data/lei/imageNet/input_torch/` (you can also customize the path in `opts_imageNet.lua`)
 *  Install the IterNorm module to Torch as a Lua package: go to the directory `./models/imagenet/cuSpatialDBN/` and run  `luarocks make cudbn-1.0-0.rockspec`. (Note that the modules in `./models/imagenet/cuSpatialDBN/` are the same as in the `./module/`, and the installation by `luarocks` is for convinience in  training ImageNet with multithreads.)
 *  run the script with `z_execute_imageNet_***'
 
 ### This project is based on the training scripts of [Wide Residual Network repo](https://github.com/szagoruyko/wide-residual-networks) and  [Facebook's ResNet repo](https://github.com/facebook/fb.resnet.torch).
 
 ## Contact
Email: lei.huang@inceptioniai.org. Discussions and suggestions are welcome!
