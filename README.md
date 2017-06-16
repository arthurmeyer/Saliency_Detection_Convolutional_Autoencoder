# Saliency_Detection_Convolutional_Autoencoder
Saliency detection with a convolutional autoencoder including an edge contrast penalty term to the loss to enforce sharp edges



## Description of the project
This project is a convolutional autoencoder that perform saliency detection. 
Specifically it generates saliency maps directly from raw pixels inputs. 
Both encoder and decoder are based on the VGG architecture.
A specific penalty term has been added to the loss to improve the peormance aswell as direct conenctions between the convolutional and deconvolution layers.



## Requirement
- Python 2.7.6 or 3
- Tensorflow 0.12 (or above) with GPU supported
- Numpy



## How to use
1. First you need to make sure that the images and their labels are located in the approriate folders. 
If you plan to use the most commons datasets, such as [MSRA10K](http://mmcheng.net/msra10k/) or [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html) or [DUT-OMRON](http://saliencydetection.net/dut-omron/#outline-container-orgheadline8) you can download the images and labels into the following folders `.../datasets_split/NAME/images/*` and `.../datasets_split/NAME/labels/*` where `NAME = msra10k` for MSRA10K, `NAME = ecssd` for ECSSD and `NAME = dutomron` for DUT-OMRON. These are the relative paths where the python files are located.

2. Then you need to have a model available (meaning a file with the weight of the network). By default, there are three available models that are B, B+D and B+D+E. B is the baseline autoencoder, B+E has direct connections in addition and B+D+E has both direct connections and the edge contrast penalty enable. If you have no model, then you can start by training one from scratch using the following command:
```
demo.py -o train -m B -init scratch 
```
If you want to train a model using available weights, such as VGG16 on [ILSVRC](http://www.image-net.org/papers/imagenet_cvpr09.pdf) then you can use the following command
```
demo.py -o train -m B -init pretrain 
```
For example you can download the following weight file [here](https://www.cs.toronto.edu/~frossard/post/vgg16/) and place in the folder `.../vgg_weight/*`
To save the final weights from the baseline model so that other model can be trained from this point, make sure to use the command 
```
demo.py -o train -m B -init pretrain -s_copy
```
Afterward you could train a different models with this weight using a command such as 
```
demo.py -o train -m BDE -init restore_w_only
``` 
Finally you could resume the training of the same model and train for a final 1000 steps with
```
demo.py -o train -m BDE -init restore -step 1000
``` 

3. If a model is available, you can test its performance on dataset ECSSD for instance with
```
demo.py -o score -m B -p test -d ecssd -save
``` 
This command will also save the resulting saliency maps in `.../log/MODEL_NAME/*` where `MODEL_NAME` depends on which model you are testing.
