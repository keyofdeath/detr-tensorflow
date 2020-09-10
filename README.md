# DEtection TRansformer Network (DETR) - Tensorflow

Implementation of the **DETR** (**DE**tection **TR**ansformer) network (Carion, Nicolas, et al., 2020) in *Tensorflow*. The model was originally developed by Facebook Inc. and implemented in *PyTorch*. This repository solely aims to make the architecture accessible for *Tensorflow* users.

&nbsp;
&nbsp;
## 1. References
- **Research Paper:** [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
- **Original PyTorch Implementation:** [GitHub](https://github.com/facebookresearch/detr)

<img src="img/DETR.png">
*Image is taken from (Carion, Nicolas, et al., 2020)

&nbsp;
&nbsp;
## 2. Packages
We use the following packages:
- python 3.7.3
- tensorflow 2.1.0
- tensorflow-addons 0.11.0.dev0 **(`*`)**
- scipy 1.4.1
- numpy 1.18.3
- numba 0.49.0

**(`*`)** You have to git clone and install this version from [here](https://github.com/tensorflow/addons.git). Required in order to allow the optimizer to handle weight decay schedules ([reference](https://github.com/tensorflow/addons/commit/b9f9ac5cc54c9c2169a8197d0d61adcb42b764e2)).

&nbsp;
&nbsp;
## 3. Start Training

&nbsp;
### 3.A. Install Modules
```console
git clone https://github.com/auvisusAI/detr-tensorflow.git 			# Clone Repository
cd detr-tensorflow/								# Change to directory
pip3 install -e .								# Install repository
pip3 install -r reqs/requirements.txt			# Install requirements
pip3 install -r reqs/test-requirements.txt		# Install test-requirements
```

&nbsp;
### 3.B. Data Preparation

You can use this repository for data stored in PascalVOC or in the COCO format. Therefore, your data should match the following folder structures:

#### COCO
```console
storage_path/				# path/to/data_storage
	images/				# includes all .jpg files
	coco.json 			# COCO annotation .JSON
```

#### PascalVOC
```console
storage_path/				# path/to/data_storage
	labels/				# includes all .txt files where each line corresponds to one object in image
	images/				# includes all .jpg files
```

**Dependent on your data structure, you can choose between two types of DataFeeder as specified in `detr_models/data_feeder`. Please make sure you take the correct one, when training your model.**


&nbsp;
### 3.C. Execute Training

#### 3.C.I. Via Command Line
You can easily start the training procedure from `detr_models/detr/` using:

```console
python3 train.py --storage_path <PATH> --output_dir <PATH> <Additional Parameters>
```

Additional parameters such as *epochs*, *batch_size* etc. can be set. Please take a look at the help text for a complete overview using:

```console
python3 train.py --help
```

If no additional parameters are used, the defaults as specified in `detr_models/detr/config.py` will be used.

#### 3.C.II. Via Jupyter Notebook

If you want to execute training (e.g. on a pre-trained model) or just get a quick overview over the model architecture, you can also use the jupyter notebook `DETR.ipynb` provided in `/notebooks`.

&nbsp;
&nbsp;
## 4. To-DOs
- [x] Adjust `data_feeder/loadlabel` to handle Coco annotations
- [x] Include unittests to verify COCOFeeder
- [ ] Take `max_obj` into config
- [ ] Include mask head to model for segmentation
- [x] Parameterize to handle images with varying shape and paddings
- [ ] Parameterize backbone config
- [ ] Include inference script
- [x] Include inference notebook
- [x] Include notebook to verify COCOFeeder
- [ ] Take `l1_cost_factor` and `iou_cost_factor` into config


## 5. Help - We need somebody

As you can see, there are still many open to-dos. We are happy for all contributions to improve this *Tensorflow* implementation.


<img align="right" src="img/auvisus.svg" width="100" >
