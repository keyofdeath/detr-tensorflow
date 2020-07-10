# DEtection TRansformer Network (DETR) - Tensorflow

Implementation of the **DETR** (**DE**tection **TR**ansformer) network (Carion, Nicolas, et al., 2020) in *Tensorflow*.


## References
- **Reserach Paper:** [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
- **Original PyTorch Implementation:** [GitHub](https://github.com/facebookresearch/detr)

## Packages
We use the following packages:
- python 3.7.3
- tensorflow 2.1.0
- scipy 1.4.1
- numpy 1.18.3
- numba 0.49.0



## Start Training

### Install Modules
```console
git clone https://github.com/auvisusAI/detr-tensorflow.git 			# Clone Repository
cd detr-tensorflow/								# Change to dir
pip3 install -e .								# Install repository and required packages
```

### Data Preparation
```console
storage_path/				# path/to/data_storage
	labels/				#.txt files where each line corresponds to one object in image
	images/				#.jpg files
```


### Execute Training

#### Via Command Line
You can easily start the training procedure from `detr_models/detr/` using:

```console
python3 train.py --storage_path <PATH> --output_dir <PATH> <Additional Parameters>
```

Additional parameters such as *epochs*, *batch_size* etc. can be set. Please take a look at the help text for a complete overview using:

```console
python3 train.py --help
```

If no additonal parameters are used, the defaults as specified in `detr_models/detr/config.py` will be used.

#### Via Jupyter Notebook

If you want to execute training (e.g. on a pre-trained model) or just get a quick overview over the model architecture, you can also use the jupyter notebook `DETR.ipynb` provided in `/notebooks`.

## To-DOs
- [] Include tests to verify code
- [] Take `max_obj` into config
- [] Include mask head to model for segmentation
- [] Parameterize to handle images with variyng shape and paddings
- [] Parameterize backbone config
- [] Include inference script
- [] Include inference notebook

<img align="right" src="img/auvisus.svg" width="100" >
