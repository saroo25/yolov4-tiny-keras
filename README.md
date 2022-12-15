---
description: >-
  In this page you will learn how to finetune the Tiny Yolov4 version
  implemented in this github repo:
  https://github.com/bubbliiiing/yolov4-tiny-keras, using google colab platform.
---

# Tiny Yolov4 FineTune

## Import Dependencies

Before start the finetuning we need to upload some data on our drive, Google colab have a good integration with drive. Using Drive API's you can import your data easly.\
Download the base weights of the model, pretrained on coco, from [this](https://github.com/bubbliiiing/yolov4-tiny-keras/releases/download/v1.1/yolov4\_tiny\_weights\_coco.h5) link, and upload on your drive.

On your drive create two folders, one for your images and the other for store the annotation, Tiny Yolov4 require Voc format annotation, like the following example:

```xml
<annotation>
    <folder>Image Folder</folder>
    <filename>_name.jpg</filename>
    <size>
        <width>1128</width>
        <height>2000</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented> 
    <object> //you can add an arbitrary number of object
        <name>Class name of the object</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <occluded>0</occluded>
        <difficult>0</difficult>
        <bndbox>
            <xmin>195</xmin>
            <ymin>437</ymin>
            <xmax>870</xmax>
            <ymax>1325</ymax>
        </bndbox>
    </object>
</annotation>
```

You need to create an annotation for each image you want to process, also if the image is named as Image.jpg then the respective annotation should be called Image.xml.

Create a classes.txt file with all the class in your dataset, and upload it, example:

{% file src=".gitbook/assets/pipe_classes.txt" %}

The last step is to define the requirements needed for the execution, note that the requirements used may change in the future creating dependencies issues, to fix it create an enviroment with the suggest versions, you can use the requirements.txt file below.

{% file src="img/requirements.txt" %}

## Set Colab enviroment

Open [google colab](https://colab.research.google.com/), and create new project, now import the Tiny Yolov4 repository using:

```shell
! git clone https://github.com/bubbliiiing/yolov4-tiny-keras
```

Now import your Drive too using:

```python
from google.colab import drive
drive.mount("/content/gdrive")
```

Copy the annotations,  from the folder to the `yolov4-tiny-keras/VOCdevkit/VOC2007/Annotations` folder. To simplify the work, I defined a copy function from source folder to destination folder, described below.

```python
import os
import shutil
def file_copy(source_folder,destination_folder):
  for annotation in os.listdir(source_folder):
    source =os.path.join(source_folder,annotation)  
    destination = os.path.join(destination_folder,annotation) 
    if os.path.isfile(source):
      shutil.copy(source, destination)
```

Now you easly use it typing:&#x20;

```python
file_copy(source_path, destination_path)
```

Do the same from Image folder to `yolov4-tiny-keras/VOCdevkit/VOC2007/JPEGImages`

Now copy `yolov4_tiny_weights_coco.h5` and `classes.txt` file from your drive to the `yolov4-tiny-keras/model_data` folder and the `requirements.txt` file to the `yolov4-tiny-keras` folder.

Run these commands to set up required dependencies:

```shell
!pip uninstall tensorflow
! pip install -r /content/yolov4-tiny-keras/requirements.txt
```

## Train the model

First we need to split the dataset into train e validation set using `yolov4-tiny-keras/voc_annotation.py`, before run the script we have to change the variable `classes_path` **** at **line 23** to our classes.txt file.

```python
classes_path = 'model_data/pipe_classes.txt'
```

After the change run:

```shell
! python yolov4-tiny-keras/voc_annotation.py
```

Running this script produced the files 2007\_train.txt and 2007\_val.txt.

We can finally finetune our mode, before run the `yolov4-tiny-keras/train.py` script, we need to change the `classes_path` at **line 50** and `model_path` at **line 80** variables as follows.

```
classes_path = 'model_data/pipe_classes.txt'
```

```
model_path = '/content/yolov4-tiny-keras/model_data/yolov4_tiny_weights_coco.h5'
```

During the execution the train.py script produces outputs that are stored in the logs folder, here we can find files like **epxxx-lossxxx-val\_lossxxx.xxx.h5**, this file rapresents the weights of the model at certain epoch, this weights are usefull when you want to resume the training from a checkpoint.

### Resume the Training

Sometimes the train can be stopped by colab, to prevent the progress loss is strongly recommended to download periodically the `yolov4-tiny-keras/logs` folder.

To resume the train instahead of upload the yolov4\_tiny\_weights\_coco.h5 weights we can upload the most recent of the **epxxx-lossxxx-val\_lossxxx.xxx.h5** files.

the variable `Inint_Epoch` at **line 163** in the train.py script should be changed as follow.

```
Init_Epoch= Epoch_value_in_epxxx
```

Note that the variables `classes_path` and `model_path` will need to be changed as described above.

## Convert to ONNX format

Once the model is trained we need to obtain the structure.json file used to convert the model to ONNX format. To do this we need to add the following lines of code to the summary.py file:

```python
json_string = model.to_json()
open('model_data/structure.json', 'w').write(json_string)
```

Also needs to be changed the variable num\_classes at **line 9**&#x20;

Now we can convert the model using the structure.json and the desidered weights using the following script:

```python
import tensorflow as tf
import os
os.environ['TF_KERAS'] = '1'
import keras2onnx
model_file = open('model_data/structure.json').read()
model = tf.keras.models.model_from_json(model_file, custom_objects={'tf': tf})
weights_path = input('model weights path: ')
model.load_weights(weights_path)
onnx_model = keras2onnx.convert_keras(
    model, 
    model.name, 
    channel_first_inputs=['input_1'], 
    target_opset=9
    )
keras2onnx.save_model(onnx_model, 'model_data/yolov4_tiny.onnx')pyth
```

## Convert ONNX to Barracuda

&#x20;If we want to run the ONNX model on Unity we need to convert it in barracuda format, you can follow this script:

```python
import numpy as np
import onnx
from onnx import checker, helper
from onnx import AttributeProto, TensorProto, GraphProto
from onnx import numpy_helper as np_helper
old_onnx = 'model_data/yolov4_tiny.onnx'
new_onnx = 'model_data/yolov4_tiny_barracuda.onnx'
def scan_split_ops(model):
  for i in range(len(model.graph.node)):
    # Node type check
    node = model.graph.node[i]
    if node.op_type != 'Split': continue
    # Output tensor shape
    output = next(v for v in model.graph.value_info if v.name == node.output[0])
    shape = tuple(map(lambda x: x.dim_value, output.type.tensor_type.shape.dim))
    shape = (shape[3], shape[3])
    # "split" attribute addition
    new_node = helper.make_node('Split',
                                 node.input,
                                 node.output, 
                                 split = shape, 
                                 axis = 3)
    # Node replacement
    model.graph.node.insert(i, new_node)
    model.graph.node.remove(node)
model = onnx.load(old_onnx)
model = onnx.shape_inference.infer_shapes(model)
scan_split_ops(model)
checker.check_model(model)
onnx.save(model,new_onnx)
```
