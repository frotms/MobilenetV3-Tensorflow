# mobilenetv3
This is a multi-GPUs Tensorflow implementation of MobileNetV3 architecture as described in the paper [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244).  
[paper V2](https://arxiv.org/pdf/1905.02244v2.pdf) changed some layers from [paper V1](https://arxiv.org/pdf/1905.02244v1.pdf).  
The implementation of paper V1 see [branch paper_v1](https://github.com/frotms/MobilenetV3-Tensorflow/tree/paper_v1) in this repository for detail.

Tested on tf1.3.0, tf1.10.0, python3.5.

# mobilenetv3 large
![](https://i.imgur.com/9wWE6GP.png)
# mobilenetv3 small
![](https://i.imgur.com/BdbM7Xp.png)
# usage
    from mobilenet_v3 import mobilenet_v3_large, mobilenetv3_small
    
    model, end_points = mobilenet_v3_large(input, num_classes, multiplier=1.0, is_training=True, reuse=None)
    
    model, end_points = mobilenet_v3_small(input, num_classes, multiplier=1.0, is_training=True, reuse=None)
