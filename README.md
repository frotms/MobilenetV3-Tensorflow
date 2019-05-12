# mobilenetv3
This is a multi-GPUs Tensorflow implementation of MobileNetV3 architecture as described in the paper [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf).

# mobilenetv3 large
![](https://i.imgur.com/wZwftDB.png)
# mobilenetv3 small
![](https://i.imgur.com/xO6fFwY.png)
# usage
    from mobilenet_v3 import mobilenet_v3_large, mobilenetv3_small
    
    model = mobilenet_v3_large(input, num_classes, multiplier=1.0, is_training=True, reuse=None)
    
    model = mobilenet_v3_small(input, num_classes, multiplier=1.0, is_training=True, reuse=None)
