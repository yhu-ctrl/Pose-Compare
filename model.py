import mxnet as mx
from gluoncv.model_zoo import get_model

ctx = mx.gpu()

detector = get_model(
    "ssd_512_mobilenet1.0_coco",
    pretrained=True, ctx=ctx)

estimator = get_model(
    "simple_pose_resnet101_v1d",
    pretrained=True, ctx=ctx)

detector.reset_class(classes=['person'], reuse_weights={'person':'person'})

detector.hybridize()
estimator.hybridize()