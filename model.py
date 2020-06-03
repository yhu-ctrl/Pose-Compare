import mxnet as mx
from gluoncv.model_zoo import get_model

ctx = mx.gpu()

detector = get_model(
    "ssd_512_mobilenet1.0_coco",
    pretrained=True, ctx=ctx)

estimator = get_model(
    "simple_pose_resnet18_v1b",
    pretrained='ccd24037', ctx=ctx)

detector.reset_class(classes=['person'], reuse_weights={'person':'person'})

detector.hybridize()
estimator.hybridize()