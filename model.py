import mxnet as mx
from gluoncv.model_zoo import get_model

ctx = mx.gpu()

detector_name = "ssd_512_mobilenet1.0_coco"
detector = get_model(detector_name, pretrained=True, ctx=ctx)

estimator_name = "simple_pose_resnet18_v1b"
estimator = get_model(estimator_name, pretrained='ccd24037', ctx=ctx)

detector.reset_class(classes=['person'], reuse_weights={'person':'person'})

detector.hybridize()
estimator.hybridize()