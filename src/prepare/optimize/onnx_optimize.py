# optimize with onnx vanilla library
import onnx
import onnx.optimizer

src_onnx = 'mobilenetv2_1.0.onnx'
opt_onnx = 'mobilenetv2_1.0.opt.onnx'

# load model
model = onnx.load(src_onnx)

# optimize
model = onnx.optimizer.optimize(model, ['fuse_bn_into_conv'])

# save optimized model
with open(opt_onnx, "wb") as f:
    f.write(model.SerializeToString())
