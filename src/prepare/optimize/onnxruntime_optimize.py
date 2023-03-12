# with onnxruntime library by microsoft


from onnxruntime.transformers import optimizer
import onnxruntime as ort
import torch

optimized_model = optimizer.optimize_model("bert.onnx", model_type='bert', num_heads=12, hidden_size=768)
optimized_model.convert_float_to_float16()  # don't use on CPU will be slower
optimized_model.save_model_to_file("bert_fp16.onnx")


quantized_model = torch.quantization.quantize_dynamic(
    pt_model, {torch.nn.Linear}, dtype=torch.qint8
)

sess_options = ort.SessionOptions()
# Set graph optimization level
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# To enable model serialization after graph optimization set this
sess_options.optimized_model_filepath = "<model_output_path\optimized_model.onnx>"
session = ort.InferenceSession("<model_path>", sess_options)


def get_session_for_provider(model_path: str, provider: str) -> ort.InferenceSession: 
    assert provider in ort.get_all_providers(), f"provider {provider} not found, {ort.get_all_providers()}"

    # Few properties that might have an impact on performances (provided by MS)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Load the model as a graph and prepare the CPU backend
    session = ort.InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()
    return session
