from transformers import AutoTokenizer
from optimum.onnxruntime import (
    AutoQuantizationConfig,
    ORTModelForQuestionAnswering,
    ORTQuantizer
)

if __name__ == "__main__":
    model_id = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained("./qa_tokenizer")

    model = ORTModelForQuestionAnswering.from_pretrained(model_id, export=True)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
    quantizer = ORTQuantizer.from_pretrained(model)
    quantizer.quantize(save_dir='./qa_model', quantization_config=qconfig)
