import re
import spacy
from onnxruntime import InferenceSession
from transformers import AutoTokenizer

QA_SESSION = None


# better practice to deploy
def get_qa_predictor():
    global QA_SESSION
    if QA_SESSION is None:
        QA_SESSION = QAPredictor()
    return QA_SESSION


class QAPredictor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.tokenizer = AutoTokenizer.from_pretrained('./tokenizer_checkpoint')
        self.session = InferenceSession("./quant_model.onnx")

    def clean_text(self, text):
        '''
        This version assumes [*DEID] are the tags to ignore instead of [DEID]
        '''
        result = re.sub(r'<[^>]+>|\[.*EID\]', '', text)
        return result.strip()

    def sentencize(self, text):
        return [str(s) for s in self.nlp(text).sents]

    def question_answering(self, question, context, clean=True):
        # ONNX Runtime expects NumPy arrays as input
        if clean:
            context = self.clean_text(context)
        inputs = self.tokenizer(
            question, context, padding='max_length', max_length=512,
            truncation="only_second", return_tensors="np"
            )
        inputs = self.tokenizer("Using DistilBERT with ONNX Runtime!", return_tensors="np")
        outputs = self.session.run(output_names=["last_hidden_state"], input_feed=dict(inputs))

        return outputs
