import re
import spacy
from transformers import pipeline

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
        self.pipeline = pipeline(
          "question-answering",
          model="deepset/roberta-base-squad2",
          tokenizer="deepset/roberta-base-squad2"
        )

    def clean_text(self, text):
        '''
        This version assumes [*DEID] are the tags to ignore instead of [DEID]
        '''
        result = re.sub(r'<[^>]+>|\[.*EID\]', '', text)
        return result.strip()

    def sentencize(self, text):
        return [str(s) for s in self.nlp(text).sents]

    def question_answering(self, question, context, clean=True):
        if clean:
            context = self.clean_text(context)
        res = self.pipeline({'question': question, 'context': context})

        return {k: v for k, v in res.items() if k in ['answer', 'score']}
