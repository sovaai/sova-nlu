from embedder import SentenceBERTModel
from models.intent_classifier import IntentClassifier
import torch
import config as cfg


class IntentPipeline:
    def __init__(self):
        self.bert_model = SentenceBERTModel()
        self.intent_classifier = IntentClassifier(output_size=len(cfg.ONE_HOT_LABELS))

        if torch.cuda.is_available():
            self.intent_classifier.load_state_dict(torch.load('data/intent_classifier.pt', map_location='cuda:0'))
        else:
            self.intent_classifier.load_state_dict(torch.load('data/intent_classifier.pt', map_location='cpu'))

        self.intent_classifier.eval()

    def __call__(self, text, threshold):
        intent_input = self.bert_model(text)
        output = self.intent_classifier(intent_input)

        return self.decode(output, threshold)

    @staticmethod
    def decode(x, threshold):
        list_ = []
        result = {}
        x = x.reshape(len(cfg.ONE_HOT_LABELS))
        decode_labels = {v: k for k, v in cfg.ONE_HOT_LABELS.items()}

        for i in range(x.shape[0]):
            if x[i].item() > threshold:
                list_.append((decode_labels[i], x[i].item()))

        list_.sort(key=lambda x: x[1], reverse=True)

        result['intents'] = [{
            'intent': elm[0],
            'confidence': elm[1]
        } for elm in list_]

        return result
