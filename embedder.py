from transformers import AutoTokenizer
from models.sentence_bert import SentenceBERT
from models.bert import BertModel
import torch


class SentenceBERTModel():
    def __init__(self, model_path='data/embedder.pt', tokenizer_path='models/tokenizer'):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        encoder = BertModel(
            vocab_size=119547,
            token_type_vocab_size=2,
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=8,
            num_attention_heads=12,
            hidden_act='gelu',
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            pad_token_id=0,
            position_embedding_type='absolute',
            layer_norm_eps=1e-12,
            gradient_checkpointing=False
        )

        self.model = SentenceBERT(encoder, pooling_mode='mean')

        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

        self.model.eval()

    def __call__(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)

        return outputs
