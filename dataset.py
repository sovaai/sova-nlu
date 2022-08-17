from torch.utils.data import Dataset
from embedder import SentenceBERTModel
import pandas as pd
import numpy as np
import config as cfg


class IntentDataset(Dataset):
    def __init__(self, csv_file=cfg.DATA):
        # One-Hot encoding labels
        LABELS = cfg.ONE_HOT_LABELS

        SentenceModel = SentenceBERTModel()

        data = pd.read_csv(csv_file, delimiter=';')
        target = data['intent']
        data = data.drop(['intent'], axis=1)

        lst = []
    
        # Data preprocess
        for i in range(len(data)):
            predict = SentenceModel(data.loc[i]['text']).detach().numpy()
            lst.append(predict[0])

        self.train_data = np.array(lst)

        # One-Hot Encoding
        for i in range(len(target)):
            target.loc[i] = LABELS[target.loc[i]]

        self.encoding_target = np.array(pd.get_dummies(target))

    def __len__(self):
        return self.encoding_target.shape[0]

    def __getitem__(self, index):
        return self.train_data[index], self.encoding_target[index]
