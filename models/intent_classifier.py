import torch.nn as nn
import torch.nn.functional as F


class IntentClassifier(nn.Module):
    def __init__(self, input_size=768, hidden_size=256, output_size=22):
        super(IntentClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, input_):
        x = self.dropout(self.fc1(input_))
        x = F.softmax(self.fc2(x), dim=1)
        return x
