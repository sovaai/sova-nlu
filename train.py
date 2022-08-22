import torch
import torch.nn as nn
from models.intent_classifier import IntentClassifier
from dataset import IntentDataset
from torch.utils.data import DataLoader
import config as cfg


def train():
    model = IntentClassifier(output_size=len(cfg.ONE_HOT_LABELS))
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM)
    dataset = IntentDataset()

    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)

    # Train
    for epoch in range(cfg.EPOCHS):
        loss_lst = []

        for data, target in dataloader:
            prediction = model(data)
    
            loss = criterion(prediction.float(), target.float())

            loss_lst.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) % 50 == 0:
            print(f'Epoch: {epoch + 1}, mean loss: {sum(loss_lst) / len(loss_lst)}')

    torch.save(model.state_dict(), 'data/intent_classifier.pt')


if __name__ == '__main__':
    train()
