from config import config
from Agent import Agent
import torch
def train():
    model = Agent(config)
    sents = torch.randn(1, 10, config.embed_dim)
    labels = torch.tensor([1])
    targets = torch.randn(1, config.embed_dim)
    sents = torch.randn(1, 10, config.embed_dim)
    pred, actions, reward = model(sents, targets, labels)
    print(pred)
    print(actions)
    print(reward)
    reward.backward()

if __name__ == "__main__":
    train()