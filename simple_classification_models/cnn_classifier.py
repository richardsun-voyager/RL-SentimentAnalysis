import torch
import torch.nn as nn
import torch.nn.functional as F
class CNNClassifier(nn.Module):
    def __init__(self, embedding_dim, output_size, kernel_dim=300, kernel_size=[3, 4, 5], dropout=0.5):
        super(CNNClassifier, self).__init__()
        
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_dim, (K, embedding_dim)) for K in kernel_size])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_size) * kernel_dim, output_size)
    
    def init_weight(self):
        pass
        
    def forward(self, inputs):
        '''
        inputs:batch_size, word_num, emb_dim
        
        '''
        inputs = inputs.unsqueeze(1)

        inputs = [F.relu(conv(inputs)).squeeze(3) for conv in self.convs]
        inputs = [F.avg_pool1d(i, i.size(2)).squeeze(2) for i in inputs]
        concated = torch.cat(inputs, 1)
        if self.training:
            concated = self.dropout(concated)
        out = F.softmax(self.fc(concated), 1)
        return out