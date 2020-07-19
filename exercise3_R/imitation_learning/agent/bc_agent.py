import torch
from agent.networks import CNN
device = torch.device("cpu")

class BCAgent:

    def __init__(self,history_length=1):
        # TODO: Define network, loss function, optimizer
        # self.net = CNN(...)
        self.net = CNN(history_length).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)
        # self.optimizer = torch.optim.SGD(self.net.parameters(),lr=conf.lr,momentum = 0.9)
        self.loss_func = torch.nn.CrossEntropyLoss().to(device)

    def update(self, X_batch, y_batch):
        # TODO: transform input to tensors
        # TODO: forward + backward + optimize
        X_batch = torch.tensor(X_batch).permute(0, 3, 1, 2).to(device)
        y_batch = torch.LongTensor(y_batch).view(-1).to(device)
        y_pred = self.predict(X_batch).to(device)
        self.optimizer.zero_grad()
        loss = self.loss_func(y_pred, y_batch).to(device)
        loss.backward()
        self.optimizer.step()

        return loss

    def predict(self, X):
        # TODO: forward pass
        outputs = self.net(X)
        # outputs = torch.FloatTensor(outputs)
        return outputs

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))


    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)
