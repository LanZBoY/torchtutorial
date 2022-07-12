import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from NeuralNetwork import FullConnectedNeuralNetwork

global device

def train(dataloader: DataLoader, model: nn.Module, loss_fn, optimizer):
    for epoch in range(20):
        print(f'epoch = {epoch + 1}')
        for x,y in dataloader:
            x,y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            finLoss = loss.item()
        print(f'Loss = {finLoss}')

def test(dataloader: DataLoader, model: nn.Module, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x,y in dataloader:
            x,y = x.to(device), y.to(device)
            pred:Tensor = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Current device is {device}')
    train_data = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor()
    )


    test_data = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor()
    )

    batch_size = 64

    train_dataloader = DataLoader(train_data, batch_size = batch_size, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size = batch_size)

    FCNN :nn.Module = FullConnectedNeuralNetwork(withSoftmax = False).to(device)
    crossEn = nn.CrossEntropyLoss()
    SGD_opt = torch.optim.SGD(FCNN.parameters(), lr = 1e-2)
    train(
        dataloader=train_dataloader,
        model=FCNN,
        loss_fn=crossEn,
        optimizer=SGD_opt
    )

    test(
        dataloader=train_dataloader,
        model=FCNN,
        loss_fn=crossEn
    )