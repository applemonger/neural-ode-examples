import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader
from torchdiffeq import odeint_adjoint
import os
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_mnist_data_loaders(batch_size: int) -> tuple[DataLoader, DataLoader]:
    # Multiprocessing for batch loading
    num_workers = os.cpu_count()
    prefetch_factor = 2 # Number of batches to buffer for each worker

    # Data transformation
    transform = Compose([ToTensor()])

    # Training dataset loader
    train_dataset = MNIST('./data/', train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        shuffle=True
    )

    # Test dataset loader
    test_data = MNIST('./data/', train=False, download=True, transform=transform)
    test_loader = DataLoader(
        test_data, 
        batch_size=1000, 
        pin_memory=True,
        shuffle=False,
        drop_last=True
    )

    return train_loader, test_loader


class ODEFunc(nn.Module):
    def __init__(self, input_size: int):
        """
        This is the dynamics function, f(x, t, theta), for the Neural ODE. It is fairly 
        simple, consisting of a single linear layer with ReLU activations. For more 
        complex morphisms, we can add more layers to this function.

        For MNIST, we expect a flattened 28 x 28 greyscale image i.e. a vector of length
        784. Note that in the forward function, we tack on time (t) onto the input, so 
        the actual expected input to the linear layer is a vector of length 785, but 
        the output is still 784.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size + 1, input_size),
            nn.ReLU()
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Create a time tensor (all elements == t) that can be concatenated onto the 
        # input batch (x)
        time = torch.ones_like(x[:, :1]) * t 
        # Append time t onto the batch
        state_and_time = torch.cat([time, x], dim=1)
        return self.net(state_and_time)
    

class ODEBlock(nn.Module):
    def __init__(self, input_size: int):
        """
        This is the ODE layer that we can add into any neural net. The ODE layer should 
        simulate the results of a deep residual network ("resnet").
        
        During the forward pass, we use the adjoint method to estimate the ODE that 
        morphs x from time 0 to time 1. The adjoint method is great because it does not 
        require us to backpropagate through the actual ODE solver. `torchdiffeq` will 
        take care of updating the weights of our dynamics function f(x, t, theta) with 
        respect to our chosen loss function.
        """
        super().__init__()
        self.ode_function = ODEFunc(input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.Tensor([0, 1]).float().type_as(x)
        _initial_state, final_state = odeint_adjoint(self.ode_function, x, t)
        return final_state


class Classifier(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        """
        Our classifier function performs the necessary transformations of flattening our
        input images (28 x 28 pixel tensors), as well as mapping the output of our ODE 
        (a vector of length 784) to the label space (labels 0 through 10). The hope is 
        that our ODEBlock learns to morph different classes of images to easily linearly
        separable spaces.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            ODEBlock(input_size),
            nn.Linear(input_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def accuracy(model: nn.Module, data_loader: DataLoader) -> float:
    """
    Calculates the accuracy of the model on a given dataset (DataLoader).
    """
    model.eval()
    n_correct = 0
    n_total = 0
    for batch, labels in data_loader:
        predictions = model(batch.to(device)).cpu()
        correct = labels == torch.argmax(predictions, dim=1)
        n_correct += correct.sum().item()
        n_total += labels.size(0)

    return n_correct / n_total


if __name__ == "__main__":
    experiment_name = "mnist_experiment_9"
    epochs = 10
    batch_size = 128
    input_size = 28 * 28 # Number of pixels in an MNIST image
    output_size = 10 # Number of labels in the MNIST dataset i.e. 0 through 9
    model = Classifier(input_size, 10).to(device)
    optimizer = Adam(model.parameters(), lr=0.02)
    loss_fn = nn.CrossEntropyLoss()

    # Load training and test dataset iterators
    train_loader, test_loader = get_mnist_data_loaders(batch_size)

    # Tensorboard logger
    writer = SummaryWriter(f'runs/{experiment_name}')

    # Training loop
    batch_i = 0
    for epoch in tqdm(range(epochs)):
        for batch, labels in tqdm(train_loader):
            optimizer.zero_grad()
            batch, labels = batch.to(device), labels.to(device)
            output = model(batch)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            # Log loss to Tensorboard
            batch_i += 1
            writer.add_scalar('loss/train', loss.item(), batch_i)

        # Calculate accuracy on validation dataset per epoch
        with torch.no_grad():
            acc = accuracy(model, test_loader)
            writer.add_scalar('accuracy/validation', acc, epoch)