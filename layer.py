import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x = torch.tensor(
    [
        [0., 0., 1.],
        [0., 1., 1.],
        [1., 0., 1.]
    ]
)
y = torch.tensor(
    [
        [1.],
        [0.],
        [1.]
    ]
)


class Net(nn.Module):
    """Some Information about Net"""

    def __init__(self):
        super(Net, self).__init__()
        self.input = nn.Linear(3, 4)
        self.sigmoid = nn.Sigmoid()
        self.h1 = nn.Linear(4, 4)
        self.output = nn.Linear(4, 1)

    def forward(self, x):
        x = self.input(x)
        x = self.sigmoid(x)
        x = self.h1(x)

        x = self.output(x)
        return x


# print(x.shape)
model = Net()


# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training

for epoch in range(6000):
    output = model(x)
    loss = criterion(output, y)
    print(loss)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

guess = model(x)
print(guess)
# result = Network(test_exp).data[0][0].item()

# print('Result is: ', result)
