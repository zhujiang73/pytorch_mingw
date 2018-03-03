import torch
import torchvision
from torch.autograd import Variable
import torch.utils.data.dataloader as Data

train_data = torchvision.datasets.MNIST(
    './minst_data', train=True, transform=torchvision.transforms.ToTensor(), download=True
)
test_data = torchvision.datasets.MNIST(
    './minst_data', train=False, transform=torchvision.transforms.ToTensor()
)
print("train_data:", train_data.train_data.size())
print("train_labels:", train_data.train_labels.size())
print("test_data:", test_data.test_data.size())

train_loader = Data.DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=64)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64 * 3 * 3, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out

model = Net()

print("model : \n{}".format(model))

optimizer = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()

for n in range(5):
    print('Loop n :  {}'.format(n + 1))
    # training: --------------------------------------
    train_loss = 0.0
    train_acc = 0.0
    
    m = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        train_loss += loss.data.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.data.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        m = m + 1
        if (m % 100 == 0):
            print("loss : {:.6f} m={}".format(loss,m) )

    print('Train Loss: {:.6f}'.format(train_loss / (len(train_data))))

    # Testing: ---------------------------------------  
    model.eval()
    eval_loss = 0.0
    eval_acc = 0.0
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = Variable(batch_x, requires_grad=False), Variable(batch_y, requires_grad=False)
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.data.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.data.item()

    print('Test Loss: {:.6f}'.format(eval_loss / (len(test_data))))



