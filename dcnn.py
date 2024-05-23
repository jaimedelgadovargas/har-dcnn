import torch.nn as nn
import torch.nn.functional as F

class DCNN(nn.Module):
  def __init__(self, num_classes):
    super(DCNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, stride=1, padding=0)
    self.pool1 = nn.AvgPool2d(kernel_size=4, stride=4, padding=0)
    self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=5, stride=1, padding=0)
    self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    self.fc1 = nn.Linear(in_features=10*2*6, out_features=120)
    self.fc2 = nn.Linear(in_features=120, out_features=num_classes)

  def forward(self, x):
    s1 = self.pool1(F.relu(self.conv1(x)))
    s2 = self.pool2(F.relu(self.conv2(s1)))
    x = x.view(-1, 10*2*6)
    f1 = F.relu(self.fc1(s2))
    output = F.softmax(self.fc2(f1), dim=1)
    return output
