import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx

class CNV2(nn.Module):
    def __init__(self):
        super(CNV2, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3)
        self.fc1 = nn.Linear(8, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 8)
        #print(x.shape)
        x = F.softmax(self.fc1(x))
        #x = torch.cat((x, x))
        return x

x = torch.randn((1, 1, 18, 18), dtype=torch.float32)
graph = CNV2()
out = graph(x)

torch.onnx.export(graph,
                  x,
                  "test.onnx",
                  export_params=True,
                  do_constant_folding=True,
                  input_names = ['input'],
                  output_names = ['output']
)

