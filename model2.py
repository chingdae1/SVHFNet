# Deeper VGG Face
import torch
import torch.nn as nn


class VisualStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),nn.ReLU(),
            nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2]),

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2]),

            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2]),

            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2]),

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2])
        )
        self.fc6 = nn.Linear(25088, 4096)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU()
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(4096, 1024)

    def forward(self, x):
        x = self.body(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout6(self.relu6(self.fc6(x)))
        x = self.dropout7(self.relu7(self.fc7(x)))
        x = self.fc8(x)
        return x


class AudioStream(nn.Module):
    def __init__(self, N=8):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(7, 7), stride=2, padding=1),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=2, padding=(0, 1)),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3)),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 2)),
        )
        self.fc6 = nn.Linear(9*1*256, 4096)
        self.relu6 = nn.ReLU()
        self.apool6 = nn.AvgPool2d((1, N))
        self.fc7 = nn.Linear(4096, 1024)

    def forward(self, x):
        x = self.body(x)  # (B, 256, 9, 8)
        x = x.view(x.shape[0], -1, x.shape[3])  # (B, 2304, 8)
        x_out_list = []
        for i in range(x.shape[2]):
            x_out = self.relu6(self.fc6(x[:, :, i]))
            x_out_list.append(x_out)
        x = torch.stack(x_out_list, dim=2)  # (B, 4096, 8)
        x = self.apool6(x)
        x = x.view(-1, 4096)
        x = self.fc7(x)
        return x


class SVHFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.vis_stream = VisualStream()
        self.aud_stream = AudioStream()
        self.fc8 = nn.Linear(3072, 1024)
        self.relu8 = nn.ReLU()
        self.fc9 = nn.Linear(1024, 512)
        self.relu9 = nn.ReLU()
        self.fc10 = nn.Linear(512, 2)

    def forward(self, face_a, face_b, audio):
        f_a_embedding_ = self.vis_stream(face_a)
        f_b_embedding = self.vis_stream(face_b)
        a_embedding = self.aud_stream(audio)
        concat = torch.cat([f_a_embedding_, f_b_embedding, a_embedding], dim=1)
        x = self.relu8(self.fc8(concat))
        x = self.relu9(self.fc9(x))
        x = self.fc10(x)
        return x

if __name__ == '__main__':
    face_a = torch.empty((4, 3, 224, 224))
    face_b = torch.empty((4, 3, 224, 224))
    audio = torch.empty((4, 1, 512, 300))

    net = SVHFNet()
    output = net(face_a, face_b, audio)
    print(output.shape)
    _, argmax = torch.max(output, 1)
    labels = torch.LongTensor([0, 0, 1, 1])
    accuracy = (labels == argmax.squeeze()).float().mean()
