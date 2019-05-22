import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
pase_path = os.path.abspath('../pase')
sys.path.append(pase_path)
print('Add pase to system path:', pase_path)
from pase.models.frontend import wf_builder
import model.model3


# RestNet for visual stream, PASE for audio stream / All network is pretrained.
# Difference from model4: Average pooling on audio time, so that it makes model invariant to time.
class AudioStream(nn.Module):
    def __init__(self, pase):
        super().__init__()
        self.pase = pase
        self.fc1 = nn.Linear(100, 1024)

    def forward(self, x):
        x = self.pase(x)  # (B, 100, 300) for 3s audio
        x = F.adaptive_avg_pool1d(x, 1).squeeze(dim=2)  # (B, 100)
        x = F.relu(x)
        x = self.fc1(x)
        return x


class SVHFNet(nn.Module):
    def __init__(self, res_ckpt_path, pase_cfg_path, pase_ckpt_path):
        super().__init__()
        m3 = model.model3.SVHFNet()
        map_location = None if torch.cuda.is_available() else 'cpu'
        check_point = torch.load(res_ckpt_path, map_location=map_location)
        state_dict = check_point['net']
        m3.load_state_dict(state_dict)
        self.vis_stream = m3.vis_stream
        pase = wf_builder(pase_cfg_path)
        pase.load_pretrained(pase_ckpt_path, load_last=True, verbose=True)
        self.aud_stream = AudioStream(pase)

        self.fc8 = nn.Linear(3072, 1024)
        self.bn8 = nn.BatchNorm1d(1024)
        self.relu8 = nn.ReLU()
        self.fc9 = nn.Linear(1024, 512)
        self.bn9 = nn.BatchNorm1d(512)
        self.relu9 = nn.ReLU()
        self.fc10 = nn.Linear(512, 2)

    def forward(self, face_a, face_b, audio):
        f_a_embedding_ = self.vis_stream(face_a)
        f_b_embedding = self.vis_stream(face_b)
        a_embedding = self.aud_stream(audio)
        concat = torch.cat([f_a_embedding_, f_b_embedding, a_embedding], dim=1)
        x = self.relu8(self.bn8(self.fc8(concat)))
        x = self.relu9(self.bn9(self.fc9(x)))
        x = self.fc10(x)
        return x

if __name__ == '__main__':
    pase_cfg_path = '../../pase/cfg/PASE.cfg'
    pase_ckpt_path = '../../pase/PASE.ckpt'
    res_ckpt_path = '../saved/model3_bn/model_16.pt'
    face_a = torch.empty((2, 3, 224, 224))
    face_b = torch.empty((2, 3, 224, 224))
    audio = torch.empty((2, 1, 48000))
    net = SVHFNet(res_ckpt_path, pase_cfg_path, pase_ckpt_path)
    output = net(face_a, face_b, audio)
    print(output.shape)