import os
import numpy as np
import torch
import librosa
import random
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class Dataset(data.Dataset):
    def __init__(self, data_dir, triplet_dir, mode, fixed_offset):
        self.data_dir = data_dir
        self.fixed_offset = fixed_offset
        triplet_path = os.path.join(triplet_dir, 'triplets_' + mode + '.txt')
        with open(triplet_path, 'r') as f:
            self.all_triplets = f.readlines()
        self.all_triplets = self.all_triplets[1:]
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

    def __getitem__(self, index):
        triplet = self.all_triplets[index].split('\t')
        print(triplet)
        triplet[1] = triplet[1].replace('.wav', '.wav.npy')
        triplet[1] = triplet[1].replace('vox1_dev_wav', 'vox1_dev_norm')
        real_audio_path = os.path.join(self.data_dir, triplet[1])
        real_face_path = os.path.join(self.data_dir, triplet[2])
        fake_face_path = os.path.join(self.data_dir, triplet[3])
        real_audio = self.load_audio(real_audio_path)
        real_face = self.load_face(real_face_path)
        fake_face = self.load_face(fake_face_path)
        which_side = random.randint(0, 1)
        if which_side == 0:
            ground_truth = torch.LongTensor([0])
            face_a = real_face
            face_b = fake_face
        else:
            ground_truth = torch.LongTensor([1])
            face_a = fake_face
            face_b = real_face
        return real_audio, face_a, face_b, ground_truth

    def load_audio(self, audio_path):
        y = np.load(audio_path)
        if self.fixed_offset:
            offset = 0
        else:
            max_offset = y.shape[2] - 300
            offset = random.randint(0, max_offset)
        y = y[:, :, offset:offset+300]
        # spect = Dataset.get_spectrogram(y)
        # for i in range(spect.shape[1]):
        #     f_bin = spect[:, i]
        #     f_bin_mean = np.mean(f_bin)
        #     f_bin_std = np.std(f_bin)
        #     spect[:, i] = (spect[:, i] - f_bin_mean) / f_bin_std
        # spect = np.expand_dims(spect, axis=0)
        return y

    def load_face(self, face_path):
        # NOTE: 3 channels are in BGR order
        img = Image.open(face_path)
        if img.size != (224, 224):
            img = img.resize((224, 224), resample=Image.BILINEAR)
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.all_triplets)

    @staticmethod
    def get_spectrogram(y, n_fft=1024, hop_length=160, win_length=400, window='hamming'):
        y_hat = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
        y_hat = y_hat[:-1, :-1]
        D = np.abs(y_hat)
        return D


def custom_collate_fn(batch):
    real_audio = [torch.from_numpy(item[0]) for item in batch]
    face_a = [item[1] for item in batch]
    face_b = [item[2] for item in batch]
    gt = [item[3] for item in batch]
    real_audio = torch.stack(real_audio, dim=0)
    face_a = torch.stack(face_a, dim=0)
    face_b = torch.stack(face_b, dim=0)
    gt = torch.cat(gt, dim=0)
    return [real_audio, face_a, face_b, gt]


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = Dataset('../SVHF_dataset', './triplets', 'train', False)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=True, num_workers=0, collate_fn=custom_collate_fn)

    for step, (real_audio, face_a, face_b, ground_truth) in enumerate(loader):
        print(real_audio.shape)  # (B, 1, 512, 300)
        print(face_a.shape)  # (B, 3, 224, 224)
        print(face_b.shape)
        print(ground_truth.shape)  # (B)
        break
