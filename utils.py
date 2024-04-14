from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from audiomentations import Compose, Trim, ApplyImpulseResponse, Mp3Compression, PitchShift, AddGaussianSNR


import librosa
import numpy as np
import matplotlib.pyplot as plt
import io

import warnings

import csv

def fxn():
    warnings.warn("deprecated", DeprecationWarning)


class AudioDataset(Dataset):
    def __init__(self, dataset_path, label_type, augmentation_flag=False):
        """

        """
        self.augmentation_flag = augmentation_flag
        aud_ls = []
        with open(f'{dataset_path}/{label_type}', newline='') as csvfile:
            audreader = csv.reader(csvfile, delimiter=',')
            next(audreader, None)  # skip the header row
            for row in audreader:
                aud_ls.append(row)

        aud_ls_tup = []

        for item in aud_ls:
            audio_file = item[0]
            label = int(item[1])

            item_tuple = (audio_file, label)
            aud_ls_tup.append(item_tuple)

        self.length = len(aud_ls_tup)
        self.tup_list = aud_ls_tup
        self.dataset_path = dataset_path

    def __len__(self):
        """

        """
        return self.length

    def __getitem__(self, idx):
        """

        """

        image_to_tensor = transforms.ToTensor()

        if idx > self.length:
            return False
        else:
            tuple = self.tup_list[idx]
            saved_file = tuple[0]
            label = tuple[1]

            # turn .wav file to waveform data using librosa
            y, sr = librosa.load(f'{self.dataset_path}/{saved_file}', sr=None, mono=True)

            import os
            from pathlib import Path
            working_directory = Path(__file__).parent
            ir_directory = os.path.join(working_directory, 'ir_audio')


            # only perform data augmentations if this dataset is a train dataset
            if self.augmentation_flag == True:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fxn()

                    # perform any data augmentations
                    transform = Compose([
                        Trim(top_db=30.0, p=0.5),
                        PitchShift(min_semitones=-5.0, max_semitones=5.0, p=0.5),
                        ApplyImpulseResponse(ir_path=ir_directory, p=0.5),
                        AddGaussianSNR(min_snr_db=5.0, max_snr_db=40.0, p=0.5),
                        Mp3Compression(min_bitrate=32, max_bitrate=64, p=0.5)
                    ])

                    augmented_sound = transform(y, sample_rate=sr)
                    input_sound = augmented_sound

            else:
                input_sound = y


            # convert waveform data to mel spectrogram
            S = librosa.feature.melspectrogram(y=input_sound, sr=sr, n_fft=32, hop_length=16, n_mels=4)
            S_dB = librosa.power_to_db(S, ref=np.max)

            # create mel spectrogram image
            fig = plt.figure()
            fig.add_subplot(1, 1, 1)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
            librosa.display.specshow(S_dB, sr=sr)

            # hand spectrogram image to memory
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)


            # open spectrogram image in PIL from memory, close memory object
            saved_image = Image.open(buf).convert('RGB')
            saved_image.load()
            buf.close()
            plt.close()

            image_tensor = image_to_tensor(saved_image)

            output_tuple = (image_tensor, label)

            return output_tuple


def load_data(dataset_path, label_type, num_workers=0, batch_size=10, **kwargs):
    dataset = AudioDataset(dataset_path, label_type, **kwargs)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)

def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()