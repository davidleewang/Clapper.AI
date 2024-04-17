from sys import byteorder
from array import array
from struct import pack
from model import CustomMobileNetV2

from torchvision import transforms

import librosa
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

import pyaudio
import wave

THRESHOLD = 1000
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 44100

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    silence = [0] * int(seconds * RATE)
    r = array('h', silence)
    r.extend(snd_data)
    r.extend(silence)
    return r

def record(clap=True):
    """
    Record a word or words from the microphone and
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the
    start and end, and pads with 0.5 seconds of
    blank sound to make sure VLC et al can play
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False
    descending_flag = False

    r = array('h')

    seconds = 1.5
    samples_required = seconds * RATE
    chunks_required = samples_required / CHUNK_SIZE

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        # record until two peaks are detected if clap = True
        if clap:
            silent = is_silent(snd_data)

            if silent and snd_started:
                if not descending_flag:
                    descending_flag = True
                else:
                    num_silent += 1
            elif not silent and not snd_started:
                snd_started = True

            if snd_started and num_silent > 30:
                break

        # record to a fixed length of time if clap = False
        else:
            if chunks_required < 0:
                break
            else:
                chunks_required -= 1


    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    # r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path, clap=True):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record(clap)
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()
def load_for_eval():
    # load pre-trained and fine-tuned model
    model = CustomMobileNetV2()

    return model

def eval(audio, model):

    # turn .wav file to waveform data using librosa
    y, sr = librosa.load(audio, sr=None, mono=True)

    # convert waveform data to mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=32, hop_length=16, n_mels=4)
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

    # transform inage to tensor and turn to expected batch form
    image_to_tensor = transforms.ToTensor()


    img = saved_image
    image_tensor = image_to_tensor(img)
    batch = image_tensor.unsqueeze(0)

    # make prediction
    prediction = model(batch).squeeze(0).softmax(0).argmax().item()

    # assign prediction
    if prediction == 1:
        print('___CLAP___')
    else:
        print('_NOT CLAP_')



if __name__ == '__main__':
    model = load_for_eval()
    filename = 'demo.wav'
    hold_flag = True
    while hold_flag == True:
        print("microphone is listening...")
        record_to_file(filename)
        print("microphone captured event...")
        print("submitting audio to model...")
        print("Result:")
        eval(filename, model)



