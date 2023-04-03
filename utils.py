import os
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
import argparse
from sklearn import preprocessing
import random
import numpy as np
import torch
import torchaudio.transforms
from torchaudio.transforms import TimeMasking, FrequencyMasking
from torch.utils.data import Dataset, DataLoader
import pickle



random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

'''
This function will record the user saying
the wakeword from their computer, and save the file
to the data folder in the project. The saved .wav
datasets will be the positive label set for our
algorithm
'''
def collect_wakeword_data(seconds, fs, num_pos_samples, file_name, channels):
    for i in range(num_pos_samples):
        input("Press any key to begin wakeword audio colelction!")
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=channels)
        sd.wait()  # Wait until recording is finished
        write(f'{file_name}/wakeword_{i}.wav', fs, myrecording)  # Save as WAV file
        print("collected and save file", i, "to dir!")
        
'''
This function will record the background noise
for the user. User should not utter the wakeword
while this function runs. Instead, move around
your office or home, and be sure to play nonrelated sounds
in the background. The more variety of noise
in this dataset, the better outcomes the algorithm
will have. We will complement this dataset with
downloaded "noise" datasets for the negative label
'''
def collect_ambient_data(seconds, fs, num_neg_samples, file_name, channels):
    for i in range(num_neg_samples):
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=channels)
        sd.wait()  # Wait until recording is finished
        write(f'{file_name}/background_{i}.wav', fs, myrecording)  # Save as WAV file
        print("collected and save file", i, "to dir!")


class DataAugmentation():

    def __init__(self, args):
        self.data_folder = args.file_name
        self.fs = args.sample_rate

    def forward_time_shift(self, audio_signal, ms_max=200):
        amt = int(ms_max * self.fs /1000)
        shift_amount = random.randint(1,amt)
        new_audio_signal = np.concatenate([np.zeros(shift_amount), audio_signal[shift_amount:]])
        assert len(new_audio_signal) == len(audio_signal)
        return new_audio_signal
        
    def backward_time_shift(self, audio_signal, ms_max=50):
        amt = int(ms_max*self.fs / 1000)
        shift_amount = random.randint(1, amt)
        s = len(audio_signal) - shift_amount
        new_audio_signal = np.concatenate(audio_signal[:s+1], np.concatenate[np.zeros(shift_amount)])
        assert len(new_audio_signal) == len(audio_signal)
        return new_audio_signal
        
    def pitch_shift(self, audio_signal, min_=-3, max_=5):
        shift = np.random.randint(min_, max_+1)
        return librosa.effects.pitch_shift(y=audio_signal, sr=self.fs, n_steps=shift)

    def frequency_mask(self, mel):
        mel_pt = torch.from_numpy(mel).reshape(1, mel.shape[0], mel.shape[1])
        frequency_mask = FrequencyMasking(freq_mask_param=30)
        masked_signal = frequency_mask(mel_pt)
        masked_signal = masked_signal.squeeze().numpy()
        assert masked_signal.shape == mel.shape
        return mel
        
    def time_mask(self, mel):
        mel_pt = torch.from_numpy(mel).reshape(1, mel.shape[0], mel.shape[1])
        time_mask = TimeMasking(time_mask_param=5)
        masked_signal = time_mask(mel_pt)
        masked_signal = masked_signal.squeeze().numpy()
        assert masked_signal.shape == mel.shape
        return masked_signal



class Preprocess():
    
    def __init__(self, args):
        self.fs = args.sample_rate

    def trim_beginning(self, file_name, ms=300):
        audio_sample, sample_rate = librosa.load(file_name, sr=None)
        amt = int(ms*self.fs / 1000)
        trimmed_audio = audio_sample[amt:]
        return trimmed_audio, sample_rate
    
    def create_mel_spectrogram_from_file(self, file_name):
        audio_sample, sample_rate = librosa.load(file_name, sr=None)
        print("sample_Rate", sample_rate, self.fs)
        mel = librosa.feature.melspectrogram(y=audio_sample, sr=self.fs)
        mel = librosa.power_to_db(mel, ref=np.max)
        return mel

    def create_mel_spectrogram_from_audio(self, audio_sample):
        mel = librosa.feature.melspectrogram(y=audio_sample, sr=self.fs)
        mel = librosa.power_to_db(mel, ref=np.max)
        return mel


def populate_numpy_on_disk(np_filename_x,np_filename_y, file_list, args, scaler=None):

    preprocess = Preprocess(args)
    augment = DataAugmentation(args)
    
    for i, f in enumerate(file_list):

        path, label, transform = f
        path = args.file_name + '/' + path
        audio_signal, _ = preprocess.trim_beginning(path)

        if not transform:
            mel = preprocess.create_mel_spectrogram_from_audio(audio_signal)
            
        elif transform == 'pitch':
            audio_signal = augment.pitch_shift(audio_signal)
            mel = preprocess.create_mel_spectrogram_from_audio(audio_signal)

        elif transform == 'forward':
            audio_signal = augment.forward_time_shift(audio_signal)
            mel = preprocess.create_mel_spectrogram_from_audio(audio_signal)

        elif transform == 'backwards':
            audio_signal = augment.backward_time_shift(audio_signal)
            mel = preprocess.create_mel_spectrogram_from_audio(audio_signal)

        elif transform == 'frequency':
            mel = preprocess.create_mel_spectrogram_from_audio(audio_signal)
            mel = augment.frequency_mask(mel)
        
        elif transform == 'time':
            mel = preprocess.create_mel_spectrogram_from_audio(audio_signal)
            mel = augment.time_mask(mel)

        else:
            mel = preprocess.create_mel_spectrogram_from_audio(audio_signal)

        np_filename_x[i, :, :] = mel
        np_filename_y[i, :] = label

    X = np_filename_x.copy()

    if not scaler:
        scaler = preprocessing.StandardScaler()
        scaler.fit(X.reshape(-1, X.shape[-1]))
        
    
    X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    np_filename_x[:,:,:] = X[:,:,:]
    np_filename_x.flush()
    np_filename_y.flush()
    return scaler



def process_data(args):

    file_names = os.listdir(args.file_name+'/')

    pos_paths = [(x, 1, None) for x in file_names if 'wakeword' in x]
    neg_paths = [(x, 0, None) for x in file_names if 'background' in x]

    preprocessor = Preprocess(args)

    
    trim_audio, _ = preprocessor.trim_beginning(args.file_name + "/" + pos_paths[0][0])
    DATA_DIM = preprocessor.create_mel_spectrogram_from_audio(trim_audio).shape
    print("DATA_DIM", DATA_DIM)

    all_paths = pos_paths + neg_paths
    random.shuffle(all_paths)

    #train_test split
    train_files = all_paths[400:]
    test_files = all_paths[:400]

    N_TRAIN_SAMPLES = len(train_files)
    N_TEST_SAMPLES = len(test_files)

    augmented_data = []
    augment_types = ['pitch', 'forward', 'backward', 'frequency', 'time']

    STEP = N_TRAIN_SAMPLES // len(augment_types)

    cnt = 0
    for i in range(0, N_TRAIN_SAMPLES, STEP):
        #print(i+STEP)
        train_files[i:i+STEP]
        augment_files = [(x[0], x[1], augment_types[cnt]) for x in train_files[i:i+STEP]]
        augmented_data.append(augment_files)
        print(len(train_files[i:i+STEP]), augment_types[cnt])
        cnt +=1
        if cnt >= len(augment_types):
            break

    '''

    print('prev_train_len', len(train_files))

    for augment in augmented_data:
        print(augment[:10])
        train_files +=augment
    print('after augment', len(train_files))
    '''

    files = []
    X_train_file = "./data_processed/X_train.dat"
    y_train_file = "./data_processed/y_train.dat"
    X_test_file = "./data_processed/X_test.dat"
    y_test_file = "./data_processed/y_test.dat"
    files.append(X_train_file)
    files.append(y_train_file)
    files.append(X_test_file)
    files.append(y_test_file)

    for file in files:
        if os.path.exists(file):
            os.remove(file)
    if not os.path.exists("data_processed"):
        os.mkdir("data_processed")

    N_TRAIN_SAMPLES = len(train_files)
    
    X_train = np.memmap(X_train_file, dtype='float32', mode='w+', shape=(N_TRAIN_SAMPLES,DATA_DIM[0], DATA_DIM[1]))
    y_train = np.memmap(y_train_file, dtype='int', mode='w+', shape=(N_TRAIN_SAMPLES, 1))
    X_test = np.memmap(X_test_file, dtype='float32', mode='w+',shape=(N_TEST_SAMPLES, DATA_DIM[0], DATA_DIM[1]))
    y_test = np.memmap(y_test_file, dtype='int', mode='w+', shape=(N_TEST_SAMPLES, 1))
    
    scaler = populate_numpy_on_disk(X_train,y_train, train_files, args)
    populate_numpy_on_disk(X_test, y_test, test_files, args, scaler)

    with open('data_processed/train_shape.txt',"w") as f:
        f.write(f"{N_TRAIN_SAMPLES},{DATA_DIM[0]},{DATA_DIM[1]}")

    with open('data_processed/test_shape.txt', "w") as f:
        f.write(f"{N_TEST_SAMPLES},{DATA_DIM[0]},{DATA_DIM[1]}")

    with open('scaler/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)


class AudioData(Dataset):
    
    def __init__(self, path, x_filename, y_filename, shape_file):

        with open(shape_file, 'r') as f:
            for line in f:
                shape = [int(x) for x in line.strip().split(",")]
                shape = tuple(shape)

        self.path = path
        self.x_filename = x_filename
        self.y_filename = y_filename
        self.n_samples = shape[0]
        self.X = np.memmap(f'{self.path}/{self.x_filename}', dtype='float32', mode='r', shape=shape)
        self.y = np.memmap(f'{self.path}/{self.y_filename}', dtype='int', mode='r', shape=(shape[0], 1))


    def __getitem__(self, index):
        X = self.X[index, :, :]
        y = self.y[index, :]
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        return X, y
    
    def __len__(self):
        return self.n_samples

def create_dataloader(batch_size=1):
    shape_file_train='data_processed/train_shape.txt'
    shape_file_test='data_processed/test_shape.txt'
    train_data = AudioData("data_processed", "X_train.dat", "y_train.dat", shape_file_train)
    test_data = AudioData("data_processed", "X_test.dat", "y_test.dat", shape_file_test)

    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_dataloader, test_dataloader


def main(args):
    
    if args.collect_data == "Y":
        collect_wakeword_data(args.seconds, args.sample_rate, args.num_pos_samples, args.file_name, args.num_channels)
        collect_ambient_data(args.seconds, args.sample_rate, args.num_neg_samples, args.file_name, args.num_channels)
    if args.parse_data == "Y":
        process_data(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parse user input arguments into function')
    parser.add_argument('--seconds',type=int, default=2, help='duration of recording')
    parser.add_argument('--sample_rate', type=int, default=44100, help='sample rate of recording')
    parser.add_argument('--num_pos_samples', type=int, default=200, help='number of positive label samples that will be recorded')
    parser.add_argument('--num_neg_samples', type=int, default=1000, help='number of negative label samples that will be recorded')
    parser.add_argument('--num_channels', type=int, default=1, help='number of channels that the recorded audio will contain')
    parser.add_argument('--file_name', type=str, default='./data', help='directory location of saved .wav wake word file')
    parser.add_argument('--collect_data', type=str, default='N', help='file will collect audio data from computer mic if set to Y')
    parser.add_argument('--parse_data', type=str, default='N', help='it set to Y, file will parse the data collected')
    args = parser.parse_args()
    main(args)