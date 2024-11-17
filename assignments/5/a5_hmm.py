import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import librosa
sys.path.append(os.path.abspath(os.path.join('..','..')))
from models.HMM.hmm import HMM

path_to_spoken_digit_dataset = "../../data/external/spoken_digit_dataset"

def extract_mfcc(file_path, n_mfcc=13):
    audio, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfccs.T

def get_digits_mfcc_data(path_to_spoken_digit_dataset):
    digit_mfcc_accum = {str(i): [] for i in range(10)}

    files = os.listdir(path_to_spoken_digit_dataset)
    for file_name in files:
        if file_name.endswith(".wav"):
            digit = file_name.split('_')[0]
            file_path = os.path.join(path_to_spoken_digit_dataset, file_name)
            mfccs = extract_mfcc(file_path)
            digit_mfcc_accum[digit].append(mfccs)

    return digit_mfcc_accum

digit_mfcc_accum = get_digits_mfcc_data(path_to_spoken_digit_dataset)

# plot mfcc for each digit
fig, axes = plt.subplots(5, 2, figsize=(15, 10))
axes = axes.ravel()

for i, (digit, mfcc_list) in enumerate(digit_mfcc_accum.items()):
    if mfcc_list:
        # a random number(0 to 299 - 300 audios per digit) is chosen here to display the mfcc of the digit  
        sample_mfcc = mfcc_list[25].T
        ax = axes[i]
        img = librosa.display.specshow(sample_mfcc, x_axis='time', ax=ax, cmap='hot')
        plt.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title(f"MFCC for Digit {digit}")
        ax.set_xlabel("Time")
        ax.set_ylabel("MFCC Coefficients")

plt.tight_layout()
plt.show()

# for every digit plot 3 random mfccs
fig, axes = plt.subplots(10, 3, figsize=(30, 40))
axes = axes.ravel()

for i, (digit, mfcc_list) in enumerate(digit_mfcc_accum.items()):
    if mfcc_list:
        for j in range(3):
            sample_mfcc = mfcc_list[np.random.randint(0, len(mfcc_list))].T
            ax = axes[i*3+j]
            img = librosa.display.specshow(sample_mfcc, ax=ax, cmap='hot')
            plt.colorbar(img, ax=ax, format="%+2.0f dB")
            ax.set_title(f"MFCC for Digit {digit}")
            # ax.set_xlabel("Time")
            ax.set_ylabel("MFCC Coefficients")

plt.tight_layout()
plt.show()

def split_train_test(digit_mfcc_accum):    
    train_data = {}
    test_data = {}

    for digit, mfcc_list in digit_mfcc_accum.items():
        n = len(mfcc_list)
        n_train = int(0.8 * n)
        np.random.shuffle(mfcc_list)
        train_data[digit] = mfcc_list[:n_train]
        test_data[digit] = mfcc_list[n_train:]

    return train_data, test_data

train_data, test_data = split_train_test(digit_mfcc_accum)


hmm_model = HMM()

hmm_model.train(train_data)

def evaluate_model(model, test_data):
    preds = []
    true_labels = []
    for digit, mfcc_list in test_data.items():
        pred = model.predict(mfcc_list)
        preds.extend(pred)
        true_labels.extend([int(digit)] * len(pred))
    return true_labels, preds

true_labels, preds = evaluate_model(hmm_model, test_data)


accuracy = np.mean(np.array(true_labels) == np.array(preds))

print(f"Accuracy: {accuracy*100:.2f}%")

path_to_my_recorded_digits = "../../data/external/my_recorded_digits"

def test_my_recorded_digits(data_path=path_to_my_recorded_digits, model=hmm_model):
    targets = []
    preds = []
    for file_name in os.listdir(data_path):
        if file_name.endswith(".wav"):
            digit = file_name.split('.')[0]
            file_path = os.path.join(data_path, file_name)
            mfccs = extract_mfcc(file_path)
            pred = model.predict([mfccs])
            preds.extend(pred)
            targets.extend([int(digit)] * len(pred))
    return targets, preds

targets, preds = test_my_recorded_digits()

print("Targets:", targets)
print("Predictions:", preds)

accuracy = np.mean(np.array(targets) == np.array(preds))
print(f"Accuracy: {accuracy*100:.2f}%")