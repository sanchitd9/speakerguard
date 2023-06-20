import numpy as np
import os
import librosa
import argparse
import pickle

# Set up arguments
parser = argparse.ArgumentParser()
parser.add_argument("-dir", required=True)
parser.add_argument("-out", default="output.pickle")
args = parser.parse_args()

# Fetch the arguments
audio_dir = args.dir
output_file = args.out

# STFT Parameters
n_fft      = 512
low_index  = 224
hop_length = 160
win_length = 400

# Function to calculate the minimum energy in the STFT
def calculate_min_energy_stft(audio_path):
    audio = librosa.load(audio_path, sr=16000)
    magnitudes = np.abs(librosa.stft(audio[0], n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False))
    energy = sum(np.square(magnitudes[low_index:,]))

    print(energy)

    min_energy = min(energy)
    index_min_energy = min(range(len(energy)), key=energy.__getitem__)

    return min_energy, index_min_energy

# Get the list of speakers
speakers = os.listdir(audio_dir)
audio_path_list = []
audio_name_list = []

# Populate the list of audios
for id in speakers:
    speaker_dir = os.path.join(audio_dir, id)
    audio_list = os.listdir(speaker_dir)

    for _, file_name in enumerate(audio_list):
        path = os.path.join(speaker_dir, file_name)
        audio_path_list.append(path)
        audio_name_list.append(file_name)
    
# Get the results for each audio
results = []
for i, file_path in enumerate(audio_path_list):
    energy, index = calculate_min_energy_stft(file_path)
    print(f"Name = {audio_name_list[i]}, Index = {index}, Energy = {energy}")
    results.append(energy)

# Save the list to a file
with open(output_file, "wb") as f:
    pickle.dump(results, f)

# Print the final result
print("Final Result:")
print(f"Length = {len(results)}, Max = {max(results)}, Min = {min(results)}, Mean = {sum(results) / len(results)}")
