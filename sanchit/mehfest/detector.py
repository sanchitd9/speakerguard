import librosa
import numpy as np

class Mehfest:

    def __init__(self, n_fft=512, low_index=224, hop_length=160, window_length=400):
        self.n_fft = n_fft
        self.low_index = low_index
        self.hop_length = hop_length
        self.window_length = window_length
        self.colors = ["b", "g", "r", "c", "m", "y", "k", "lime", "pink", "gold", "olive", "skyblue", "thistle"]
    
    def calculate_min_energy_stft(self, audio):
        magnitudes = np.abs(librosa.stft(audio[0], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.window_length, center=False))
        energy = sum(np.square(magnitudes[self.low_index:,]))
        min_energy = min(energy)
        index_min_energy = min(range(len(energy)), key=energy.__getitem__)

        return min_energy, index_min_energy
    
    def calculate_kth_min_energy_stft(self, audio, k):
        magnitudes = np.abs(librosa.stft(audio[0], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.window_length, center=False))
        energy = sum(np.square(magnitudes[self.low_index:,]))
        index = sorted(range(len(energy)), key=lambda x: energy[x])

        len_index = len(energy)
        index_set = [index[0]]
        cur_index = 1
        found = False

        if len(index_set) == k:
            found = True
        
        while (not found) and (cur_index < len_index):
            found_cur = True
            len_index_set = len(index_set)

            for i in range(len_index_set):
                if abs(index[cur_index] - index_set[i]) <= (self.n_fft // self.hop_length):
                    found_cur = False
                    break
            
            if found_cur:
                index_set.append(index[cur_index])
                if len(index_set) == k:
                    found = True
            
            cur_index += 1
        
        if found:
            index_kth_min_energy = index_set[-1]
        else:
            print(f"Warning: Could not find the {k}-th minimum energy in the STFT!")
            index_kth_min_energy = index[len_index - 1]
        
        kth_min_energy = energy[index_kth_min_energy]

        return kth_min_energy, index_kth_min_energy
