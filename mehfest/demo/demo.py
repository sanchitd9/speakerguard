# %% [markdown]
# # Demo to show the idea of the Min Energy in High FrEquencies for Short Time (MEH-FEST) detection method

# %%
import numpy as np
import matplotlib.pyplot as plt
import IPython
import librosa
import librosa.display

def plot_time_domain_waveform(audio, Fs, title, file_name, 
                              start_time=0):    
    t_wav = np.arange(audio.shape[0]) / Fs + start_time
    print ("Fs = %d, audio signal length = %d" % (Fs, audio.shape[0]))
    
    plt.plot(t_wav, audio, color='r')
    plt.xlim(t_wav[0], t_wav[-1])
    plt.title(title)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    
    plt.savefig(file_name, dpi = 300)
    plt.show()
    
    plt.close()

def plot_spectrum_magnitude(audio, Fs, n_fft, hop_length, win_length,
                            low_index, title, file_name):
    magnitudes = np.abs(librosa.stft(audio, 
                                     n_fft=n_fft, 
                                     hop_length=hop_length,
                                     win_length=win_length,
                                     center=False))
    
    energy = sum(np.square(magnitudes[low_index:]))
    print(energy)
    
    freq = np.arange(n_fft/2 + 1) * Fs / n_fft
    plt.plot(freq, magnitudes, 'b')
    plt.xlim(freq[0], freq[-1])
    plt.title(title)
    plt.xlabel('Frequence (Hz)', fontsize=12)
    plt.ylabel('Magnitude', fontsize=12)
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    
    plt.savefig(file_name, dpi = 300)
    plt.show()
    
    plt.close()
    
    print("mean = %f, std = %f" % (np.mean(magnitudes),
                                   np.std(magnitudes)))
        
def plot_energy_stft(audio_path, n_fft, hop_length, win_length,
                     low_index, title, file_name):
    audio = librosa.load(audio_path, sr=16000)
    magnitudes = np.abs(librosa.stft(audio[0], 
                                     n_fft=n_fft, 
                                     hop_length=hop_length, 
                                     win_length=win_length,
                                     center=False))
    energy = sum(np.square(magnitudes[low_index:,]))
    print(energy.shape)
    print(min(energy))
    
    time_frame = np.arange(len(energy))
    plt.plot(time_frame, energy, 'xr-')
    plt.xlim(time_frame[0], time_frame[-1])
    plt.title(title)
    plt.xlabel('Time frame', fontsize=12)
    plt.ylabel('Energy in high frequencies', fontsize=12)
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    
    plt.savefig(file_name, dpi = 300)
    plt.show()
    
    plt.close()

def plot_energy_stft_both(org_audio_path, adv_audio_path, n_fft,
                          hop_length, win_length, low_index, 
                          x_max, title, file_name):
    audio = librosa.load(org_audio_path, sr=16000)
    magnitudes = np.abs(librosa.stft(audio[0], 
                                     n_fft=n_fft, 
                                     hop_length=hop_length,
                                     win_length=win_length,
                                     center=False))
    energy = sum(np.square(magnitudes[low_index:,]))
    print(min(energy))
    time_frame = np.arange(len(energy))    
    plt.plot(time_frame, energy, 'xr:', label='Original audio')
    
    audio = librosa.load(adv_audio_path, sr=16000)
    magnitudes = np.abs(librosa.stft(audio[0], 
                                     n_fft=n_fft, 
                                     hop_length=hop_length,
                                     win_length=win_length,
                                     center=False))
    energy = sum(np.square(magnitudes[low_index:,]))
    print(min(energy))
    time_frame = np.arange(len(energy))
    plt.plot(time_frame, energy, '+b-', label='Adversarial audio')
    
    plt.xlim(time_frame[0], x_max)
    plt.yscale('log')
    plt.title(title)
    plt.xlabel('Time frame', fontsize=12)
    plt.ylabel('Energy in high frequencies', fontsize=12)
    plt.tick_params(axis='both', labelsize=12)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(file_name, dpi = 300)
    plt.show()
    
    plt.close()


def plot_mel_spectrogram(audio, Fs, n_fft, hop_length, win_length,
                         n_mels, specmin, specmax, title, file_name):
    mel_spec = librosa.feature.melspectrogram(audio, 
                                              sr=Fs, 
                                              n_fft=n_fft, 
                                              hop_length=hop_length,
                                              win_length=win_length,
                                              n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec)
    librosa.display.specshow(log_mel_spec, x_axis='time',
                             y_axis='mel', sr=Fs,
                             hop_length=hop_length,
                             cmap='coolwarm',
                             vmin=specmin, vmax=specmax)
    
    plt.xlabel('Time (seconds)',fontsize=12)
    plt.ylabel('Frequency (Hz)',fontsize=12)
    
    cbar = plt.colorbar(format='%+2.f dB')
    cbar.set_label(label='Magnitude (dB)',size=12)
    cbar.ax.tick_params(labelsize=12)
    plt.tick_params(labelsize=12)
    plt.title(title, fontsize=12)
    plt.tight_layout()
    
    plt.savefig(file_name, dpi = 300)
    plt.show()
    
    plt.close()
    
def calculate_min_energy_stft(audio_path, n_fft, hop_length,
                              win_length, low_index) :
    audio = librosa.load(audio_path, sr=16000)
    magnitudes = np.abs(librosa.stft(audio[0], 
                                      n_fft=n_fft, 
                                      hop_length=hop_length,
                                      win_length=win_length,
                                      center=False))
    energy = sum(np.square(magnitudes[low_index:,]))
    print(energy)
    min_emergy = min(energy)
    index_min_emergy = min(range(len(energy)), 
                         key = energy.__getitem__)
    return min_emergy, index_min_emergy 

# %% [markdown]
# The original audio is from: ~/Desktop/research/detection_fakebob/experiments/origin_audios/data/illegal-set/6829/6829-68769-0016.wav  

# %%
org_audio_path = '6829-68769-0016-org.wav'
IPython.display.Audio(org_audio_path)

# %% [markdown]
# The adversarial audio is from: ~/Desktop/research/detection_fakebob/experiments/gmm_epsilon_002/gmm-SV-targeted/4446/6829/6829-68769-0016.wav,
# which is with a perturbation threshold of 0.002 in the GMM SV.
# The FakeBob only took 1 iteration to generate this adversarial audio.

# %%
adv_audio_path = '6829-68769-0016-adv.wav'
IPython.display.Audio(adv_audio_path)

# %% [markdown]
# # Time domain waveforms

# %%
org_audio, Fs = librosa.load(org_audio_path, sr = 16000)
plot_time_domain_waveform(org_audio, Fs, "Original Audio", 
                          "./figs/time_waveform_org_audio.png")

# %%
adv_audio, Fs = librosa.load(adv_audio_path, sr = 16000)
plot_time_domain_waveform(adv_audio, Fs, "Adversarial Audio", 
                          "./figs/time_waveform_adv_audio.png")

# %%
difference = adv_audio - org_audio
plot_time_domain_waveform(difference, Fs, 
                          "Differences Between Adversarial and Original Audios", 
                          "./figs/time_waveform_differences.png")
print("mean = %f, std = %f" % (np.mean(difference), np.std(difference)))

# %% [markdown]
# # Mel Spectrograms
# Note: the sample frequency is 16000 Hz. The range of frequency is between 0 and 8000 Hz. In the mel spectrograms, y-axis is in the log scale, so that 8000 Hz is not shown in the y-axis (the next number to 4096 is 8192).

# %%
plot_mel_spectrogram(org_audio, Fs=16000, n_fft = 512, 
                     hop_length = 160, win_length = 400, 
                     n_mels = 128, 
                     specmin = -70, specmax = 20, 
                     title = "Original Audio", 
                     file_name = "./figs/mel_spectrogram_org_audio.png")

# %%
plot_mel_spectrogram(adv_audio, Fs=16000, n_fft = 512, 
                     hop_length = 160, win_length = 400, 
                     n_mels = 128, 
                     specmin = -70, specmax = 20, 
                     title = "Adversarial Audio", 
                     file_name = "./figs/mel_spectrogram_adv_audio.png")

# %%
plot_mel_spectrogram(difference, Fs=16000, n_fft = 512, 
                     hop_length = 160, win_length = 400,
                     n_mels = 128, 
                     specmin = -70, specmax = 20, 
                     title = "Differences Between Adversarial and Original Audios", 
                     file_name = "./figs/mel_spectrogram_differences.png")

# %% [markdown]
# # MEH-FEST Detection Method

# %%
print(calculate_min_energy_stft(org_audio_path, 
                                n_fft = 512, 
                                hop_length = 160,
                                win_length = 400,
                                low_index = 224))
print(calculate_min_energy_stft(adv_audio_path, 
                                n_fft = 512, 
                                hop_length = 160,
                                win_length = 400,
                                low_index = 224))

# %%
print(512/16000)
print(398*160/16000)
print((398*160+512)/16000)
print(350*160/16000)
print((350*160+512)/16000)

# %% [markdown]
# Look into specific time period: 398 * 160 / 16000 ~ (398 * 160 + 512) / 16000 = 3.98 s ~ 4.012 s.
# As a comparason, show another time period: 3.5 s ~ 3.532 s at the end.

# %%
n_fft = 512
index = 398
hop_length = 160

org_audio_short_time = org_audio[hop_length * index : 
                                 hop_length * index + n_fft]
plot_time_domain_waveform(org_audio_short_time, Fs, "Original Audio", 
                          "./figs/time_waveform_org_audio_short_time.png",
                          hop_length * index / Fs)

print(min(org_audio_short_time))
print(max(org_audio_short_time))

# %%
adv_audio_short_time = adv_audio[hop_length * index : 
                                 hop_length * index + n_fft]
plot_time_domain_waveform(adv_audio_short_time, Fs, "Adversarial Audio", 
                          "./figs/time_waveform_adv_audio_short_time.png",
                          hop_length * index / Fs)

print(min(adv_audio_short_time))
print(max(adv_audio_short_time))

# %%
difference_short_time = adv_audio_short_time - org_audio_short_time
plot_time_domain_waveform(difference_short_time, Fs, 
                          "Differences Between Adversarial and Original Audios", 
                          "./figs/time_waveform_differences_short_time.png",
                          hop_length * index / Fs)
print("mean = %f, std = %f" % (np.mean(difference_short_time), 
                               np.std(difference_short_time)))

# %% [markdown]
# Spectrum of short-time signals (both original and adversarial audios):

# %%
plot_spectrum_magnitude(org_audio_short_time, Fs = 16000, 
                        n_fft = 512, hop_length = 512, 
                        win_length = 400, low_index = 224, 
                        title = "Spectrum of original audio in short time", 
                        file_name = "./figs/spectrum_org_audio_short_time.png")

# %%
plot_spectrum_magnitude(adv_audio_short_time, Fs=16000, 
                        n_fft = 512, hop_length = 512, 
                        win_length = 400, low_index = 224,  
                        title = "Spectrum of adversarial audio in short time", 
                        file_name = "./figs/spectrum_adv_audio_short_time.png")

# %%
plot_spectrum_magnitude(difference_short_time, Fs=16000, 
                        n_fft = 512, hop_length = 512, 
                        win_length = 400, low_index = 224,  
                        title = "Spectrum of perturbations in short time", 
                        file_name = "./figs/spectrum_perturbation_audio_short_time.png")


# %% [markdown]
# Another short time signals between 3.5 s ~ 3.532 s.

# %%
n_fft = 512
index = 350
hop_length = 160

org_audio_short_time = org_audio[hop_length * index : 
                                 hop_length * index + n_fft]
plot_time_domain_waveform(org_audio_short_time, Fs, "Original Audio", 
                          "./figs/time_waveform_org_audio_short_time_2.png",
                          hop_length * index / Fs)

print(min(org_audio_short_time))
print(max(org_audio_short_time))

# %%
adv_audio_short_time = adv_audio[hop_length * index : 
                                 hop_length * index + n_fft]
plot_time_domain_waveform(adv_audio_short_time, Fs, "Adversarial Audio", 
                          "./figs/time_waveform_adv_audio_short_time_2.png",
                          hop_length * index / Fs)

print(min(adv_audio_short_time))
print(max(adv_audio_short_time))

# %%
difference_short_time = adv_audio_short_time - org_audio_short_time
plot_time_domain_waveform(difference_short_time, Fs, 
                          "Differences Between Adversarial and Original Audios", 
                          "./figs/time_waveform_differences_short_time_2.png",
                          hop_length * index / Fs)
print("mean = %f, std = %f" % (np.mean(difference_short_time), 
                               np.std(difference_short_time)))

# %% [markdown]
# Show the energy in high frequencies based on time frames:

# %%
plot_energy_stft(org_audio_path, 
                 n_fft = 512, 
                 hop_length = 160,
                 win_length = 400,
                 low_index = 224, 
                 title = "Energy in high frequencies for original aduio", 
                 file_name = "./figs/hf-stft_org_audio.png")

# %%
plot_energy_stft(adv_audio_path, 
                 n_fft = 512, 
                 hop_length = 160,
                 win_length = 400,
                 low_index = 224, 
                 title = "Energy in high frequencies for adversarial aduio", 
                 file_name = "./figs/hf-stft_adv_audio.png")

# %%
plot_energy_stft_both(org_audio_path, 
                      adv_audio_path, 
                      n_fft = 512, 
                      hop_length = 160,
                      win_length = 400,
                      low_index = 224, 
                      x_max = 530,
                      title = "Energy in high frequencies for both audios", 
                      file_name = "./figs/hf-stft_both.png")

# %%



