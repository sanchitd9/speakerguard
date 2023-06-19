# Extend from detection.py to consider to apply k-th min energy 
# in high frequencies of STFT as a detector

import numpy as np
import os
import librosa
import sys
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-test_dir', default='../FAKEBOB/data/test-set')
parser.add_argument('-illegal_dir', default='../FAKEBOB/data/illegal-set')
parser.add_argument('-adv_dir', required=True, default=None)
parser.add_argument('-arch', default="iv_plda", choices=["iv_plda", "xv_plda"])

args = parser.parse_args()

# test_dir = "./data/test-set"
# illegal_dir = "./data/illegal-set"
# adversarial_dir = "adversarial-audio"
# gmm_adversarial_dir = "gmm-SV-targeted_epsilon_" + sys.argv[2]
# iv_adversarial_dir = "iv-SV-targeted_epsilon_" + sys.argv[2]
test_dir = args.test_dir
illegal_dir = args.illegal_dir
adversarial_dir = args.adv_dir

# spk_id_list = ["1580", "2830", "4446", "5142", "61"]
#spk_id_list = os.listdir(args.adv_dir)
spk_id_list = os.listdir(args.test_dir)

# archi = "iv" or "gmm"
# archi = sys.argv[1]
archi = args.arch

# k_extend = int(sys.argv[3])

n_fft = 512  # main parameter for STFFT (4096)
low_index = 224 # indicate the high frequency range and depending on n_fft (when n_fft = 512, the value of 224 refers to 7KHz)!
hop_length = 160 # frame step = 10ms 
win_length = 400 # frame length = 25ms 

def calculate_min_energy_stft(audio_path):
    audio = librosa.load(audio_path, sr=16000)
    magnitudes = np.abs(librosa.stft(audio[0], n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False))
    energy = sum(np.square(magnitudes[low_index:,]))
    print(energy)
    min_energy = min(energy)
    index_min_energy = min(range(len(energy)), key=energy.__getitem__)

    return min_energy, index_min_energy

# def calculate_kth_min_energy_stft(audio_path) :
#     audio = librosa.load(audio_path, sr=16000)
#     magnitudes = np.abs(librosa.stft(audio[0], n_fft=n_fft, 
#                         hop_length=hop_length, win_length=win_length,
#                         center=False))
#     energy = sum(np.square(magnitudes[low_index:,]))
#     index = sorted(range(len(energy)), key=lambda k: energy[k])
    
#     # attempt to find the index for k-th min energy
#     # need to begin from 2nd one and go through the same process until k-th one
#     len_index = len(energy)
#     index_set = [index[0]]   # put 1st min energy index
#     cur_index = 1

#     found = False 
#     if len(index_set) == k_extend :
#         found = True

#     while (not found) and (cur_index < len_index):
#         found_cur = True 
#         len_index_set = len(index_set)

#         for i in range(0, len_index_set): 
#             if abs(index[cur_index] - index_set[i]) <= (n_fft // hop_length):
#                 found_cur = False 
#                 break
        
#         if found_cur :
#             index_set.append(index[cur_index])
#             if len(index_set) == k_extend :
#                 found = True 

#         cur_index += 1 

#     if found :
#         index_kth_min_energy = index_set[-1]        
#     else :
#         print("Warning: cannot find the %d-th min energy in STFT" % (k_extend))
#         index_kth_min_energy = index[len_index - 1]
    
#     print(index_set)

#     kth_min_energy = energy[index_kth_min_energy]

#     return kth_min_energy, index_kth_min_energy  

f = open("output_values.txt", "a")


# get all illegal audios 
illegal_audio_list = []
illegal_audio_names = []
spk_iter = os.listdir(illegal_dir)
for spk_id in spk_iter:
    spk_dir = os.path.join(illegal_dir, spk_id)
    audio_iter = os.listdir(spk_dir)
    for _, audio_name in enumerate(audio_iter):
        path = os.path.join(spk_dir, audio_name)
        illegal_audio_list.append(path)
        illegal_audio_names.append(audio_name)

''' Results for illegal audios
'''
print("Results for illegal audios:\n")
illegal_audio_results = []
for i, illegal_audio_path in enumerate(illegal_audio_list):
    energy, index = calculate_min_energy_stft(illegal_audio_path)

    print("    audio name = %s, index = %d, energy = %f" % (illegal_audio_names[i], index, energy))
    illegal_audio_results.append(energy)

f.write("%s\n\n" % illegal_audio_results)
print("")

''' Results for each registered user (legal audios and adversarial audios)
'''
legal_audio_results = []
adversarial_audio_results = []
for spk_id in spk_id_list:    # loop through speakers
    print("spk = %s\n" % (spk_id))

    # get audio list of this speaker 
    audio_list = []
    audio_names = []
    spk_dir = os.path.join(test_dir, spk_id)
    audio_iter = os.listdir(spk_dir)
    for _, audio_name in enumerate(audio_iter):
        path = os.path.join(spk_dir, audio_name)
        audio_list.append(path)
        audio_names.append(audio_name)

    ''' Results for legal audios
    '''
    print("  Results for legal audios:\n")
    for i, legal_audio_path in enumerate(audio_list):
        energy, index = calculate_min_energy_stft(legal_audio_path)

        print("    audio name = %s, index = %d, energy = %f" % (audio_names[i], index, energy))
        legal_audio_results.append(energy)
    print("")

    # get adversarial audios of this speaker 
    # if archi == "gmm":
    #     target_dir = gmm_adversarial_dir
    # elif archi == "iv_plda":
    #     target_dir = iv_adversarial_dir
    # else:
    #     print("Error on getting a archi!")
    #     exit(1) 

    target_dir = adversarial_dir

    # GMM or i-vector adversarial audios     
    adv_list = []
    adv_audio_names = []
    # adv_dir = os.path.join(adversarial_dir, target_dir)
    # adv_dir = os.path.join(adv_dir, spk_id)
    # adv_dir = os.path.join(adversarial_dir, spk_id)
    adv_dir = adversarial_dir

    adv_iter = os.listdir(adv_dir)

    for _, adv_spk_id in enumerate(adv_iter):
        #print(adv_spk_id)
        #sys.exit(0)
        adv_audio_dir = os.path.join(adv_dir, adv_spk_id)
        adv_audio_iter = os.listdir(adv_audio_dir)
        for _, adv_audio_name in enumerate(adv_audio_iter):
            path = os.path.join(adv_audio_dir, adv_audio_name) 
            adv_list.append(path)
            adv_audio_names.append(adv_audio_name) 

    ''' Results for adversarial audios
    '''
    print("  Results for adversarial audios:\n")
    for i, adv_audio_path in enumerate(adv_list):
        energy, index = calculate_min_energy_stft(adv_audio_path)

        print("    audio name = %s, index = %d, energy = %f" % (adv_audio_names[i], index, energy))
        adversarial_audio_results.append(energy)
    print("")
    print("")

f.write("%s\n\n" % legal_audio_results)
f.write("%s\n\n" % adversarial_audio_results)

f.close()
# print("Final results for the %d-th min energy in STFT:\n" % (k_extend))

print("illegal audios: len = %d, max = %f, min = %f, average = %f" % (len(illegal_audio_results), max(illegal_audio_results), min(illegal_audio_results), sum(illegal_audio_results)/len(illegal_audio_results)))
illegal_audio_results.sort()
print(illegal_audio_results)
print("")

print("legal audios: len = %d, max = %f, min = %f, average = %f" % (len(legal_audio_results), max(legal_audio_results), min(legal_audio_results), sum(legal_audio_results)/len(legal_audio_results)))
legal_audio_results.sort()
print(legal_audio_results)
print("")

print("adversarial audios: len = %d, max = %f, min = %f, average = %f" % (len(adversarial_audio_results), max(adversarial_audio_results), min(adversarial_audio_results), sum(adversarial_audio_results)/len(adversarial_audio_results)))
adversarial_audio_results.sort()
print(adversarial_audio_results)
print("")
