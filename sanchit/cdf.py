import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

# Set up arguments
parser = argparse.ArgumentParser()
parser.add_argument("-input", required=True)
args = parser.parse_args()

# Fetch the arguments
input_file = args.input

def preprocess(detections):
    n = len(detections)
    print(f"Size = {n}, Max = {np.max(detections)}, Min = {np.min(detections)}, Mean = {np.mean(detections): 10.8f}, STD = {np.std(detections): 10.8f}, D = {np.mean(detections) + 3*np.std(detections): 10.8f}")

    index = np.zeros(n)
    results = np.zeros(n)

    for i, result in enumerate(detections):
        index[i] = i + 1
        results[i] = result

    index /= n
    results *= 10000

    return index, results

def plot_cdf(detections):
    results, index = preprocess(detections)

    plt.plot(results, index)
    plt.xlabel(r"Detections scores $\mathrm{x 10^{4}}$", fontsize=12)
    plt.ylabel("CDF", fontsize=12)
    plt.tick_params(axis="both", labelsize=12)
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_cdf_original_audio(train_data, test_data, all_data):
    train_index, train_results = preprocess(train_data)
    test_index, test_results = preprocess(test_data)
    all_index, all_results = preprocess(all_data)

    plt.plot(train_results, train_index, "r-", label="Training Data")
    plt.plot(test_results, test_index, "b:", label="Test Data")
    plt.plot(all_results, all_index, "k--", label="All Original Audios")

    plt.xlabel(r"Energy in high frequencies (i.e., $E$) $\mathrm{x 10^{4}}$", fontsize=12)
    plt.ylabel("CDF", fontsize=12)
    plt.tick_params(axis="both", labelsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_cdf_adversarial_audio(all_original_detections, adversarial_detections, attack_name):
    original_index, original_results = preprocess(all_original_detections)
    adversarial_index, adversarial_results = preprocess(adversarial_detections)

    plt.plot(original_results, original_index, "k--", label="All Original Audios")
    plt.plot(adversarial_results, adversarial_index, "r-", label=f"{attack_name}")

    plt.xlabel(r'Energy in high frequencies (i.e., $E$) $\mathrm{x 10^{4}}$', fontsize=12)
    plt.ylabel('CDF', fontsize=12)
    plt.tick_params(axis='both', labelsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()

def split_train_test(data, test_ratio):
    # the code refers to https://github.com/ageron/handson-ml2/blob/master/02_end_to_end_machine_learning_project.ipynb
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data[train_indices], data[test_indices]


with open(input_file, "rb") as f:
    detections = pickle.load(f)

detections.sort()
# plot_cdf(detections)


