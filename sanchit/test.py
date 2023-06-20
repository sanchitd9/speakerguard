import pickle

with open("output.pickle", "br") as f:
    loaded_list = pickle.load(f)

print(loaded_list)
print(len(loaded_list))