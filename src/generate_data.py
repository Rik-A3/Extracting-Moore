import os
import numpy as np
import torch

from src.MooreOracle import TransformerOracle
from src.TargetMachines import get_target_moore, accepting_states
from extraction.Moore import Moore

from torch.utils.data import Dataset

torch.set_warn_always(False)

def remove_garbage_state(moore: Moore, F):
    moore = moore.copy()

    states = set(moore.Q)

    rejecting_states = states - set(F)

    while rejecting_states:
        s = rejecting_states.pop()
        is_garbage = True
        for a in moore.delta[s]:
            if moore.delta[s][a] != s:
                is_garbage = False
                break
        if is_garbage and s != moore.q0:
            moore.Q.remove(s)
            moore.delta.pop(s)
            moore.X.pop(s)
            for s_ in moore.delta:
                to_remove = []
                for a_  in moore.delta[s_]:
                    if moore.delta[s_][a_] == s:
                        to_remove.append(a_)
                for a_ in to_remove:
                    moore.delta[s_].pop(a_)
                if to_remove and s_ not in F:
                    rejecting_states.add(s_)
    return moore

"""
Generate `N` random sequences of `length` by drawing uniformly at random from the alphabet
No duplicates will only be guaranteed if the flag is set 

If N is larger than the number of unique sequences of `length`, the returned dataset will be smaller
"""
def generate_random(length, N, alphabet=["0","1"], no_duplicates=False):
    data = []

    if no_duplicates:
        data = [""]
        curr_length = 0

        while True:
            if curr_length >= length:
                return data
            new_data = []
            if len(data) >= N:
                for seq in data:
                    seq += np.random.choice(alphabet)
                    new_data.append(seq)
                data = new_data
                curr_length += 1
            else:
                for i, seq in enumerate(data):
                    if len(new_data) + (len(data) - i) >= N:
                        seq += np.random.choice(alphabet)
                        new_data.append(seq)
                    else:
                        for char in alphabet:
                            new_data.append(seq + char)
                            if len(new_data) + (len(data) - i - 1) >= N:
                                break
                data = new_data
                curr_length += 1

    else:
        for _ in range(N):
            seq = ""
            for _ in range(length):
                seq += np.random.choice(alphabet)
            data.append(seq)
    
    return data

"""
Generates all sequences of `length` that can be constructed from the alphabet
"""
def generate_all(length, alphabet=["0","1"]):
    return generate_random(length, len(alphabet)**length, alphabet, no_duplicates=True)

def generate_N_positive_seqs(seq, curr_q, target_moore, accepting_states, length, N, acc):
    # base case
    if length == 0 and curr_q in accepting_states:
        acc.append(seq)
        return acc
    elif length == 0:
        return acc
    
    chars = list(target_moore.delta[curr_q].keys())
    np.random.shuffle(chars)
    for char in chars:
        acc = generate_N_positive_seqs(seq + char, target_moore.delta[curr_q][char], target_moore, accepting_states, length-1, N, acc)

        if len(acc) >= N:
            return acc
    
    return acc

"""
Generate sequences that end in accepting states
"""
def generate_positive_only(target_moore, accepting_states, length, N, no_duplicates=False):
    target_moore = remove_garbage_state(target_moore, accepting_states)
    data = []
    
    if no_duplicates:
        data = generate_N_positive_seqs("", target_moore.q0, target_moore, accepting_states, length, N, [])
        while len(data) < N:
            data += data[:min(N - len(data), len(data))] # Make copies if there are less than N unique sequences
    else:
        for _ in range(N):
            data += generate_N_positive_seqs("", target_moore.q0, target_moore, accepting_states, length, 1, [])

    return data


"""
Generate sequences such that every reachable state (is part of a loop) is represented equally in the dataset (all label classes will be equal)

Works only for binary alphabet and allows duplicates
"""
def generate_balanced(target_moore, length, N, no_duplicates=False):
    def extend(seq, curr_q, target_moore, counts):
        c1 = counts[target_moore.delta[curr_q][target_moore.alphabet[0]]]
        c2 = counts[target_moore.delta[curr_q][target_moore.alphabet[1]]]
        seq += np.random.choice(target_moore.alphabet, p=[(c2+1)/(c1+c2+2), (c1+1)/(c1+c2+2)])
        new_q = target_moore.delta[curr_q][seq[-1]]
        counts[new_q] += 1

        return seq, new_q, counts
    
    assert len(target_moore.alphabet) == 2, "Balanced method only works for binary alphabet"
    
    if no_duplicates:
        return NotImplementedError("Balanced method does not support no_duplicates=True")

    counts = {q: 0 for q in target_moore.Q}  # counts for each transition
    data = []

    for _ in range(N):
        seq = ""
        curr_state = target_moore.q0
        for _ in range(length):
            seq, curr_state, counts = extend(seq, curr_state, target_moore, counts)
        data.append(seq)

    print(f"Balanced state counts ({length},{N}):", counts)
    
    return data

def label_data_oracle(data, oracle: TransformerOracle):
    labels = []
    for d in data:
        labels.append(oracle.label_word(d))
    return labels

"""
Label `data` according to the output function of the `target_moore` machine
"""
def label_data(data, target_moore: Moore):
    labels = []
    for d in data:
        labels.append(target_moore.label_word(d))
    return labels

def data_to_tensor(data, bos_token):
    # add BOS
    data = [bos_token + d for d in data]

    # data to list
    data = [[int(c) for c in word] for word in data]

    # to tensors
    idxs = torch.randperm(torch.tensor(data, dtype=torch.float32).shape[0])       

    data = torch.tensor(data, dtype=torch.long)[idxs]    # [ batch_size * seq_len ]

    return data

"""
Make batched tensors out of data and labels and add <BOS> token + <BOS> label

Returns:
- (nb_sequences * sequence_length, nb_sequences * sequence_length)
"""
def to_tensors(data, labels, bos_label, bos_token="2"):
    # add BOS
    data = [bos_token + d for d in data]
    labels = [[bos_label] + l for l in labels]

    # data to list
    data = [[int(c) for c in word] for word in data]

    # to tensors
    idxs = torch.randperm(torch.tensor(data, dtype=torch.float32).shape[0])       

    data = torch.tensor(data, dtype=torch.long)[idxs]    # [ batch_size * seq_len ]
    labels = torch.tensor(labels, dtype=torch.long)[idxs]    # [ batch_size * seq_len ]

    assert data.shape == labels.shape, "data and labels must have the same shape [batch_size * sequence length]"

    return data, labels

def get_dataset_name(language_name, length, N, method, no_duplicates, train : bool, data : bool, test: bool = False, oracle_name: str = None):
    if method == "all":
        return ("test-" if test else ("train-" if train else "val-")) + ("data-" if data else (f"{oracle_name}-" if oracle_name else "labels-")) + f"l{length}-N{N}-{method}-no_dup={no_duplicates}.pkl"
    return ("test-" if test else ("train-" if train else "val-")) + ("data-" if data else (f"{oracle_name}-" if oracle_name else "labels-")) + f"{language_name}-l{length}-N{N}-{method}-no_dup={no_duplicates}.pkl"

class MooreDataset(Dataset):
    def __init__(self, data: torch.TensorType, targets: torch.TensorType, device="cpu"):
        self.device = device
        self.data = data          # [nb_sequences * sequence_length] : torch.long range(0, d_vocab)
        self.targets = targets     # [nb_sequences * sequence_length] : torch.long range(0,d_vocab_out)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, :].to(self.device), self.targets[idx, :].to(self.device)
"""
language_name : Name of target moore machine defined in TargetMachines
length : Length of examples
N : Number of examples (in the train + test set)
train : Whether to return the train or test partition
method : Generate data random, all, positive_only or balanced
no_duplicates : Whether to allow duplicates
train_frac : Fraction of N to put in the train set
"""
def get_dataset(language_name, length, N, train : bool, test=False, method="random", no_duplicates=False, device="cpu") -> MooreDataset:
    SAVE_DIR = "/data/"

    data_file_name = get_dataset_name(language_name=language_name, length=length, N=N, method=method, no_duplicates=no_duplicates, train=train, test=test, data=True)
    labels_file_name = get_dataset_name(language_name=language_name, length=length, N=N, method=method, no_duplicates=no_duplicates, train=train, test=test, data=False)

    if os.path.exists(SAVE_DIR + data_file_name) and os.path.exists(SAVE_DIR + labels_file_name):
        data = torch.load(SAVE_DIR  + data_file_name)
        labels = torch.load(SAVE_DIR + labels_file_name)

        return data, labels
    
    if method == "random":
        target_moore = get_target_moore(language_name)
        data = generate_random(length,N, no_duplicates=no_duplicates, alphabet=target_moore.alphabet)
        labels = label_data(data, target_moore)
        data, labels = to_tensors(data, labels, target_moore.X[target_moore.q0], str(len(target_moore.alphabet)))

    elif method == "all":
        target_moore = get_target_moore(language_name)
        if os.path.exists(SAVE_DIR + data_file_name):
            data = torch.load(SAVE_DIR  + data_file_name)
        else:
            data = generate_all(length, alphabet=target_moore.alphabet)
        labels = label_data(data, target_moore)
        data, labels = to_tensors(data, labels, target_moore.X[target_moore.q0], str(len(target_moore.alphabet)))

    elif method == "positive_only":
        target_moore = get_target_moore(language_name)
        data = generate_positive_only(target_moore, accepting_states[language_name], length, N, no_duplicates)
        labels = label_data(data, target_moore)
        data, labels = to_tensors(data, labels, target_moore.X[target_moore.q0], str(len(target_moore.alphabet)))
        
    elif method == "balanced":
        target_moore = get_target_moore(language_name)
        data = generate_balanced(target_moore, length, N, no_duplicates)
        labels = label_data(data, target_moore)
        data, labels = to_tensors(data, labels, target_moore.X[target_moore.q0], str(len(target_moore.alphabet)))

    elif method == "character_prediction":
        target_moore = get_target_moore(language_name, training_task="character_prediction")
        data = generate_balanced(target_moore, length, N, no_duplicates)
        labels = label_data(data, target_moore)
        data, labels = to_tensors(data, labels, target_moore.X[target_moore.q0], str(len(target_moore.alphabet)))

    elif method == "membership_prediction":
        target_moore = get_target_moore(language_name, training_task="membership_prediction")
        data = generate_balanced(target_moore, length, N, no_duplicates)
        labels = label_data(data, target_moore)
        data, labels = to_tensors(data, labels, target_moore.X[target_moore.q0], str(len(target_moore.alphabet)))

    else:
        raise NotImplementedError(f"Method {method} not implemented")

    torch.save(data, os.getcwd() + SAVE_DIR + data_file_name)
    torch.save(labels, os.getcwd() + SAVE_DIR + labels_file_name)

    return MooreDataset(data, labels, device=device)

def get_dataset_from_oracle(oracle: TransformerOracle, language_name, length, N, train : bool, test=False, method="random", no_duplicates=False, train_frac=0.9) -> MooreDataset:
    SAVE_DIR = "/data/"

    if not test and train:
        N = int(train_frac*N)
    else:
        N = N - int((train_frac)*N)

    data_file_name = get_dataset_name(language_name=language_name, length=length, N=N, method=method, no_duplicates=no_duplicates, train=train, test=test, data=True)
    labels_file_name = get_dataset_name(language_name=language_name, length=length, N=N, method=method, no_duplicates=no_duplicates, train=train, test=test, data=False, oracle_name=oracle.full_cfg["name"])

    save_data = True

    if os.path.exists(SAVE_DIR + data_file_name):
        data = torch.load(SAVE_DIR  + data_file_name)
        if os.path.exists(SAVE_DIR + labels_file_name):
            labels = torch.load(SAVE_DIR + labels_file_name)
            return data, labels
        save_data = False
    else:
        target_moore = get_target_moore(language_name, training_task=method)
        
        if method == "random":
            data_words = generate_random(length,N, no_duplicates=no_duplicates, alphabet=target_moore.alphabet)
            data = data_to_tensor(data_words, str(len(target_moore.alphabet)))

        if method == "all":
            data_words = generate_all(length, alphabet=target_moore.alphabet)
            data = data_to_tensor(data, str(len(target_moore.alphabet)))

        if method == "positive_only":
            data_words = generate_positive_only(target_moore, accepting_states[language_name], length, N, no_duplicates, alphabet=target_moore.alphabet)
            data = data_to_tensor(data, str(len(target_moore.alphabet)))
            
        if method == "balanced":
            target_moore = get_target_moore(language_name)
            data = generate_balanced(target_moore, length, N, no_duplicates)
            labels = label_data(data, target_moore)
            data, labels = to_tensors(data, labels, target_moore.X[target_moore.q0], str(len(target_moore.alphabet)))

        if method == "character_prediction":
            target_moore = get_target_moore(language_name, training_task="character_prediction")
            data = generate_balanced(target_moore, length, N, no_duplicates)
            labels = label_data(data, target_moore)
            data, labels = to_tensors(data, labels, target_moore.X[target_moore.q0], str(len(target_moore.alphabet)))

        if method == "membership_prediction":
            target_moore = get_target_moore(language_name, training_task="membership_prediction")
            data = generate_balanced(target_moore, length, N, no_duplicates)
            labels = label_data(data, target_moore)
            data, labels = to_tensors(data, labels, target_moore.X[target_moore.q0], str(len(target_moore.alphabet)))

    
    # includes bos-tokens label
    labels = label_data_oracle(data, oracle)
    labels = torch.tensor(labels, dtype=torch.long)    # [ batch_size * seq_len ]

    assert labels.shape == data.shape

    if save_data:
        torch.save(data, os.getcwd() + SAVE_DIR + data_file_name)
    torch.save(labels, os.getcwd() + SAVE_DIR + labels_file_name)

    return MooreDataset(data, labels)