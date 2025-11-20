import torch
from torch.utils.data import Dataset

## --- DATA READING AND VOCABULARY GENERATION ---

def read_file(path, eos_token="<eos>"):
    """
    Reads the raw text file and appends an End-Of-Sentence (EOS) token to each line.

    :param path: File path to the corpus text (e.g., ptb.train.txt).
    :param eos_token: The token to mark the end of a sequence (Language Model standard).
    :return: A list of strings, where each string is a sentence with '<eos>' appended.
    """
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            # Add a space before the EOS token for correct tokenization later
            output.append(line.strip() + " " + eos_token)
    return output

def get_vocab(corpus, special_tokens=[]):
    """
    Creates a mapping from tokens (words) to unique integer IDs.
    This is used to build the complete vocabulary (vocab) from the training corpus.

    :param corpus: List of sentences (strings).
    :param special_tokens: List of tokens reserved for specific purposes (e.g., <pad>, <eos>).
    :return: Dictionary mapping word (str) to ID (int).
    """
    output = {}
    i = 0
    # Assign IDs to special tokens starting from 0 (conventionally <pad>=0)
    for st in special_tokens:
        output[st] = i
        i += 1
    # Assign IDs to all unique words in the corpus
    for sentence in corpus:
        for w in sentence.split():
            if w not in output:
                output[w] = i
                i += 1
    return output

class Lang():
    """
    A class to hold and manage the vocabulary mappings (Word -> ID and ID -> Word).
    """
    def __init__(self, corpus, special_tokens=[]):
        # Build the dictionary mapping words to integer IDs
        self.word2id = self.get_vocab(corpus, special_tokens)
        # Build the reverse dictionary mapping IDs back to words
        self.id2word = {v:k for k, v in self.word2id.items()}
    
    def get_vocab(self, corpus, special_tokens=[]):
        # Note: This method is functionally identical to the standalone get_vocab
        output = {}
        i = 0
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output

## --- PYTORCH DATASET CLASS ---

class PennTreeBank(Dataset):
    """
    Pytorch Dataset class for the Language Modeling task.
    It prepares the data for next-word prediction (source = sequence, target = sequence shifted by one).
    """
    def __init__(self, corpus, lang):
        self.source = [] # Input sequences (x_1, x_2, ..., x_{T-1})
        self.target = [] # Output sequences (x_2, x_3, ..., x_T)

        for sentence in corpus:
            tokens = sentence.split()
            # Source: All tokens except the last one
            self.source.append(tokens[0:-1]) 
            # Target: All tokens except the first one (the target for x_t is x_{t+1})
            self.target.append(tokens[1:]) 
            
        # Convert list of token strings into list of integer IDs
        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        # Total number of sentences/samples in the dataset
        return len(self.source)

    def __getitem__(self, idx):
        # Returns a single sample (a source sequence and its corresponding target sequence)
        # Converts sequences into PyTorch LongTensors (required for indices/IDs)
        src = torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample

    def mapping_seq(self, data, lang):
        """
        Maps a list of sequences (tokens) to their corresponding vocabulary IDs.
        """
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq:
                # check for OOV (Out-Of-Vocabulary) words
                if x in lang.word2id: 
                    tmp_seq.append(lang.word2id[x])
                else: 
                    break 
            res.append(tmp_seq)
        return res

## --- DATALOADER COLLATING FUNCTION ---

def collate_fn(data, pad_token):
    """
    A custom collating function used by the DataLoader to handle variable-length sequences.
    It pads sequences to the maximum length in the batch and prepares tensors for the model.

    :param data: List of samples (dictionaries) returned by __getitem__.
    :param pad_token: The integer ID used for padding (typically 0).
    :return: A dictionary containing padded source/target tensors and the total token count.
    """
    
    def merge(sequences):
        """Helper to pad sequences to the same length."""
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        
        # Create a matrix filled with PAD_TOKEN ID
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(pad_token)
        
        # Copy each sequence into the matrix up to its original length
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq
            
        return padded_seqs, lengths

    # Sort data by sequence length in descending order (required for padding efficiency)
    data.sort(key=lambda x: len(x["source"]), reverse=True)
    
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    # Pad source and target sequences
    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])

    # Assign padded tensors to the output dictionary
    new_item["source"] = source
    new_item["target"] = target
    
    # Calculate the total number of non-padding tokens in the batch (used for normalization)
    new_item["number_tokens"] = sum(lengths)
    
    return new_item