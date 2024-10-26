
import torch
import re
from torch.utils.data import Dataset, DataLoader

class IndexesDataset(Dataset):
    """
    IndexesDataset is a custom dataset class for preprocessing and handling text data for machine learning models.
    Attributes:
        samples (list): List of preprocessed text samples.
        labels (list): List of corresponding labels for the text samples.
        max_seq_len (int): Maximum sequence length for padding/truncating samples.
        device (str): Device to be used (e.g., "cuda" or "cpu").
        vocab (list): Vocabulary list containing unique words from the dataset.
    Methods:
        __init__(dataset, vocab=None, max_seq_len=64, device="cuda", max_samples_count=20000, max_vocab_size=30000):
            Initializes the dataset with given parameters and preprocesses the text samples.
        __len__():
            Returns the number of samples in the dataset.
        __getitem__(index):
            Returns the sample and label at the specified index, with the sample converted to indices and padded/truncated to max_seq_len.
        _make_vocab(max_vocab_size=30000):
            Creates a vocabulary from the samples, limited to max_vocab_size unique words.
        get_vocab():
            Returns the vocabulary list.
        get_index(word):
            Returns the index of the given word in the vocabulary. If the word is not found, returns the index of "[UNK]".
        collate(batch):
            Collates a batch of samples and labels into tensors for model input.
    """
    def __init__(self, dataset, vocab=None, max_seq_len=64, device="cuda", max_samples_count=20000,max_vocab_size=30000):
        """
        Initializes the data preprocessing class.
        Args:
            dataset (list): A list of tuples where each tuple contains a label and a sample text.
            vocab (dict, optional): A predefined vocabulary dictionary. Defaults to None.
            max_seq_len (int, optional): Maximum sequence length for the samples. Defaults to 64.
            device (str, optional): Device to be used for processing ('cuda' or 'cpu'). Defaults to "cuda".
            max_samples_count (int, optional): Maximum number of samples to process. Defaults to 20000.
            max_vocab_size (int, optional): Maximum size of the vocabulary. Defaults to 30000.
        Attributes:
            samples (list): List of processed sample texts.
            labels (list): List of corresponding labels for the samples.
            max_seq_len (int): Maximum sequence length for the samples.
            device (str): Device to be used for processing.
            vocab (dict): Vocabulary dictionary.
        """
        self.samples = []
        self.labels = []
        self.max_seq_len = max_seq_len
        self.device = device

        for row in dataset:
            label = row[0]
            sample = row[1]

            sample = re.sub(r"([.,!?'])", r" \1", sample)
            sample = re.sub(r"[^a-zA-Z0-9.,!?']", " ", sample)

            self.labels.append(int(label) - 1)
            self.samples.append(sample)

            if len(self.samples) > max_samples_count:
                break

        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = self._make_vocab(max_vocab_size=max_vocab_size)

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.samples)

    def __getitem__(self, index):
        """
        Retrieve the sample and label at the specified index.
        Args:
            index (int): The index of the sample to retrieve.
        Returns:
            tuple: A tuple containing the processed sample and its corresponding label.
                - sample (list): A list of indices representing the words in the sample, 
                  padded to `self.max_seq_len` with the index of the "[PAD]" token if necessary.
                - label: The label corresponding to the sample.
        """
        sample = self.samples[index]
        sample = [self.get_index(word) for word in sample.split()]

        sample = sample[:self.max_seq_len]
        pad_len = self.max_seq_len - len(sample)
        sample += [self.get_index("[PAD]")] * pad_len

        label = self.labels[index]

        return sample, label

    def _make_vocab(self, max_vocab_size=30000):
        """
        Constructs a vocabulary from the samples provided, with a maximum size limit.
        This method creates a vocabulary dictionary where each unique word in the samples
        is a key, and its value is the frequency of that word. The vocabulary is then sorted
        by frequency in descending order, and truncated to the specified maximum size.
        Special tokens "[PAD]" and "[UNK]" are added to the vocabulary with very high initial
        frequencies to ensure they are always included.
        Args:
            max_vocab_size (int): The maximum number of words to include in the vocabulary.
                      Default is 30,000.
        Returns:
            list: A list of words representing the vocabulary, sorted by frequency in descending order,
              truncated to the maximum size specified.
        """
        vocab = {"[PAD]": 1000000000000001, "[UNK]": 100000000000000}
        for sample in self.samples:
            for word in sample.split():
                if word not in vocab.keys():
                    vocab[word] = 1
                else:
                    vocab[word] += 1

        vocab = dict(sorted(vocab.items(), key=lambda item: item[1], reverse=True))

        vocab = list(vocab.keys())[:max_vocab_size]
        return vocab

    def get_vocab(self):
        """
        Retrieve the vocabulary.

        Returns:
            dict: The vocabulary dictionary.
        """
        return self.vocab

    def get_index(self, word):
        """
        Retrieve the index of a given word from the vocabulary.
        Args:
            word (str): The word to find the index for.
        Returns:
            int: The index of the word in the vocabulary. If the word is not found,
                 returns the index of the "[UNK]" token.
        """
        if word in self.vocab:
            index = self.vocab.index(word)
        else:
            index = self.vocab.index("[UNK]")

        return index

    def collate(self, batch):
        """
        Collates a batch of data into tensors.

        Args:
            batch (list of tuples): A list where each element is a tuple containing input_ids and targets.

        Returns:
            tuple: A tuple containing two tensors:
                - input_ids (torch.Tensor): Tensor of input IDs.
                - targets (torch.Tensor): Tensor of targets with dtype torch.float32.
        """
        input_ids, targets = list(zip(*batch))
        return torch.tensor(input_ids), torch.tensor(targets, dtype=torch.float32)

def get_loader(dataset_df, batch_size):
    """
    Creates a DataLoader for the given dataset.

    Args:
        dataset_df (Dataset): The dataset to load.
        batch_size (int): The number of samples per batch to load.

    Returns:
        DataLoader: A DataLoader instance for the dataset.
    """
    return DataLoader(dataset_df, batch_size=batch_size, num_workers=1, collate_fn=dataset_df.collate)