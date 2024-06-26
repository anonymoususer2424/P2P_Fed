import os
import torch

from PIL import Image
from datasets.nlp_utils.util import Tokenizer, Vocab
from torch.utils.data import Dataset
from pathlib import Path
from torch.utils.data import Dataset

class Sent140Dataset(Dataset):
    def __init__(self,
                 client_id: int,
                 client_str: str,
                 data: list,
                 targets: list,
                 is_to_tokens: bool = True,
                 tokenizer: Tokenizer = None):
        """get `Dataset` for sent140 dataset
        Args:
            client_id (int): client id
            client_str (str): client name string
            data (list): sentence list data
            targets (list): next-character target list
            is_to_tokens (bool, optional), if tokenize data by using tokenizer
            tokenizer (Tokenizer, optional), tokenizer
        """
        self.client_id = client_id
        self.client_str = client_str
        self.data = data
        self.targets = targets
        self.data_token = []
        self.data_tokens_tensor = []
        self.targets_tensor = []
        self.tokenizer = tokenizer if tokenizer else Tokenizer()

        self._process_data_target()
        if is_to_tokens:
            self._data2token()

    def _process_data_target(self):
        """process client's data and target
        """
        self.data = [e[4] for e in self.data]
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def _data2token(self):
        assert self.data is not None
        for sen in self.data:
            self.data_token.append(self.tokenizer(sen))

    def encode(self, vocab: 'Vocab', fix_len: int):
        """transform token data to indices sequence by `Vocab`
        Args:
            vocab (fedlab_benchmark.leaf.nlp_utils.util.vocab): vocab for data_token
            fix_len (int): max length of sentence
        Returns:
            list of integer list for data_token, and a list of tensor target
        """
        if len(self.data_tokens_tensor) > 0:
            self.data_tokens_tensor.clear()
            self.targets_tensor.clear()
        pad_idx = vocab.get_index('<pad>')
        assert self.data_token is not None
        for tokens in self.data_token:
            self.data_tokens_tensor.append(self.__encode_tokens(tokens, vocab, pad_idx, fix_len))
        for target in self.targets:
            self.targets_tensor.append(torch.tensor(target))

    def __encode_tokens(self, tokens, vocab, pad_idx, fix_len) -> torch.Tensor:
        """encode `fix_len` length for token_data to get indices list in `self.vocab`
        if one sentence length is shorter than fix_len, it will use pad word for padding to fix_len
        if one sentence length is longer than fix_len, it will cut the first max_words words
        Args:
            tokens (list[str]): data after tokenizer
            vocab  (fedlab_benchmark.leaf.nlp_utils.util.vocab): vocab for data_token
            pad_idx (int): '<pad>' index in vocab
            fix_len (int): max length of sentence
        Returns:
            integer list of indices with `fix_len` length for tokens input
        """
        x = [pad_idx for _ in range(fix_len)]
        for idx, word in enumerate(tokens[:fix_len]):
            x[idx] = vocab.get_index(word)
        return torch.tensor(x)

    def __len__(self):
        return len(self.targets_tensor)

    def __getitem__(self, item):
        return self.data_tokens_tensor[item], self.targets_tensor[item]


class CelebADataset(Dataset):
    def __init__(self,
                 client_id: int,
                 client_str: str,
                 data: list,
                 targets: list,
                 image_root: str,
                 transform=None):
        """get `Dataset` for CelebA dataset
         Args:
            client_id (int): client id
            client_str (str): client name string
            data (list): input image name list data
            targets (list):  output label list
        """
        self.client_id = client_id
        self.client_str = client_str
        self.image_root = Path(__file__).parent.resolve() / image_root
        self.transform = transform
        self.data = data
        self.targets = targets
        self._process_data_target()

    def _process_data_target(self):
        """process client's data and target
        """
        data = []
        targets = []
        for idx in range(len(self.data)):
            image_path = self.image_root / self.data[idx]
            image = Image.open(image_path).convert('RGB')
            data.append(image)
            targets.append(torch.tensor(self.targets[idx], dtype=torch.long))
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        data = self.data[index]
        if self.transform:
            data = self.transform(data)
        target = self.targets[index]
        return data, target


class ShakespeareDataset(Dataset):
    def __init__(self, client_id: int, client_str: str, data: list,
                 targets: list):
        """get `Dataset` for shakespeare dataset
        Args:
            client_id (int): client id
            client_str (str): client name string
            data (list): sentence list data
            targets (list): next-character target list
        """
        self.client_id = client_id
        self.client_str = client_str
        self.ALL_LETTERS, self.VOCAB_SIZE = self._build_vocab()
        self.data = data
        self.targets = targets
        self._process_data_target()

    def _build_vocab(self):
        """ according all letters to build vocab
        Vocabulary re-used from the Federated Learning for Text Generation tutorial.
        https://www.tensorflow.org/federated/tutorials/federated_learning_for_text_generation
        Returns:
            all letters vocabulary list and length of vocab list
        """
        ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
        VOCAB_SIZE = len(ALL_LETTERS)
        return ALL_LETTERS, VOCAB_SIZE

    def _process_data_target(self):
        """process client's data and target
        """
        self.data = torch.tensor(
            [self.__sentence_to_indices(sentence) for sentence in self.data])
        self.targets = torch.tensor(
            [self.__letter_to_index(letter) for letter in self.targets])

    def __sentence_to_indices(self, sentence: str):
        """Returns list of integer for character indices in ALL_LETTERS
        Args:
            sentence (str): input sentence
        Returns: a integer list of character indices
        """
        indices = []
        for c in sentence:
            indices.append(self.ALL_LETTERS.find(c))
        return indices

    def __letter_to_index(self, letter: str):
        """Returns index in ALL_LETTERS of given letter
        Args:
            letter (char/str[0]): input letter
        Returns: int index of input letter
        """
        index = self.ALL_LETTERS.find(letter)
        return index

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
        

class FemnistDataset(Dataset):
    def __init__(self, client_id: int, client_str: str, data: list,
                 targets: list):
        """get `Dataset` for femnist dataset
         Args:
            client_id (int): client id
            client_str (str): client name string
            data (list): image data list
            targets (list): image class target list
        """
        self.client_id = client_id
        self.client_str = client_str
        self.data = data
        self.targets = targets
        self._process_data_target()

    def _process_data_target(self):
        """process client's data and target
        """
        self.data = torch.tensor(self.data,
                                 dtype=torch.float32).reshape(-1, 1, 28, 28)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
