import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class SentimentDataset(Dataset):
    def __init__(self, data):
        self.texts = data["text"].values
        self.labels = data["label"].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        return torch.tensor(text, dtype=torch.long), torch.tensor(
            label, dtype=torch.long
        )


def create_dataset(path, delimiter=","):
    df = pd.read_csv(path, delimiter=delimiter, header=None, names=["label", "text"])

    def preprocess_text(text):
        return text.lower().split()

    df["text"] = df["text"].apply(preprocess_text)
    df = df[["text", "label"]]

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["label"])

    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

    vocab = set([word for phrase in df["text"] for word in phrase])
    word_to_idx = {word: idx for idx, word in enumerate(vocab, 1)}

    def encode_phrase(phrase):
        return [word_to_idx[word] for word in phrase]

    train_data["text"] = train_data["text"].apply(encode_phrase)
    test_data["text"] = test_data["text"].apply(encode_phrase)

    max_length = max(df["text"].apply(len))

    def pad_sequence(seq, max_length):
        return seq + [0] * (max_length - len(seq))

    train_data["text"] = train_data["text"].apply(lambda x: pad_sequence(x, max_length))
    test_data["text"] = test_data["text"].apply(lambda x: pad_sequence(x, max_length))

    return train_data, test_data, vocab


def create_dataloader():
    url = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"
    train_data, test_data, vocab = create_dataset(url, delimiter="\t")

    train_dataset = SentimentDataset(train_data)
    test_dataset = SentimentDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader, vocab
