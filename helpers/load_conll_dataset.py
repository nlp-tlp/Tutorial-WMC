import os

def to_conll_document(s: str):
    """Parse a CONLL-formatted document into a dictionary of
    tokens and labels.

    Args:
        s (str): A string, separated by newlines, where each
        line is a token, then a space, then a label.

    Returns:
        dict: A dict of tokens and labels.
    """
    tokens, labels = [], []
    for line in s.split("\n"):
        if len(line.strip()) == 0:
            continue
        token, label = line.split()

        tokens.append(token)
        labels.append(label)
    return {'tokens': tokens, 'labels': labels}


def load_conll_dataset(filename: str) -> list:
    """Load a list of documents from the given CONLL-formatted dataset.

    Args:
        filename (str): The filename to load from.

    Returns:
        list: A list of documents, where each document is a dict of tokens and labels.
    """
    documents = []
    with open(filename, "r") as f:
        docs = f.read().split("\n\n")
        for d in docs:
            if len(d) == 0:
                continue
            document = to_conll_document(d)
            documents.append(document)
    print(f"Loaded {len(documents)} documents from {filename}.")
    return documents

