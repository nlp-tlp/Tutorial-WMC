""" A Flair-based relation extraction model.
This one uses Flair's TextClassifier model to classify the
relation type of a given row.
"""

import os
import json
from typing import List

import flair
from flair.trainers import ModelTrainer
from flair.datasets import CSVClassificationCorpus
from flair.embeddings import (
    PooledFlairEmbeddings,
    DocumentRNNEmbeddings,
)
from flair.data import Sentence
from typing import List
from flair.models import TextClassifier, SequenceTagger
from flair.visual.training_curves import Plotter

from huggingface_hub import hf_hub_download

import torch

MAX_EPOCHS = 1
HIDDEN_SIZE = 256

# Check whether CUDA is available and set the device accordingly
if torch.cuda.is_available():
    flair.device = torch.device("cuda:0")
else:
    flair.device = torch.device("cpu")
print("Device:", flair.device)


class FlairREModel(REModel):

    """The Flair-based RE model."""

    model_name: str = "Flair"

    def __init__(self):
        super(FlairREModel, self).__init__()
        self.model = None

    def train(self, datasets_path: os.path, trained_model_path: os.path):
        """Train the Flair RE model on the given CSV datasets.

        Args:
            datasets_path (os.path): The path containing the train and dev
               datasets.
            trained_model_path (os.path): The path to save the trained model.
        """

        column_name_map = {
            0: "text",
            1: "text",
            2: "text",
            3: "text",
            4: "text",
            7: "label_relation",
        }

        # Define corpus, labels, word embeddings, doc embeddings
        corpus = CSVClassificationCorpus(
            datasets_path,
            column_name_map,
            delimiter=",",
            label_type="relation",
        )

        label_dict = corpus.make_label_dictionary(label_type="relation")

        word_embeddings = [
            PooledFlairEmbeddings("mix-forward"),
            PooledFlairEmbeddings("mix-backward"),
        ]

        document_embeddings = DocumentRNNEmbeddings(
            word_embeddings, hidden_size=HIDDEN_SIZE
        )

        # Initialise sequence tagger
        tagger = TextClassifier(
            document_embeddings,
            label_dictionary=label_dict,
            label_type="relation",
        )

        # Initialize trainer
        trainer = ModelTrainer(tagger, corpus)

        sm = "cpu"
        if torch.cuda.is_available():
            sm = "gpu"
        
        # Start training
        trainer.train(
            trained_model_path,
            learning_rate=0.1,
            mini_batch_size=32,
            max_epochs=MAX_EPOCHS,
            patience=3,
            embeddings_storage_mode=sm,
        )

        self.load(os.path.join(trained_model_path, 'final-model.pt'))

    def load(self, model_path: str):
        """Load the model from the given path.

        Args:
            model_path (str): The filename containing the model.
               Can also be the name of a repo on Huggingface.
        """
        
        # We need to hard-code this one because TextClassifier doesn't
        # yet support HuggingFace.
        if(model_path == "nlp-tlp/mwo-re"):         
            model_path = hf_hub_download(
                repo_id="nlp-tlp/mwo-re",
                filename="pytorch_model.bin",
                cache_dir=flair.cache_root / "models" / "mwo-re"
            )
        
        self.model = TextClassifier.load(model_path)

    def inference(self, row: list) -> str:
        """Run the inference over the given document.

        Args:
            row (list): The row to predict the relation of.

        Returns:
            str: The relation type.
        """
        
        s = Sentence(" ".join(row[:5]))
        label = "O"
        self.model.predict(s)
        if len(s.labels) > 0:
            label = str(s.labels[0].value)
        return label