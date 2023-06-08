import os
import json
import pickle as pkl
from abc import ABC, abstractmethod


class NERModel(ABC):

    """Abstract base class for the NER Model."""

    def __init__(self):
        pass

    @abstractmethod
    def train(self, conll_datasets_path: str):
        pass

    @abstractmethod
    def inference(self, raw_sents: list):
        pass

    @abstractmethod
    def load(self, model_path):
        pass
