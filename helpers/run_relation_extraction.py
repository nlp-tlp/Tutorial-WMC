import os
import json
from .SimpleMWOREModel import SimpleMWOREModel
from .prepare_relations import prepare_relations


def run_relation_extraction(tagged_bio_sents):
    RE_DATASET_PATH = "data/re_dataset"

    def load_re_dataset(filename: str) -> list:
        """Load the Relation Extraction dataset into a list.

        Args:
            filename (str): The name of the file to load.
        """
        re_data = []
        with open(filename, "r") as f:
            for row in f:
                re_data.append(row.strip().split(","))
        return re_data

    train_dataset = load_re_dataset(os.path.join(RE_DATASET_PATH, "train.csv"))

    relations, tagged_sents = prepare_relations(tagged_bio_sents)
    # rel_model = flair_re_model
    tagged_relations = tag_all_relations(relations)
    return tagged_relations, tagged_sents


def tag_all_relations(relations: list):
    """Run model inference over every potential relation in the list of
    relations.

    Args:
        relations(list): The list of (untagged) relations.

    Returns:
        tagged_relations(list): The same list, but with the rel_type in the
           8th column.

    """
    tagged_relations = []
    rel_model = SimpleMWOREModel()

    for rel in relations:
        tagged_rel = rel[:]
        rel_type = rel_model.inference(rel)
        tagged_rel[7] = rel_type
        tagged_relations.append(tagged_rel)
    return tagged_relations
