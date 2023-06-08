import json

def bio_to_mention(bio_doc: dict):
    """Return a Mention-format representation of a BIO-formatted
    tagged sentence.

    Args:
        bio_doc (dict): The BIO doc to convert to the Mention-based doc.

    Returns:
        dict: A mention-formatted dict created from the bio_doc.
    """
    tokens = bio_doc["tokens"]
    labels = bio_doc["labels"]
    mentions_list = []

    start = 0
    end = 0
    label = None
    for i, (token, label) in enumerate(
        zip(tokens, labels)
    ):
        if label.startswith("B-"):
            if len(mentions_list) > 0:
                mentions_list[-1]["end"] = i
            mentions_list.append({"start": i, "labels": [label[2:]]})
        elif label == "O" and len(mentions_list) > 0:
            mentions_list[-1]["end"] = i
        if len(mentions_list) == 0:
            continue
        if i == (len(tokens) - 1) and "end" not in mentions_list[-1]:
            mentions_list[-1]["end"] = i + 1
            
    for m in mentions_list:
        m['phrase'] = " ".join(tokens[m['start']:m['end']])
    return {'tokens': tokens, 'mentions': mentions_list}


# Let's print our example sentence again, this time with the mention-based
# representation.
# We'll use json.dumps to make it a bit easier to read.
#print(json.dumps(tagged_sents[15],indent=1))

def build_potential_relations(tagged_sents) -> list:
    """Build a list of potential relations, i.e. all possible relationships
    between each entity in each document. The 8th column (which denotes the
    relationship type) will be set to None. The 9th column is the document index.
    
    Args:
        tagged_sents(list): The list of tagged sentences, where each sentence is a
            dict of tokens: [list of tokens] and mentions: [list of mentions].
    
    Returns:
        list: A list of rows, where each row is a potential relationship.
    """

    relations = []
    for doc_idx, doc in enumerate(tagged_sents):
        for m1_idx, mention_1 in enumerate(doc['mentions']):
            entity_1 = " ".join(doc['tokens'][mention_1['start']: mention_1['end']])
            label_1 = mention_1['labels'][0]

            for m2_idx, mention_2 in enumerate(doc['mentions']):
                if m1_idx == m2_idx:
                    continue
                entity_2 = " ".join(doc['tokens'][mention_2['start']: mention_2['end']])
                label_2 = mention_2['labels'][0]
                mention_text = " ".join(doc['tokens'][mention_1['start']:mention_2['end']]   )         

                relations.append(
                    [entity_1, entity_2, label_1, label_2, mention_text, m1_idx, m2_idx, None, doc_idx]         
                )
    return relations

def prepare_relations(tagged_bio_sents):
    tagged_sents = []
    for doc in tagged_bio_sents:
        mention_doc = bio_to_mention(doc)
        tagged_sents.append(mention_doc)     
    relations = build_potential_relations(tagged_sents)
    return relations, tagged_sents
