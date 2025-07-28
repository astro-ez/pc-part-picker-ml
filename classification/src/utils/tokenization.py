import spacy

def spacy_tokenize(dataset: list[str]) -> list[list[str]]:
    """
    Tokenizes a list of sentences using spaCy.

    Args:
        dataset (list[str]): A list of sentences to tokenize.

    Returns:
        list[list[str]]: A list of tokenized sentences, where each sentence is a list of tokens.
    """
    nlp = spacy.load("en_core_web_sm")
    return [[token.text for token in nlp(sentence)] for sentence in dataset]