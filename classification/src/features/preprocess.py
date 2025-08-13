import pandas as pd
from ..utils.tokenization import spacy_tokenize
import logging

logging.basicConfig(level=logging.INFO)

def preprocess_data():
    df = pd.read_csv('../shared/data/interim/pc-parts-merged.csv')
    
    logging.info("Lowering the part names")
    # lower case all the pc_part names
    df['part_name'] = df['part_name'].str.lower()

    # remove the punctuations replace with space
    logging.info("Removing the punctuations")
    df['part_name'] = df['part_name'].str.replace('[^\w\s]', ' ', regex=True)

    # tokenize the part names
    logging.info("Tokenizing the part names")
    tokenized_part_names = spacy_tokenize(df['part_name'].tolist())
    # tokenized_part_names = df['part_name'].str.split(" ")

    df['part_name_tokenized'] = tokenized_part_names

    logging.info("Removing duplicate tokens")
    # remove the duplicate tokens
    df['part_name_tokenized'] = df['part_name_tokenized'].apply(lambda x: list(dict.fromkeys(x)))

    logging.info("Joining the tokens back to a string")
    # join the tokens back to a string
    df['part_name'] = df['part_name_tokenized'].apply(lambda x: ' '.join(x))

    logging.info("Dropping the tokenized column")
    # drop the tokenized column
    df.drop(columns=['part_name_tokenized'], inplace=True)

    logging.info("Saving the preprocessed data")
    # save the preprocessed data
    df.to_csv('../shared/data/processed/pc-parts-processed.csv', index=False)


if __name__ == "__main__":
    preprocess_data()
    logging.info("Data preprocessing completed.") 