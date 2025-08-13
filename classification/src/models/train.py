import logging
from sklearn.model_selection import train_test_split 
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import yaml
# from skl2onnx import convert_sklearn
# from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from src.models.evaluate import evaluate_model
import joblib

def train():

    params = yaml.safe_load(open('params/train.yaml', 'r'))

    logging.info("Training model...")

    df = pd.read_csv('../shared/data/processed/pc-parts-processed.csv')

    sentence_model = SentenceTransformer(params['sentence_transformer_model'])

    logging.info("embedding the part names")
    embeddings = sentence_model.encode(df['part_name'].tolist(), show_progress_bar=True)

    X = embeddings
    y = df['part_type']
    logging.info("Splitting the data into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['test_size'], random_state=params['random_state'])

    logging.info("Initializing the logistic regression model")
    clf_logistic = LogisticRegression(max_iter=params['max_iter'], random_state=params['random_state'], penalty=params['penalty'], verbose=params['verbose'])

    logging.info("Fitting the logistic regression model")
    clf_logistic.fit(X_train, y_train)

    logging.info("Model training completed.")

    logging.info("Evaluating the model")
    evaluate_model(clf_logistic, X_test, y_test)

    pipeline = {
        "embedder": sentence_model,
        "classifier": clf_logistic
    }

    logging.info("Saving the trained model")
    with open('models/full_pipeline_model.pkl', 'wb') as f:
        joblib.dump(pipeline, f)
    
    logging.info("Model saved successfully.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting training pipeline")
    train()
    logging.info("Training pipeline completed.")
