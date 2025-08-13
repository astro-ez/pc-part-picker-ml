import logging
import numpy as np
import joblib

logging.basicConfig(level=logging.INFO)

def run_manual_test(model_path, inputs_string: list[str]):

    pipeline = joblib.load('../../classification/models/full_pipeline_model.pkl')

    logging.info("embedding the part names")
    embeddings = pipeline["embedder"].encode(inputs_string, show_progress_bar=True)

    results = pipeline["classifier"].predict(embeddings)

    logging.info("Results:")
    for i, res in enumerate(results):
        logging.info(f"  output[{i}]: value={res}")

    return results

# Example test
if __name__ == "__main__" :
    print("Input: Carte mère H81")
    output = run_manual_test("../../classification/models/logistic_regression_model.onnx", ["Carte mère H81".lower()])

    print("Input: Motherboard H81")
    output = run_manual_test("../../classification/models/logistic_regression_model.onnx", ["Motherboard H81".lower()])

    print("Input: Arctic Liquid Freezer III Pro 360 A-RGB BLACK")
    output = run_manual_test("../../classification/models/logistic_regression_model.onnx", ["Arctic Liquid Freezer III Pro 360 A-RGB BLACK".lower()])

    print("Input: Clés USB publicitaires ")
    output = run_manual_test("../../classification/models/logistic_regression_model.onnx", ["Clés USB publicitaires".lower()])

    print("Input: COMPTEUSE TRIEUSE DE BILLETS RIBAO")
    output = run_manual_test("../../classification/models/logistic_regression_model.onnx", ["COMPTEUSE TRIEUSE DE BILLETS RIBAO".lower()])

