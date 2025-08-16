import logging
import numpy as np
import joblib
import json
import pandas as pd
from preprocess_utils import preprocess_title

logging.basicConfig(level=logging.INFO)


def run_manual_test(inputs_string: list[str], model_path = "../../classification/models/full_pipeline_model.pkl"):

    pipeline = joblib.load(model_path)

    logging.info("vectorizing the part names")
    embeddings = pipeline["embedder"].transform(inputs_string)

    results = pipeline["classifier"].predict(embeddings)

    logging.info("Results:")
    for i, res in enumerate(results):
        logging.info(f"  output[{i}]: value={res}")

    return results

# Example test
if __name__ == "__main__" :
    # print("Input: Carte mère H81")
    # output = run_manual_test("../../classification/models/logistic_regression_model.onnx", ["Carte mère H81".lower()])

    # print("Input: Motherboard H81")
    # output = run_manual_test("../../classification/models/logistic_regression_model.onnx", ["Motherboard H81".lower()])

    # print("Input: Arctic Liquid Freezer III Pro 360 A-RGB BLACK")
    # output = run_manual_test("../../classification/models/logistic_regression_model.onnx", ["Arctic Liquid Freezer III Pro 360 A-RGB BLACK".lower()])

    # print("Input: Clés USB publicitaires ")
    # output = run_manual_test("../../classification/models/logistic_regression_model.onnx", ["Clés USB publicitaires".lower()])

    # print("Input: COMPTEUSE TRIEUSE DE BILLETS RIBAO")
    # output = run_manual_test("../../classification/models/logistic_regression_model.onnx", ["COMPTEUSE TRIEUSE DE BILLETS RIBAO".lower()])
    # Read from json
    with open('./PcPartPickerDB.parts.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.read_json('./PcPartPickerDB.parts.json', encoding='utf-8')
    
    # keep the parts with sitename Campus Informatique

    df = df[df['siteName'] == 'Campus Informatique']

    # Run the model on the rows and add new column called predicted category and save the results into a new json {id, title, cateogry, predictedCategory}
    results = []
    for index, row in df.iterrows():
        title = row['title']
        category = row['category']
        preprocessed_title = preprocess_title(title)
        # predicted_category = run_manual_test([preprocessed_title['normalized_title'].lower()])
        predicted_category = run_manual_test([title.lower()])
        results.append({
            "id": index,
            "title": title,
            "category": category,
            "predictedCategory": predicted_category[0],
            "preProcessedAttrs": preprocessed_title
        })
    
    print(results)

    # Save the results into a new json file
    with open('./PcPartPickerDB.parts.predictions-min-preprocess.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)