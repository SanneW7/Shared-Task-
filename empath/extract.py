import csv
import json
import time
import argparse
from pathlib import Path
from empath import Empath

def analyze_sent(lexicon, sent, categories):
    return lexicon.analyze(sent, categories=categories, normalize=True)

def extract_probs(input_dict, categories):
    probs = [input_dict[cat] for cat in categories]
    return probs

def get_features(lexicon, input_file, categories):
    # Get attribute probabilities of all training data
    empath_dict = dict()
    with open(input_file, mode='r') as input_f:
        input_data = csv.DictReader(input_f)
        for row in input_data:
            response = analyze_sent(lexicon, row['text'], categories)
            probs = extract_probs(response, categories)
            empath_dict[row['rewire_id']] = probs
    return empath_dict

def save_to_json(output_file, inp_dict):
    # Save attribute probabilities to a JSON file
    with open(output_file, "w") as outfile:
        json.dump(inp_dict, outfile, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", default='../data/dev_task_b_entries.csv', type=str,
                        help="Input file to learn from (default train.txt)")
    args = parser.parse_args()

    # List of used attributes
    categories = [ "sexism", "violence", "money", "valuable", "domestic work",
                 "hate", "aggression", "anticipation", "crime", "weakness",
                 "horror", "swearing terms", "kill", "sexual", "cooking",
                 "exasperation", "body", "ridicule", "disgust", "anger", "rage"]

    lexicon = Empath()
    input_filename = args.train_file
    empath_dict = get_features(lexicon, input_filename, categories)
    output_filename = f"empath_{Path(input_filename).stem}.json"
    save_to_json(input_filename.replace(Path(input_filename).name, output_filename),empath_dict)

if __name__ == "__main__":
    main()
