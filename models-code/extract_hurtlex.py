import pandas as pd
import csv
from tqdm import tqdm
import json
from nltk.tokenize import word_tokenize
import nltk
import argparse

nltk.download('punkt')

header_list = ["rewire_id", "text", "label_sexist", "label_category", "label_vector", "ps", "pa", "ddf", "ddp", "asf", "pr", "om", "qas", "cds", "asm"]

"""
Counts occurrence of each word of the example in the Hurtlex dictionary. 

Param:
df                        dataframe of examples to add features to
save_as_filename          if specified, the function will save each row to a csv file with specified name
"""
def get_hurtlex_counts(df, hurtlex, save_as_filename):
  data_with_hurtlex_counts = pd.DataFrame(columns=header_list)
  
  for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    sentence_count = pd.DataFrame(columns=header_list)
    for word in word_tokenize(row["text"]):
      if not hurtlex[hurtlex["lemma"]==word.lower()].empty:
        hurtlex_counts = hurtlex[hurtlex["lemma"]==word.lower()].category.value_counts()
        hurtlex_counts = hurtlex_counts.to_frame()
        hurtlex_counts = hurtlex_counts.rename(columns={"category": 0})

        new = pd.concat([row, hurtlex_counts],axis=1).T
        sentence_count = pd.concat([sentence_count, new])

    if sentence_count.empty:
      sentence_count = pd.DataFrame(row).T
      
    else:
      for col in ["ps","pa", "ddf", "ddp", "asf", "pr", "om", "qas", "cds", "asm"]:
          sentence_count[col] = sentence_count[col].sum()

    if save_as_filename:
      sentence_count.to_csv(save_as_filename, mode="a", header=False)

    data_with_hurtlex_counts = pd.concat([data_with_hurtlex_counts, sentence_count])

  data_with_hurtlex_counts = data_with_hurtlex_counts.drop_duplicates(subset="rewire_id")
  data_with_hurtlex_counts = data_with_hurtlex_counts.dropna(subset=["rewire_id"])
  return data_with_hurtlex_counts

"""
Formats output from get_hurtlex_counts function and saves it to a json file.

Param:
df                         dataframe with hurtlex counts to format
save_as_filename           the function will save the data to a json file with specified name
"""
def format_hurtlex_output(df, save_as_json):
  df = df.drop(columns=['an','or', 'dmc', 're', 'svp', 'is'])
  
  if 'rci' in df.columns:
    df = df.drop(columns=["rci"])

  id_only = df[["rewire_id", "ps","pa", "ddf", "ddp", "asf", "pr", "om", "qas", "cds", "asm"]]
  id_only = id_only.fillna(0)

  id_new = id_only.set_index("rewire_id")
  features = id_new.to_dict(orient="index")
  feature_list = id_new.values.tolist()
  ids = id_only["rewire_id"].tolist()
  features = {ids[i]: feature_list[i] for i in range(len(ids))}

  with open(save_as_json, "w") as outfile:
    json.dump(features, outfile)

def run_hurtlex_extraction(df, hurtlex, save_as_json, save_as_csv=None):
  hurtlex_extraction_not_formatted = get_hurtlex_counts(df, hurtlex, save_as_filename)
  format_hurtlex_output(hurtlex_extraction_not_formatted, save_as_json)

def create_arg_parser():
  parser = argparse.ArgumentParser()
  
  parser.add_argument("-d", "--data_file", default='../../../../data/train.csv', type=str,
                      help="Input file to add features to (default train.txt)")
    
  parser.add_argument("-hl", "--hurtlex_file", default='../../../../data/hurtlex_EN.tsv', type=str, 
                      help="Path to Hurtlex file (default in data folder)")
  
  parser.add_argument("--save_as_csv", default=None, type=str,
                      help="Path/filename to save features to in csv file")
  
  parser.add_argument("--save_as_json", default="hurtlex.json", type=str,
                      help="Path/filename to save features to in json file (default hurtlex.json)")
                      
  args = parser.parse_args()
  return args

def main():
  args = create_arg_parser()
  print(args)

  hurtlex_path = args.hurtlex_file
  hurtlex = pd.read_csv(args.hurtlex_file, sep="\t")
  data = pd.read_csv(args.data_file)
  run_hurtlex_extraction(data, hurtlex, args.save_as_csv, args.save_as_json)

main()