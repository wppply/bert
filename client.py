"""
docker run -it --rm -p 8500:8500 -p 8501:8501 \
   -v "/Users/s0l04qa/Desktop/bert_export:/models/bert" \
   -e MODEL_NAME=bert \
   tensorflow/serving:latest
"""
import json
import os
import requests
import tokenization
import time
import pandas as pd
import numpy as np

class BertClient(object):
  def __init__(self, vocab_file, do_lower_case, max_seq_length=128, server_url="http://localhost:8501/v1/models", model_name="bert"):
    self.endpoints = "%s/%s:predict" % (server_url, model_name)
    self.headers = {"content-type": "application-json"}
    self.max_seq_length = max_seq_length
    self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

  def tokenize(self, query):
    """
    tokenize string sentence to vocab ids, an arr of int
    :param query: string
    :return: input_ids
    """
    token_a = self.tokenizer.tokenize(query)
    tokens = []
    tokens.append("[CLS]")
    segment_ids = []
    segment_ids.append(0)
    for token in token_a:
      tokens.append(token)
      segment_ids.append(0)
    tokens.append('[SEP]')
    segment_ids.append(0)
    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

    return input_ids

  def predict(self, query):
    """given query and return probs"""
    input_ids = self.tokenize(query)
    input_mask = [1] * len(input_ids)
    while len(input_ids) < self.max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
    segment_ids = [0] * len(input_ids)
    label_id = 0
    instances = [{"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids, "label_ids": label_id}]
    data = json.dumps({"signature_name": "serving_default", "instances": instances})

    response = requests.post(self.endpoints, data=data, headers=self.headers)
    prediction = json.loads(response.text)['predictions']
    return prediction

def performance_test(client):

  df = pd.read_csv("/Users/s0l04qa/Downloads/head_torso_2019_spk_w_tags 2.csv", sep=',', encoding='ISO-8859-1')
  chosen_idx = np.random.choice(230000, replace=False, size=1000)
  queries = df.iloc[chosen_idx]['query'].tolist()

  s_t = time.time()
  for query in queries:
    client.predict(query)
  avg_time = (time.time() - s_t) / len(queries)
  print("avg inference time: %s" % avg_time )

if __name__ == "__main__":
  client = BertClient("/Users/s0l04qa/Desktop/uncased_L-2_H-128_A-2/vocab.txt", False)
  prob = client.predict("good good study, day day up")
  print("Looks Good!", prob)

  performance_test(client)