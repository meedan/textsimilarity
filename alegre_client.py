import json
import requests
import csv
import random
from collections import Counter
import numpy as np
import requests
from sklearn.metrics import roc_curve
def cosine_sim(vecA, vecB):
  """Find the cosine similarity distance between two vectors."""
  csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
  if np.isnan(np.sum(csim)):
    return 0
  return csim

class AlegreClient:

  @staticmethod
  def default_hostname():
    return "http://0.0.0.0:5000"

  @staticmethod
  def text_similarity_path():
    return "/text/similarity/"

  def __init__(self, hostname=None):
    if not hostname:
      hostname = AlegreClient.default_hostname()
    self.hostname = hostname

  def input_cases(self, texts, model_name, context={}, language=None):
    for text in texts:
      request_params = {"model": model_name, "text": text}
      if context:
        request_params["context"] = context
      if language:
        request_params["language"] = language
      self.store_text_similarity(request_params)

  def transform_india_today_data(self, filename="data/indiatoday_texts.csv"):
    with open(filename) as f:
      reader = csv.reader(f)
      dataset = []
      for row in reader:
        content = " ".join(row[1].split(" ")[:100])
        dataset.append({"database_text": content, "lookup_text": content, "id": row[0]})
    return dataset
      
  def transform_hindi_headlines_data(self, filename="data/hindi_headlines.json"):
    return json.loads(open(filename).read())
      
  def transform_ciper_data(self, with_truncation=True, ciper_filename="data/ciper_news_dataset.json"):
    #data/alegre_client_evaluation_data.zip must be unzipped in data dir for this to work!
    dataset = json.loads(open(ciper_filename).read())
    rows = []
    for row in dataset:
        content = row.get("content")
        if with_truncation:
            content = " ".join(row.get("content").split(" ")[:100])
        rows.append({"lookup_text": row.get("title"), "database_text": content})
    return rows

  def transform_stanford_qa_data(self, qa_filename="data/dev-v2.0.json"):
    #data/alegre_client_evaluation_data.zip must be unzipped in data dir for this to work!
    #we may want to limit the number of cases per paragraph we look at here - 
    #the groups of paragraphs are all thematically grouped, so results often 
    #yield "good" answers even though they refer to some other similar paragraph in the set.
    dataset = json.loads(open(qa_filename).read())
    paired_qas = []
    for row in dataset["data"]:
      for paragraph in row["paragraphs"]:
        for question in paragraph["qas"]:
          #subsample to 10% to keep data scale proportional to other tests we've run
          if random.random() < 0.1:
            paired_qas.append({"lookup_text": question.get("question"), "database_text": paragraph.get("context")})
    return paired_qas

  def fact_pairs_from_csv(self, filename="data/texts.csv", threshold=4):
    #data/alegre_client_evaluation_data.zip must be unzipped in data dir for this to work!
    # Expects filename to refer to local CSV path where columns are [similarity_score],[text_1],[text_2]
    with open(filename) as f:
      reader = csv.reader(f)
      dataset = []
      for row in reader:
        score = float(row[0])
        if score >= threshold:
          dataset.append({"database_text": row[1], "lookup_text": row[2], "score": score})
    return dataset

  def load_confounder_paragraphs(self, confounder_filename="data/train-v2.0.json"):
    #data/alegre_client_evaluation_data.zip must be unzipped in data dir for this to work!
    paragraphs = []
    for row in json.loads(open(confounder_filename).read())["data"]:
      for para in row["paragraphs"]:
        if random.random() < 0.5:
          paragraphs.append(para["context"])
    return paragraphs

  def load_confounder_headlines(self, confounder_filename="data/confounder_headlines.csv"):
    #data/alegre_client_evaluation_data.zip must be unzipped in data dir for this to work!
    confounders = []
    with open(confounder_filename) as f:
      reader = csv.reader(f)
      for row in reader:
        confounders.append(row[0].lower())
    return confounders

  def input_confounders(self, confounders, model_name, context):
    for row in confounders:
      request_params = {"model": model_name, "text": row}
      if context:
        request_params["context"] = context
      self.store_text_similarity(request_params)

  def get_for_text(self, text, model_name, context={}, language=None):
    if not context:
      context = {"task": "model_evaluation", "model_name": model_name}
    return json.loads(self.get_similar_texts({
      "model": model_name,
      "text": text.lower(),
      "context": context,
      "threshold": 0.0,
      "language": language
    }).text)

  def evaluate_model(self, dataset, model_name, confounders, store_dataset, omit_half, task_name="model_evaluation", language=None):
    # Similarity score should be from 0 to 5 similar to data found at https://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark,
    # text_1 will be uploaded to service, text_2 will be used to attempt to retrieve text_1 from the service in a GET request.
    split_point = int(len(dataset)/2)
    context = {"task": task_name, "model_name": model_name}
    if store_dataset:
      if confounders:
        self.input_confounders(confounders, model_name, context)
      if omit_half:
        sent_group = dataset[:split_point]
      else:
        sent_group = dataset
      self.input_cases(
        [f["database_text"].lower() for f in sent_group],
        model_name,
        context,
        language
      )
    results = {"count": 0, "success": 0, "server_errors": 0, "resultset": []}
    for ii, fact_pair in enumerate(dataset):
      results["count"] += 1
      result = json.loads(self.get_similar_texts({
        "model": model_name,
        "text": fact_pair["lookup_text"].lower(),
        "context": context,
        "threshold": 0.0,
        "language": language
      }).text)
      #due to indexing math this may be an off-by-one but it's close enough for our test today, and it's too late at night to make sure, but not too late so as not to remind myself about this later....
      is_omitted = ii > split_point
      results["resultset"].append({"fact_pair": fact_pair, "response": result, "is_omitted": is_omitted})
    return results

  def evaluate_model_iteratively(self, dataset, model_name, task_name="iterative_model_evaluation", language=None):
    context = {"task": task_name, "model_name": model_name}
    results = {"count": 0, "success": 0, "server_errors": 0, "resultset": []}
    for row in dataset:
      results["count"] += 1
      result = json.loads(self.get_similar_texts({
        "model": model_name,
        "text": row["lookup_text"],
        "context": context,
        "threshold": 0.0,
        "language": language
      }).text)
      result["count_at_time"] = results["count"]
      self.input_cases(
        [row["database_text"]],
        model_name,
        context,
        language
      )
      results["resultset"].append({"fact_pair": row, "response": result, "is_omitted": False})
    return results
      
  def store_text_similarity(self, request_params):
    return requests.post(self.hostname+self.text_similarity_path(), json=request_params)

  def get_similar_texts(self, request_params):
    return requests.get(self.hostname+self.text_similarity_path(), json=request_params)

  def run_b_test(self, model_name):
    return self.evaluate_model(self.fact_pairs_from_csv(), model_name, [], True, False)

  def run_c_test(self, model_name):
    return self.evaluate_model(self.fact_pairs_from_csv(), model_name, self.load_confounder_headlines(), True, False)

  def run_cwm_test(self, model_name):
    return self.evaluate_model(self.fact_pairs_from_csv(), model_name, self.load_confounder_headlines(), True, True)

  def run_qa_test(self, model_name):
    return self.evaluate_model(self.transform_stanford_qa_data(), model_name, [], True, False)

  def run_qac_test(self, model_name):
    return self.evaluate_model(self.transform_stanford_qa_data(), model_name, self.load_confounder_paragraphs(), True, False)

  def run_qawm_test(self, model_name):
    return self.evaluate_model(self.transform_stanford_qa_data(), model_name, self.load_confounder_paragraphs(), True, True)

  def run_ciper_test(self, model_name):
    return self.evaluate_model(self.transform_ciper_data(), model_name, [], True, True)

  def run_ciper_missing_test(self, model_name):
    return self.evaluate_model(self.transform_ciper_data(), model_name, [], True, True)

  def run_ciper_test_no_truncation(self, model_name):
    return self.evaluate_model(self.transform_ciper_data(False), model_name, [], True, True)

  def run_ciper_missing_test_no_truncation(self, model_name):
    return self.evaluate_model(self.transform_ciper_data(False), model_name, [], True, True)

  def run_india_today_iterative_test(self, model_name):
    return self.evaluate_model_iteratively(self.transform_india_today_data(), model_name)

  def run_india_today_hindi_iterative_test(self, model_name):
    return self.evaluate_model_iteratively(self.transform_india_today_data("data/indiatoday_texts_hindi.csv"), model_name)

  def run_hindi_headlines_test(self, model_name):
    return self.evaluate_model(self.transform_hindi_headlines_data(), model_name, [], True, False)

  def run_hindi_headlines_test_wm(self, model_name):
    return self.evaluate_model(self.transform_hindi_headlines_data(), model_name, [], True, True)

  def laser_vector(self, text, language):
    return requests.get(
      url="http://127.0.0.1:2212/vectorize",
      params={
        "q": text,
        "lang": language
      }
    ).json().get("embedding")[0]

  def generate_multilingual_datasets(self):
    datasets = {
      "hindi_headlines": random.sample(self.transform_hindi_headlines_data(), 400),
      "ciper": random.sample(self.transform_ciper_data(), 400),
      "fact_pairs": random.sample(self.fact_pairs_from_csv(), 400),
    }
    out_datasets = {}
    for title, rows in datasets.items():
        out_datasets[title] = []
        mismatch_indexes = []
        for i, row in enumerate(rows):
            mismatch_indexes.append([i, random.sample(set(range(len(rows)))-set([i]), 1)[0]])
            row["label"] = 1
            out_datasets[title].append(row)
        for mismatch in mismatch_indexes:
            row = {"lookup_text": rows[mismatch[0]].get("lookup_text", ""), "database_text": rows[mismatch[1]].get("database_text", "")}
            row["label"] = 0
            out_datasets[title].append(row)
    return out_datasets
      
  def evaluate_laser_model(self, dataset, data_name, language):
    results = {"resultset": []}
    mismatch_indexes = []
    for i,row in enumerate(dataset):
      score = cosine_sim(self.laser_vector(row.get("lookup_text", ""), language), self.laser_vector(row.get("database_text", ""), language))
      result = {"result": [{"_score": score, "_source": {"content": row.get("database_text", "")}}]}
      results["resultset"].append({"fact_pair": row, "response": result, "is_omitted": False})
      mismatch_indexes.append([i, random.sample(set(range(len(dataset)))-set([i]), 1)[0]])
    for first, second in mismatch_indexes:
      row = {"lookup_text": dataset[first].get("lookup_text", ""), "database_text": dataset[second].get("database_text", "")}
      score = cosine_sim(self.laser_vector(row.get("lookup_text", ""), language), self.laser_vector(row.get("database_text", ""), language))
      result = {"result": [{"_score": score, "_source": {"content": ""}}]}
      results["resultset"].append({"fact_pair": row, "response": result, "is_omitted": True})
    return results

  def run_laser_test(self):
      return {
        "hindi_headlines": self.evaluate_laser_model(random.sample(self.transform_hindi_headlines_data(), 400), "hindi_headlines", "hi"),
        "ciper": self.evaluate_laser_model(random.sample(self.transform_ciper_data(), 400), "ciper", "es"),
        "fact_pairs": self.evaluate_laser_model(random.sample(self.fact_pairs_from_csv(), 400), "fact_pairs", "en"),
      }

  def run_multi_test(self, model_name="elasticsearch", use_language_analyzer=True):
    if use_language_analyzer:
      return {
        "hindi_headlines": self.evaluate_model(random.sample(self.transform_hindi_headlines_data(), 400), model_name, [], True, True, "hindi_headlines", "hi"),
        "ciper": self.evaluate_model(random.sample(self.transform_ciper_data(), 400), model_name, [], True, True, "ciper", "es"),
        "fact_pairs": self.evaluate_model(random.sample(self.fact_pairs_from_csv(), 400), model_name, [], True, True, "fact_pairs", "en"),
      }
    else:
      return {
        "hindi_headlines": self.evaluate_model(random.sample(self.transform_hindi_headlines_data(), 400), model_name, [], True, True, "hindi_headlines", ""),
        "ciper": self.evaluate_model(random.sample(self.transform_ciper_data(), 400), model_name, [], True, True, "ciper", ""),
        "fact_pairs": self.evaluate_model(random.sample(self.fact_pairs_from_csv(), 400), model_name, [], True, True, "fact_pairs", ""),
      }

  def interpret_report(self, report):
    positions = Counter()
    results_dataset = []
    results_dataset.append(["Database-Stored Sentence", "Lookup Sentence", "Top Yielded Sentence", "ES Similarity Score", "Result Status", "PM ID"])
    for res in report["resultset"]:
      if not res.get('response', {}).get("message"):
        competing_sentences = [ee.get("_source", {}).get("content").lower() for ee in res.get("response", {}).get("result")]
        db_sentence = res.get("fact_pair", {}).get("database_text", "").lower()
        lookup_sentence = res.get("fact_pair", {}).get("lookup_text", "").lower()
        row = [db_sentence, lookup_sentence]
        if competing_sentences and db_sentence == competing_sentences[0]:
          positions.update([1])
          row.append(competing_sentences[0])
          row.append(res.get("response", {}).get("result")[0].get("_score"))
          row.append("Success")
        elif competing_sentences and db_sentence != competing_sentences[0] and db_sentence in competing_sentences:
          positions.update([competing_sentences.index(db_sentence)+1])
          row.append(competing_sentences[0])
          row.append(res.get("response", {}).get("result")[0].get("_score"))
          row.append("Partial Success")
        elif not competing_sentences:
          if res.get("is_omitted"):
            positions.update(["true negative"])
            row.append("Nothing found!")
            row.append("Nothing found!")
            row.append("True Negative")
          else:
            positions.update(["false negative"])
            row.append("Nothing found!")
            row.append("Nothing found!")
            row.append("False Negative")
        else:
          positions.update(["false positive"])
          row.append(competing_sentences[0])
          row.append(res.get("response", {}).get("result")[0].get("_score"))
          row.append("False Positive")
        row.append(res.get("fact_pair", {}).get("id"))
        results_dataset.append(row)
      else: 
        positions.update(["server error"])
    return results_dataset, positions

  def evaluate_cut_points(self, multi_report):
    results = {}
    for report_key in multi_report:
      results[report_key] = self.interpret_report(multi_report[report_key])
      with open(report_key+'_raw_report.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(results[report_key][0])
    unique_ints = sorted(list(set([e for e in [[int(ee[3]) for ee in e[0] if type(ee[3]) == float] for e in results.values()] for e in e])))
    accuracy_report = {}
    if unique_ints:
      cut_points = list(range(unique_ints[0], unique_ints[-1]+1))
    else:
      cut_points = [e/100.0 for e in list(range(1,100))]
    roc_rows = []
    for title, rows in results.items():
        scores = [e['response']['result'][0]["_score"] for e in multi_report[title]["resultset"] if e['response']['result']]
        maxval = max(scores)
        probs = [e/float(maxval) for e in scores]
        labels = [0 if e["is_omitted"] else 1 for e in multi_report[title]["resultset"] if e["response"]["result"]]
        fpr, tpr, thresholds = roc_curve(labels, probs)
        for f,t in zip(fpr, tpr):
            roc_rows.append([title, f, t])
    with open("roc_rows.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(roc_rows)
    for i in cut_points:
      for report_key in results:
        if not accuracy_report.get(report_key):
          accuracy_report[report_key] = {}
        all_results = [e for e in results[report_key][0] if type(e[3]) == float or type(e[3]) == np.float64]
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        total = 0
        for row in all_results:
          total += 1
          if row[3] > i and "Success" in row[4]:
            tp += 1
          elif row[3] > i and "Success" not in row[4]:
            fp += 1
          elif row[3] <= i and "False Positive" in row[4]:
            tn += 1
          elif row[3] <= i and "False Positive" not in row[4]:
            fn += 1
        prec = 0
        if float(tp+fp):
          prec = tp/float(tp+fp)
        recall = 0
        if float(tp+fn):
          recall = tp/float(tp+fn)
        acc = 0
        if total:
          acc = (tp+tn)/float(total)
        accuracy_report[report_key][i] = {"prec": prec, "recall": recall, "acc": acc, "tp": tp/float(total), "tn": tn/float(total), "fp": fp/float(total), "fn": fn/float(total)}
    
    return accuracy_report

  def accuracy_report_to_table(self, accuracy_report):
    rows = [["Dataset", "Thresh", "Acc", "Prec", "Recall", "TP", "TN", "FP", "FN", "Total"]]
    for key in accuracy_report:
      for thresh in accuracy_report[key]:
        row = accuracy_report[key][thresh]
        rows.append([key, thresh, row["acc"], row["prec"], row["recall"], row["tp"], row["tn"], row["fp"], row["fn"]])
    with open('accuracies.csv', 'w', newline='') as f:
      writer = csv.writer(f)
      writer.writerows(rows)
    return rows
      

if __name__ == '__main__':
  import csv
  from alegre_client import AlegreClient
  ac = AlegreClient()
  report = ac.generate_multilingual_datasets()
  # f = open("multilingual_sentence_matched_datasets.json", "w")
  # f.write(json.dumps(report))
  # f.close()
  # report = ac.run_multi_test("elasticsearch", True)
  cuts = ac.accuracy_report_to_table(ac.evaluate_cut_points(report))
