# from generator import retriever
from passage_retrieval import Retriever

import torch
if __name__ == "__main__":
  # device = torch.device('cuda')  # 첫 번째 GPU 사용

  print(torch.cuda.is_available()) 
  from passage_retrieval import Retriever
  retriever = Retriever({})
  retriever.setup_retriever_demo("facebook/contriever-msmarco", "enwiki_2020_intro_only/enwiki_2020_dec_intro_only.jsonl", "enwiki_2020_intro_only/enwiki_dec_2020_contriever_intro/*",  n_docs=5, save_or_load_index=False)
  doc = retriever.search_document_demo("What is the relationship between the lifetime of a quantum state and its energy uncertainty?",1)
  print('어쨋든해결됨')
  print(doc["text"])


  