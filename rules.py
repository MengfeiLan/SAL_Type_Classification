import nltk
from nltk.stem import PorterStemmer

nltk.download("punkt")
ps = PorterStemmer()


def check_funding(l1):
  for i in l1[0]:
    if ps.stem(i) in ["financi", "financ", "fund"]:
      return True
  return False