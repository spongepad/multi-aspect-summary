import pandas as pd
from konlpy.tag import Okt

okt = Okt()

def make_df():
  ae = pd.read_csv("data/earphone/aspect.txt")

  aspect = [ae.values[i][0] for i in range(1, len(ae), 2)]
  rel = [ae.values[i][0] for i in range(0, len(ae), 2)]

  aspect.insert(0, ae.columns.values[0])

  rel = pd.Series(rel, index=aspect)

  return rel