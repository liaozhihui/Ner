from birnncrf.NERMODEL import train,predict
import json
import pandas as pd
import torch

# torch.backends.cudnn.enabled=False
corpus = "./data/example.dev"
word_dict = {}
sen_parsed = []
tags = []
tag=[]
parsed = []
with open(corpus, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            word,t = line.split(" ")
            # word_dict[word] = word_dict.get(word,0)+1
            # print(word,t)
            parsed.append(word.strip())
            tag.append(t.strip())

        else:
            sen_parsed.append(" ".join(parsed))
            tags.append(" ".join(tag))
            parsed = []
            tag = []
            # continue
df = pd.DataFrame(data={"content":sen_parsed,"tags":tags})
df.to_csv("datadev.csv")
# print(df)
# print(word_dict.keys())
# with open("./vocab_dict.json","w",encoding="utf-8") as f:
#     json.dump(list(word_dict.keys()),f,ensure_ascii=False)




deal_corpus = "./datadev.csv"

train(deal_corpus,max_sequence_length=5)
# predict()
