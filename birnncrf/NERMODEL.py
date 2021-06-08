import os
import torch
import json
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader
from . import birnn_crf
import logging
from tqdm import tqdm
import pickle

logger = logging.getLogger(__name__)

PAD = "<PAD>"
OOV = "<OOV>"
PAD_VOCAB_IDX = 0
EMBEDDING_DIM = 128
HIDDEN_DIM = 32
ENTITY_TYPES = ['DIS', 'SYM', 'SGN', 'TES', 'DRU', 'SUR', 'PRE', 'PT', 'Dur', 'TP', 'REG', 'ORG', 'AT', 'PSB', 'DEG', 'FW',
                'CL']


def __eval_model(model,device,dataloader):
    model.eval()
    with torch.no_grad():
         losses, nums = zip(*[(model.loss(xb.to(device),yb.to(device)),len(xb)) for xb,yb in dataloader])

    return np.sum(np.multiply(losses,nums))/np.sum(nums)


def __save_loss(losses,file_path):
    pd.DataFrame(data=losses,columns=["epoch","batch","train_loss","val_loss"]).to_csv(file_path,index=False)


def __get_entities_from_tags(sentence, tags,ix_to_tag,word_to_ix, max_sequence_length):
    entities = []
    entity = None
    e_type = None
    e_start = None
    e_tags = []

    for i in range(len(tags)):
        t =ix_to_tag[tags[i]]
        e_tags.append(t)

        if e_type is not None:
            if e_type == t[2:]:
                entity = entity+sentence[i]

            if t == f"E-{e_type}":
                entities.append({"entity":entity,"e_type":e_type,"start":e_start,"end":i+1})
                entity = None
                e_type = None
                e_start = None

            if t.startswith("B-"):
                entity = sentence[i]
                e_type = t[2:]
                e_start = i

    log_output = [f"{w}({word_to_ix.get(w,'OOV')})={e_tags[idx]}({tags[idx]})" for idx,w in enumerate(list(sentence)[:max_sequence_length])]
    logger.debug(f"sentence vector representation is {log_output}")
    logger.info(f"{sentence} has entities:{entities}")
    return entities

def running_device(device=None):
    return device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # return "cpu"
def save_model(model,word_to_ix,tag_to_ix, save_path = "./model/ner_model.pt"):

    torch.save(model.state_dict(),save_path)
    embedding_file = save_path.replace(".pt","_embedding.pkl")
    with open(embedding_file,"wb") as f:
        pickle.dump(word_to_ix,f)

    tag_mapping_file = save_path.replace(".pt","_tag_mapping.pkl")

    with open(tag_mapping_file,"wb") as f:
        pickle.dump(tag_to_ix,f)

def load_model(model_file,embedding_dim,hidden_dim,device=None):
    embedding_file = model_file.replace(".pt","_embedding.pkl")
    with open(embedding_file,"rb") as f:
        word_to_ix = pickle.load(f)
        logger.debug(f"load ner model embedding from {embedding_file}")
    logger.info(f"OVV_IDX = {word_to_ix.get(OOV)}")

    tag_mapping_file = model_file.replace(".pt","_tag_mapping.pkl")

    with open(tag_mapping_file,"rb") as f:
        tag_to_ix = pickle.load(f)
        logger.debug(f"load ner model tag mapping from {tag_mapping_file}")

    model = birnn_crf.BiLSTMCRF(len(word_to_ix),len(tag_to_ix),embedding_dim,hidden_dim)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        state_dict = torch.load(model_file)
    else:
        device = torch.device("cpu")
        state_dict = torch.load(model_file,map_location=device)

    logger.debug(f"running on {device}")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.debug(f"load ner model from {model_file}")

    return model,word_to_ix,tag_to_ix


def preprocess_corpus(corpus,vocab_dict,max_length=0):
    df = pd.read_csv(corpus, index_col=0)
    df = df.sample(frac=1)
    logger.debug("shuffle training corpus")
    OOV_IDX = len(vocab_dict)-1
    logger.debug(f"load training model_data from {corpus}, size={df.shape[0]}")
    sentences_embeddings = []
    tags_vec = []
    tag_dict = {PAD:0}
    for idx, row in df.iterrows():
        sentence = row[0].split()
        max_seq_len = max_length if max_length >0 else len(sentence)
        vec = [vocab_dict.get(c,OOV_IDX) for c in sentence[:max_seq_len]]
        sentences_embeddings.append(vec+[PAD_VOCAB_IDX]*(max_seq_len-len(vec)))

        tag = row[1].split()
        vec = []
        for t in tag[:max_seq_len]:
            tag_dict[t] = tag_dict.get(t,len(tag_dict))
            vec.append(tag_dict[t])

        tags_vec.append(vec+[tag_dict[PAD]]*(max_seq_len-len(vec)))

    logger.debug(f"vocabulary size is {len(vocab_dict)}")
    logger.debug(f"target size is {len(tag_dict)}")
    xs,ys = np.asarray(sentences_embeddings),np.asarray(tags_vec)
    sentences_embeddings, tags_vec = map(torch.tensor,(xs,ys))

    return sentences_embeddings,tags_vec,tag_dict

def load_vocab_list(vocab_list):
    with open(vocab_list, encoding="utf8") as f:
        logger.debug(f"load vocab_dict form {vocab_list}")
        vocab_list = json.load(f)
        vocab_list.insert(PAD_VOCAB_IDX,PAD)
        vocab_list.append(OOV)
        vocab_list = {w: idx for idx,w in enumerate(vocab_list)}
    return vocab_list


def train(corpus, vocab_dict="./vocab_dict.json",val_split=0.05, test_split = 0.01, embedding_dim = EMBEDDING_DIM,
          hidden_dim = HIDDEN_DIM, max_sequence_length = 100, lr=1e-3, weight_decay=0,epochs=200,batch_size=2,device=None):

    logger.debug(f'start to train bilstm_crf')
    vocab_dict = load_vocab_list(vocab_dict)
    xs,ys,tag_to_ix = preprocess_corpus(corpus,vocab_dict,max_length=max_sequence_length)
    logger.info(f'tag count is {len(tag_to_ix)}')

    total_count = len(xs)
    assert total_count==len(ys)
    val_count = int(total_count*val_split)
    test_count = int(total_count*test_split)
    train_count = total_count-val_count-test_count
    assert train_count>0 and val_count>0

    indices = np.cumsum([0,train_count,val_count,test_count])

    (x_train,y_train),(x_val,y_val),(x_test,y_test) = [(xs[s:e],ys[s:e]) for s,e in zip(indices[:-1],indices[1:])]
    train_dl = DataLoader(TensorDataset(x_train,y_train),batch_size = batch_size,shuffle=True)
    valid_dl = DataLoader(TensorDataset(x_val,y_val),batch_size=batch_size,shuffle=True)
    test_dl = DataLoader(TensorDataset(x_test,y_test),batch_size=batch_size,shuffle=True)

    logger.debug(f"train_size = {train_count}, validation_size = {val_count}, test_size = {test_count}")

    model = birnn_crf.BiLSTMCRF(len(vocab_dict),len(tag_to_ix),embedding_dim=embedding_dim,hidden_dim=hidden_dim)
    losses = []
    optimizer = optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)
    device = running_device(device)
    model.to(device)
    logger.debug(f"running on {device}")
    print(f"running on {device}")

    best_val_loss = 1e4
    bar = tqdm(range(epochs))

    for epoch in bar:
        model.train()
        for bi,(xb,yb) in enumerate(train_dl):
            optimizer.zero_grad()
            loss = model.loss(xb.to(device),yb.to(device))
            loss.backward()
            optimizer.step()
            losses.append([epoch,bi,loss.item(),np.nan])


        val_loss = __eval_model(model,device,dataloader=valid_dl).item()

        losses[-1][-1] = val_loss

        bar.set_description_str("{:2d}/{},val_loss:{:5.2f}".format(epoch+1,epochs,val_loss))

        if val_loss<best_val_loss:
            best_val_loss = val_loss
            save_model(model,vocab_dict,tag_to_ix)


    test_loss = __eval_model(model,device,dataloader=test_dl).item()
    last_loss = losses[-1][:]
    last_loss[-1] = test_loss
    losses.append(last_loss)
    logger.info("training completed, test loss:{:.2f}".format(best_val_loss))
    return model

def predict(model,sentences, embedding_dim=EMBEDDING_DIM,hidden_dim=HIDDEN_DIM,max_sequence_length=100,device=None):
    model,word_to_ix,tag_to_ix = load_model(model,embedding_dim,hidden_dim,device)
    ix_to_tag = {}
    for k,v in tag_to_ix.items():
        ix_to_tag[v] = k
    OOV_IDX = word_to_ix[OOV]
    device = running_device(device)
    sent_tensor = []
    for sentence in sentences:
        vec = [word_to_ix.get(s,OOV_IDX) for s in sentence[:max_sequence_length]]
        vec = vec+[PAD_VOCAB_IDX]*(max_sequence_length-len(vec))
        sent_tensor.append(vec)

    sent_tensor = np.asarray(sent_tensor)
    sent_tensor = torch.from_numpy(sent_tensor).to(device)
    with torch.no_grad():
        scores, tags = model(sent_tensor)
        res = list(map(lambda sent,tag,score:{"ner":__get_entities_from_tags(sent,tag,ix_to_tag,word_to_ix,max_sequence_length),
                                              "score":score.item()},sentences,tags,scores))
    return res


def predict_on_excel(file,col,model="../model_data/ner_model.pt",format_entities = False):
    df = pd.read_excel(file,dtype={col:str})
    df = df.dropna(subset=[col]).drop_duplicates([col])
    sentences = df[col].to_list()
    res = predict(model,sentences)
    df["ner"] = res
    if format_entities:
        for e_type in ENTITY_TYPES:
            df[e_type] = df.ner.apply(lambda x:",".join(map(lambda ev:ev["entity"],filter(lambda e:e["e_type"]==e_type,x['ner']))))
            df[f"{e_type}_revised"] = df[e_type]
    df.to_excel(file,index=False)

def evaluate_metrics_on_ner(file,save_fp_json=True,col="question"):
    df = pd.read_excel(file,index=False,keep_default_na=False)
    eva_metrics = {}

    for t in ENTITY_TYPES:
        eva_metrics[f"{t}_COR"]=0
        eva_metrics[f"{t}_POS"]=0
        eva_metrics[f"{t}_ACT"]=0

    unmatched_sentences = set()
    for index,row in df.iterrows():
        for t in ENTITY_TYPES:
            act = str(row[f"{t}"]).split(",")
            pos = str(row[f"{t}_revised"]).split(",")
            cor = []

            for one in pos:
                if one in act:
                    cor.append(one)
                if len(pos) != len(cor) or len(act) != len(cor):
                    unmatched_sentences.add(index)
                eva_metrics[f'{t}_COR'] += len(cor)
                eva_metrics[f'{t}_POS'] += len(cor)
                eva_metrics[f'{t}_ACT'] += len(cor)

    if save_fp_json:
        json_list = []
        logger.debug(f"Unmatched entities total {len(unmatched_sentences)}")

        for s_idx in unmatched_sentences:

            s_entities = []
            s = df.loc[s_idx].at[col]
            for tt in ENTITY_TYPES:

                start = 0
                et_list = str(df.loc[s_idx].at[f"{tt}_revised"])

                if et_list != "":
                    for entity in et_list.split(","):
                        start = s.find(entity,start)
                        s_entities.append({"start":start,"end":start+len(entity),"value":entity,"entity":tt})
            json_list.append({"text":s,"intent":"na","entities":s_entities})

        with open("./ner_fp.json","w+", encoding="utf-8") as f:
            json.dump({"rasa_nlu_data":{"common_examples":json_list}},f,ensure_ascii=False)

    total_cor = 0
    total_pos = 0
    total_act = 0
    logger.info(f"Evaluation Metrics for {file}")

    for t in ENTITY_TYPES:
        cor = eva_metrics[f"{t}_COR"]
        pos = eva_metrics[f"{t}_POS"]
        act = eva_metrics[f"{t}_ACT"]

        total_cor += cor
        total_pos += pos
        total_act += act
        precision = cor/act
        recall = cor/pos
        logger.info(f"{t}:Precision={precision: .3f},Recall={recall: .3f},F1 score={2*precision*recall/(precision+recall):.3f}")

    total_precision = total_cor/total_act
    total_recall = total_cor/total_pos
    logger.info(f"Total:Precision={total_precision: .3f}, Recall={total_recall: .3f}, "
                f"F1 score = {2*total_precision*total_recall/(total_precision+total_recall): .3f}")











