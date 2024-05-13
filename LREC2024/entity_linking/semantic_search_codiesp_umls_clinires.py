
import numpy as np
import pandas as pd
import torch
import pickle5 as pickle 
import os
import tqdm
import sys
from sentence_transformers import util
from faiss_utils import faiss_search
from transformer_utils import texts2vectors, cls_pooling, get_chunks, flatten, mean_pooling
from sklearn.metrics import f1_score


### DATASET
train_dir = 'corpus/codiesp'
train_file = 'test_cui_mapped_grouped.tsv'
# train_dir = '/DATA/ezotova_data/ICD-10_CodiEsp/search_rerank'
df_train = pd.read_csv(os.path.join(train_dir, train_file), sep='\t').fillna('')
df_empty = df_train[df_train['CUI'] == '']
print(len(df_empty))
print(df_empty.head())
# df_train = df_train.drop_duplicates(subset=['text'], keep='first')
print('Corpus after dropping duplicates', len(df_train))

df_train_d = df_train.loc[df_train['label'] == 'DIAGNOSTICO' ]
df_train_p = df_train.loc[df_train['label'] == 'PROCEDIMIENTO']
df_train_list = [df_train_d, df_train_p, df_train] #  

# CLINIRES
df_database = pd.read_csv('database_/UMLS_WIKI_DESCRS_SEMGROUP_20240105.txt', sep='|', dtype='str').fillna('')
df_database = df_database[(df_database['LAT'] == 'SPA') 
                          & ((df_database['SEMGROUP'] == 'DISO') 
                          | (df_database['SEMGROUP'] == 'PROC') 
                          | (df_database['SEMGROUP'] == 'CONC'))]

print('CLINIRES SPA',  len(df_database))
print(df_database.head())

sab_values = ['ICD10_WIKI', 'ICD10CM_WIKI', 'ICD10', 'ICD10PCS_WIKI', 'ICD10PCS', 'ICD10CM', 'ICD10CM_SPA', 'ICD10PCS_SPA', 'SNOMEDCT2ICD10_SPA']

# df_icd10_select = df_database[df_database['SAB'].isin(sab_values)]
# icd10_codes = df_icd10_select['CODE'].to_list()
# print('ICD-10 unique', len(list(set(icd10_codes))))
# print('ICD-10', len(df_icd10_select))
# sys.exit()

descriptions0 = df_database['STR'].str.lower().to_list() # descriptions used for semantic search 
codes = df_database['CUI'].to_list() # codes used for predictions

# df_train = df_train[['id', 'label', 'code', 'text', 'offset', 'CUI']]
print(df_train.head())
labels = df_train.CUI.str.split().to_list()
print(len(labels))
codes_true =df_train.code.to_list()

## VECTORS
output_folder = 'SapBERT_codiesp_clinires'
if not os.path.exists(output_folder):
	os.mkdir(output_folder)

output_file_corpus = os.path.join(output_folder, 'corpus_vectors_test.pkl')

if os.path.exists(output_file_corpus): 
    print('Corpus embeddings exist, loading')
    query_embeddings = pickle.load( open(output_file_corpus, "rb"))
else: 
    print('Calculating corpus embeddings')
    queries0 = df_train.text.str.lower().to_list()
    queries_chunks = get_chunks(queries0, 1200)
    print('Corpus queries chunks', len(queries_chunks))

    query_embeddings_chunks = []
    for i, queries in enumerate(queries_chunks): 
        print('Corpus chunk', i)
        model_output_queries, attention_mask = texts2vectors(queries)
        query_embeddings0 = cls_pooling(model_output_queries)
        query_embeddings0 = query_embeddings0.cpu().detach().numpy()
        query_embeddings_chunks.append(query_embeddings0)

    query_embeddings = flatten(query_embeddings_chunks)
    print(query_embeddings[0])
    with open(output_file_corpus, 'wb') as handle:
        pickle.dump(query_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

output_file_icd10 = os.path.join(output_folder, 'clinires_vectors.pkl')

if os.path.exists(output_file_icd10): 
    print('Database emdeddings exist, loading')
    descriptions_embeddings = pickle.load( open(output_file_icd10, "rb"))
else: 
    print('Calcualting database embeddings')
    descriptions_chunks = get_chunks(descriptions0, 1200)
    
    print('Database chunks', len(descriptions_chunks))

    descriptions_embeddings_chunks = []
    for descriptions in tqdm.tqdm(descriptions_chunks):  
        model_output_descr, attention_mask = texts2vectors(descriptions)
        
        descriptions_embeddings0  = cls_pooling(model_output_descr)
        descriptions_embeddings0 = descriptions_embeddings0.cpu().detach().numpy()
        descriptions_embeddings_chunks.append(descriptions_embeddings0)

    descriptions_embeddings = flatten(descriptions_embeddings_chunks)
    with open(output_file_icd10, 'wb') as handle:
        pickle.dump(descriptions_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

d = 1024 # model vector demension
k = 64   # number of nearest neighbors

D, I = faiss_search(descriptions_embeddings, query_embeddings, k=k, d=d)

result = []
for inds, ds, vecs, label, code in tqdm.tqdm(zip(I, D, query_embeddings, labels, codes_true)): #inds in ICD-10 corpus
    top_descriptions = [descriptions0[i] for i in inds]
    top_predictions = [codes[i] for i in inds]

    result.append({'descriptions': top_descriptions, 'distancies': ds, 'predicted_labels': top_predictions, 'true': label, 'code_true': code})


ns = [1, 10, 20, 30, 64]
for n in ns: 
    print('Top N', n)
    predictions = []
    predictions_icd = []
    dictances = []
    top_n = []
    top_n_icd = []
    top_n_icd_any = []
    pred_descriptions = []
    for item in tqdm.tqdm(result):
        predictions.append(str(item['predicted_labels'][0]))
        dictances.append(item['distancies'][0])
        pred_descriptions.append(item['descriptions'][0])
        top = item['predicted_labels']
        # print(item['true'])
        # if item['true'] in top[:n]:
        if any(i in top[:n] for i in item['true']):
            top_n.append(1)
        else:
            top_n.append(0)

        df_cui_to_icd10 = df_database[(df_database['CUI'] == str(item['predicted_labels'][0])) & (df_database['SAB'].isin(sab_values))]
        codes = df_cui_to_icd10['CODE'].to_list()
        if len(codes) == 0:
            predictions_icd.append('NOCODE')
        else:
            predictions_icd.append(codes[0])

        if len(codes) != 0 and codes[0] == item['code_true'].upper():
            top_n_icd.append(1)
        else:
            top_n_icd.append(0)

        if len(codes) != 0 and item['code_true'].upper() in codes:
            top_n_icd_any.append(1)
        else:
            top_n_icd_any.append(0)
        
    accuracy = np.array(top_n).sum()/len(top_n)
    accuracy_icd = np.array(top_n_icd).sum()/len(top_n_icd)

    print('P@{} Bi-Encoder : {}'.format(n, accuracy))
    print('Accuracy ICD-10 : {}'.format(accuracy_icd))
    print('Accuracy ICD-10 any: {}'.format(np.array(top_n_icd_any).sum()/len(top_n_icd_any)))

    # f1 = f1_score(labels, predictions, average='macro')
    # print('F1-score Bi-Encoder: {}'.format(f1)) 
    # print()

    df_train['prediction'] = predictions
    df_train['pred_cd10'] = predictions_icd
    df_train['pred_descr'] = pred_descriptions
    df_train.to_csv(os.path.join(output_folder, 'predictions.tsv'), sep='\t')

    # accuracy = np.array(top_n).sum()/len(top_n)
    # print('P@{} Bi-Encoder : {}'.format(n, accuracy))

    # # f1 = f1_score(labels, predictions, average='macro')
    # # print('F1-score Bi-Encoder: {}'.format(f1)) 
    # # print()

    # df_train['prediction'] = predictions
    # df_train['pred_descr'] = pred_descriptions
    # df_train.to_csv(os.path.join(output_folder, 'predictions.tsv'), sep='\t')

cuis_select = df_train.CUI.str.split().to_list() 
predictions_select = df_train.prediction.to_list()

to_select = flatten(cuis_select) + predictions_select

df_select = df_database[df_database['CUI'].isin(to_select)]
print('SELECT')
print(df_select.head(10))
df_select.to_csv(os.path.join(output_folder, 'cuis_select.tsv'), sep='\t', index=False)