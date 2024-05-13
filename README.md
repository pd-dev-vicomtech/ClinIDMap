## ClinIDMap

**ClinIDMap**  is a tool for mapping identifiers (ID, codes) between clinical ontologies and lexical resources.

The tool interlinks identifiers from UMLS, SMOMED-CT, ICD-10, the corresponding Wikipedia articles and WordNet synsets. It's main goal is to provide semantic interoperability across the clinical concepts from various knowledge bases. 


### Requirements and Installation 

1. Before starting, the database should be indexed in Elasticsearch. The  database is under UMLS and SNOMED CT licences. 
https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/license_agreement_snomed.html

URL

2. The database will be uploaded and indexed in Elasticseach, which should be either installed on your device or used as a Docker image. Here, we install Elasticsearch as a Docker application with the following command.  


```shell script
docker-compose up
```

Stop and delete Docker application: 

```shell script
docker-compose down
```

```shell script
uvicorn application.web.main:create_app --host 0.0.0.0 --port 5858 --reload
```

### API use 

The API has three methods

1) application/{index_name} Post Index - method for database indexing 

When the Elasticsearch API is up, we should update databases in Elasticsearch index 

We pass the text file to the API. To process them correctly, the following arguments should be provided. 

```shell script
{
  "index_name": "string" # the name of the index where the database is indexed
  "path": "string",
  "headers": ["string"], # optional
  "separator": "string"
}
```


2) application/{index_name} Delete Index - method for index deleting 

To delete the index in Elasticseach istance: 

```shell script
{
  "index_name": "string"
}
```

3) application/map Get Item Mapping - the main method for code mapping


Input format contains the source ID we want to map, the type of the taxonomy and the flag if we need to get the infromation about this ID from Wikipedia and WordNet.

The source type must me UMLS, SNOMED_CT, ICD10CM or ICD10PCS. 

```shell script
index_name: clinires 
{
  "source_id": "1003369001",
  "source_type": "SNOMED_CT",
  "language": "SPA",
  "wiki": false
}
```

### CLI 

To update the Wikidata codes

```shell script
python -m application.wiki_wordnet.update_wiki
```


### Elastic commands 

curl 'localhost:9200/_cat/indices?v'

curl -XDELETE 'http://localhost:9500/clinires'


#### Examples 


% UMLS
C0011860 - diabetes tipo 2
C0025519 - metabolsm 

% SNOMED_CT
19997007 - hipnosis


% ICD10CM
H35.35 - Cystoid macular degeneration


### Cite

```
@inproceedings{zotova-etal-2022-clinidmap,
    title = "{Clin{IDM}ap: Towards a Clinical IDs Mapping for Data Interoperability}",
    author = "Zotova, Elena  and
      Cuadros, Montse  and
      Rigau, German",
    booktitle = "Proceedings of the Thirteenth Language Resources and Evaluation Conference",
    year = "2022",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    pages = "3661--3669",
}
```
