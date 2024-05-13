import pandas as pd

from application.mapping.elastic_search import query_search, get_umls_query, get_sab_query, result2dicts, result2list_unique
from application import constants


def map_source_umls(source_id, language): 
    '''All codes related to UMLS CUI'''
    result = {} 
    result['status'] = 'OK'

    q_dic = get_umls_query(source_id, language) 
    
    umls_result = query_search(q_dic, constants.UMLS_EXT)

    result = result2dicts(umls_result)

    return result 

def map_source_sab(source_id, target_sab, language): 
    result = {}
    result['status'] = 'OK'
    result['source_id'] = source_id

    q_dic = get_sab_query(source_id=source_id, target_sab=target_sab, lang=language) #SCTSPA
    print('QUERY', q_dic)
    umls_result = query_search(q_dic, constants.UMLS_EXT)

    result = result2dicts(umls_result)
    print(result)

    return result


