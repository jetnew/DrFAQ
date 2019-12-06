from elasticsearch import Elasticsearch
"""
This is a reference Python file that is not used in the system.
Ref: https://medium.com/naukri-engineering/elasticsearch-tutorial-for-beginners-using-python-b9cb48edcedc
"""

# Connect to the elastic cluster
es = Elasticsearch([{'host': 'localhost',
                     'port': 9200}])

# Load documents
e1 = {
    "text": "Love to play cricket",
}
e4 = {
    "text": "Love to play football",
}
res = es.index(index='megacorp',doc_type='employee',id=1,body=e1)
print("Indexing:", res['result'])
res = es.index(index='megacorp',doc_type='employee',id=4,body=e4)
print("Indexing:", res['result'])

# Get document
res=es.get(index='megacorp',doc_type='employee',id=1)
print("Getting:", res['_source']['text'])
res=es.get(index='megacorp',doc_type='employee',id=4)
print("Getting:", res['_source']['text'])

# Full text search
res = es.search(index='megacorp', body={
        'query':{
            'match':{
                "about": "play football"
            }
        }
    })
print("Searching:")
for hit in res['hits']['hits']:
    print(hit['_score'], hit['_source']['about'])

# Delete document
res = es.delete(index='megacorp',doc_type='employee',id=1)
res = es.delete(index='megacorp',doc_type='employee',id=4)
# print("Deleting:", res['result'])