from elasticsearch import Elasticsearch
import time


class Search:
    def __init__(self):
        self.elastic = Elasticsearch([{'host': 'localhost', 'port': 9200}])

    def load(self, id, text, verbose=False):
        """Loads documents in the format {'text': text}."""
        doc = {'text': text}
        res = self.elastic.index(index='default', doc_type='default', id=id, body=doc)
        if verbose:
            print("Indexing:", res['result'])

    def get(self, id, verbose=False):
        """Gets a document by id, for testing."""
        res = self.elastic.get(index='default', doc_type='default', id=id)
        if verbose:
            print("Getting:", res['_source']['text'])

    def delete(self, id, verbose=False):
        """Deletes a document by id, for testing."""
        res = self.elastic.delete(index='default', doc_type='default', id=id)
        if verbose:
            print("Deleting:", res['result'])

    def search(self, query, verbose=False):
        """Full text search."""
        time.sleep(1)
        res = self.elastic.search(index='default', body={
            'query': {
                'match': {
                    'text': query,
                }
            }
        })
        if verbose:
            print("Searching:")
            for hit in res['hits']['hits']:
                print(hit['_score'], hit['_source']['text'])
