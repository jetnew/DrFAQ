from elasticsearch import Elasticsearch
import time

"""
Client: elasticsearch-7.5.0
Ref: https://www.elastic.co/downloads/elasticsearch
Python library: elasticsearch-7.1.0
Ref: https://elasticsearch-py.readthedocs.io/en/master/
"""


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
            for hit in res['hits']['hits']:
                print(hit['_score'], hit['_source']['text'])

        top_hit = res['hits']['hits']
        if top_hit == []:
            print("No search results.")
            return None
        else:
            result = top_hit[0]['_source']['text']
            score = top_hit[0]['_score']

        print("Search score:", score)
        if score > 0.5:
            return result
        else:
            return None


if __name__ == "__main__":
    """Example"""
    search = Search()
    verbose = True

    text1 = "Love to play cricket"
    text2 = "Love to play football"

    search.load(id=0, text=text1, verbose=verbose)
    search.load(id=1, text=text2, verbose=verbose)

    search.search("play cricket", verbose=verbose)

    search.delete(id=0, verbose=verbose)
    search.delete(id=1, verbose=verbose)
