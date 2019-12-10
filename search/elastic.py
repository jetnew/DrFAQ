import os, base64, re, logging
from elasticsearch import Elasticsearch
import time

"""
Client: elasticsearch-7.5.0
Ref: https://www.elastic.co/downloads/elasticsearch
Python library: elasticsearch-7.1.0
Ref: https://elasticsearch-py.readthedocs.io/en/master/
"""


class Search:
    def __init__(self, text_file):
        # Local
        # self.elastic = Elasticsearch([{'host': 'localhost', 'port': 9200}])

        # Bonsai Deployment
        # Log transport details (optional):
        logging.basicConfig(level=logging.INFO)

        # Parse the auth and host from env:
        bonsai = os.environ['BONSAI_URL']
        auth = re.search('https\:\/\/(.*)\@', bonsai).group(1).split(':')
        host = bonsai.replace('https://%s:%s@' % (auth[0], auth[1]), '')

        # optional port
        match = re.search('(:\d+)', host)
        if match:
            p = match.group(0)
            host = host.replace(p, '')
            port = int(p.split(':')[1])
        else:
            port = 443

        # Connect to cluster over SSL using auth for best security:
        es_header = [{
            'host': host,
            'port': port,
            'use_ssl': True,
            'http_auth': (auth[0], auth[1])
        }]

        # Instantiate the new Elasticsearch connection:
        self.elastic = Elasticsearch(es_header)

        # Load search corpus into ElasticSearch
        self.elastic.indices.create(index='default', ignore=400)
        with open(text_file) as f:
            for i, line in enumerate(f):
                self.load(id=i, text=line)

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
