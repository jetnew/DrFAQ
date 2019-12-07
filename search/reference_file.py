from search.elastic import Search
"""
IMPORTANT:
For some reason this file keeps being run.
Ref: run.py > handlers.py > chat.py
"""


search = Search()
verbose = True

text1 = "Love to play cricket"
text2 = "Love to play football"

search.load(id=0, text=text1, verbose=verbose)
search.load(id=1, text=text2, verbose=verbose)

search.search("play cricket", verbose=verbose)

search.delete(id=0, verbose=verbose)
search.delete(id=1, verbose=verbose)
