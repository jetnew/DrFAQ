# DrFAQ
DrFAQ is a Question Answering NLP Chatbot for Text Document Corpora

# Objective
* Given an organisation's corpus of documents, generate a chatbot to enable natural question-answering capabilities.

# Demo
* Try it out - [t.me/DrFAQ_Bot](t.me/DrFAQ_Bot)
* Due to Heroku's free tier limits, only FAQ Question Matching using spaCy's Similarity and Answer Search using ElasticSearch functions are enabled.

# Methodology
When a question is asked, the following processes are performed:
1. FAQ Question Matching using spaCy's Similarity
    * From a given list of Frequently Asked Questions (FAQs), the chatbot detects similarity to the specified question and selects the best answer from the existing list.
2. NLP Question Answering using huggingface's BERT
    * If the question asked is dissimilar to any existing FAQs, perform question answering on the knowledge base and return a sufficiently confident answer.
3. Answer Search using ElasticSearch
    * If the answer is not sufficiently confident, perform a search on the document corpus and return the search results.
4. Human Intervention
    * If the search results are still not relevant, prompt a human to add the question-answer pair to the existing list of specified FAQs, or speak to a human.

# Tasks
* Telegram Chatbot Hosting
    * The Telegram Chatbot is currently deployed at [t.me/DrFAQ_Bot](t.me/DrFAQ_Bot) for demo purposes.
    * Hosting is done by Heroku, and due to free tier limits, only FAQ question matching and ElasticSearch is enabled. Unfortunately, NLP question answering would exceed Heroku's free tier memory limit.


# References
* [explosion/spaCy](https://github.com/explosion/spaCy) - Industrial-strength Natural Language Processing (NLP) with Python and Cython
* [huggingface/transformers](https://github.com/huggingface/transformers) - Transformers: State-of-the-art Natural Language Processing for TensorFlow 2.0 and Pytorch
* [elastic/elasticsearch-py](https://github.com/elastic/elasticsearch-py) - Official Python low-level client for Elasticsearch
