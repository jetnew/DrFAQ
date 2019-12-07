from chat.default import Default
from nlp.nlp import NLP
from match.similarity import Match
from nlp.qa import QA
from search.elastic import Search


class Chat:
    def __init__(self):
        """Main Chat interface for Chatbot replies."""
        self.default = Default()
        self.nlp = NLP()
        self.match = Match()
        self.qa = QA()
        self.search = Search()

    def default(self, key):
        """Get default replies based on the key."""
        return self.default.get_default_reply(key)

    def nlp(self, message):
        """Returns a NLP reply."""
        return self.nlp.reply(message)

    def ask(self, question):
        """Ask a question to the QA system."""
        return self.qa.ask(question)

    def search(self, query):
        """Searches the database."""
        return self.search.search(query)
