from chat.default import Default
from nlp.nlp import NLP
from search.elastic import Search


class Chat:
    def __init__(self):
        """Main Chat interface for Chatbot replies."""
        self.default = Default()
        self.nlp_chat = NLP()
        self.search = Search()

    def default(self, key):
        """Get default replies based on the key."""
        return self.default.get_default_reply(key)

    def nlp(self, message):
        """Returns a NLP reply."""
        return self.nlp_chat.reply(message)

    def search(self, query):
        """Searches the database."""
        return self.search.search(query)
