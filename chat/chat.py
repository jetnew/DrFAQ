from default_chat import DefaultChat
from nlp_chat import NLPChat
from search.search import Search


class Chat:
    def __init__(self):
        """Main Chat interface for Chatbot replies."""
        self.default_chat = DefaultChat()
        self.nlp_chat = NLPChat()
        self.search = Search()

    def default(self, key):
        """Get default replies based on the key."""
        return self.default_chat.get_default_reply(key)

    def nlp(self, message):
        """Returns a NLP reply."""
        return self.nlp_chat.reply(message)

    def search(self, query):
        """Searches the database."""
        return self.search.search(query)
