from default_chat import DefaultChat
from nlp_chat import NLPChat


class Chat:
    def __init__(self):
        self.default_chat = DefaultChat()
        self.nlp_chat = NLPChat()

    def default(self, key):
        return self.default_chat.get_default_reply(key)

    def help(self, key):
        return self.default_chat.get_default_reply(key)

    def nlp(self, message):
        return self.nlp_chat.reply(message)
