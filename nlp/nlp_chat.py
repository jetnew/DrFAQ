from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer


class NLPChat:
    def __init__(self):
        self.chatbot = ChatBot('Ron Obvious')
        trainer = ChatterBotCorpusTrainer(self.chatbot)
        trainer.train("chatterbot.corpus.english")

    def reply(self, message):
        return str(self.chatbot.get_response(message))
