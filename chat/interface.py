from chat.default import Default
from match.faq import FAQ
# from nlp.qa import QA
from search.elastic import Search


class ChatInterface:
    def __init__(self):
        """Main Chat interface for Chatbot replies."""
        self.default = Default()
        self.faq = FAQ("match/FAQ.xlsx")
        # self.qa = QA("nlp/QACorpus.txt")
        self.search = Search("search/SearchCorpus.txt")

    def default_reply(self, key):
        """Get default replies based on the key."""
        return self.default.get_default_reply(key)

    def reply(self, message):
        print("Message received:", message)
        # Phase 1: FAQ Matching
        print("Phase 1: FAQ Matching")
        answer = self.faq.ask_faq(message, threshold=0.95)  # change to 0.9 for large model
        if answer:
            print("Answer:", answer)
            return answer

        # # Phase 2: NLP Question Answering
        # print("Phase 2: NLP Question Answering")
        # answer = self.qa.ask(message, threshold=1.0)
        # if answer:
        #     print("Answer:", answer)
        #     return answer

        # Phase 3: Search
        print("Phase 3: Search")
        answer = self.search.search(message)
        if answer:
            print("Answer:", answer)
            return answer
        else:
            return "No content found."

    def ask(self, question):
        """Ask a question to the QA system."""
        return self.qa.query(question)

    def search(self, query):
        """Searches the database."""
        return self.search.search(query)
