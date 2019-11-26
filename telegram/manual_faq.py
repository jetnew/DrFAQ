class ManualFAQ:
    def __init__(self):
        self.database = {}

    def save(self, question, answer):
        self.database[question] = answer
        return f"FAQ saved. Try asking!\nQuestion: {question}\nAnswer: {answer}"

    def query(self, question):
        return self.database.get(question, 0)

