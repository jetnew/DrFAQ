class ManualFAQ:
    def __init__(self):
        self.database = {}

    def save(self, question, answer):
        self.database[question] = answer

    def query(self, question):
        return self.database.get(question, 0)

