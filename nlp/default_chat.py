class DefaultChat:
    def __init__(self):
        self.default = {
            'start': "Hi, I'm Dr FAQ. Ask me anything! :)\nUsage:\n/add_faq \"question\" \"answer\"",
        }

    def get_default_reply(self, key):
        return self.default[key]
