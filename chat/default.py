class Default:
    def __init__(self):
        self.default = {
            'start': "Hi, I'm Dr FAQ. Ask me anything! :)",
            'help': "Ask any question! You can add custom FAQs by using the command:\n" +
                    "Usage:\n/add_faq \"question\" \"answer\"",
        }

    def get_default_reply(self, key):
        return self.default[key]
