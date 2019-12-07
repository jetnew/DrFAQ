import spacy


class Match:
    """Ref: https://spacy.io/"""
    def __init__(self):
        """Load language model."""
        spacy.prefer_gpu()
        self.nlp = spacy.load("en_core_web_sm")
        # self.nlp = spacy.load("en_core_web_lg") # Switch to large version for performance

    def compare(self, s1, s2):
        """
        Measure similarity score between two strings.
        Ref: https://spacy.io/api/doc#similarity
        """
        p1 = self.nlp(s1)
        p2 = self.nlp(s2)
        return p1.similarity(p2)


if __name__ == "__main__":
    """Example"""
    match = Match()
    s1 = "How much is the cost of one student's school fees?"
    s2 = "How much do I need to pay for my child's school fees?"
    score = match.compare(s1, s2)
    print("Similarity score:", score)
