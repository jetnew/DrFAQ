import pandas as pd
from match.similarity import Match


class FAQ:
    def __init__(self, excel_file):
        self.df = pd.read_excel(excel_file)
        self.match = Match()

    def ask_faq(self, text, threshold=0.7):
        """Returns FAQ answer if similarity score exceeds threshold"""
        max_score = 0
        question_idx = 0
        for i, q in enumerate(self.df['Question']):
            s = self.match.compare(text, q)
            if s > max_score:
                max_score = s
                question_idx = i
        print("FAQ score:", max_score)
        if max_score > threshold:
            return self.df['Answer'][question_idx]
        else:
            return None


if __name__ == "__main__":
    faq = FAQ("FAQ.xlsx")
    answer = faq.ask_faq("What are the semester school fees?")
    print(answer)
