import torch
from transformers import BertTokenizer, BertForQuestionAnswering

class QA:
    """
    HuggingFace BERT language model pre-trained on SQUAD.
    Ref: https://huggingface.co/transformers/index.html

    How does BERT answer questions?
    Ref: https://openreview.net/pdf?id=SygMXE2vAE
    """
    def __init__(self, text_file):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

        with open(text_file, 'r') as file:
            self.passage = file.read().replace('\n', ' ')

    def ask(self, question, threshold=1.0):
        """Ask question to QA."""
        score, answer = self.query(question)
        print("NLP score:", score)
        print("Answer:", answer)

        if score > threshold:
            return answer
        else:
            return None

    def query(self, question):
        """
        Query question with reference to the previously given passage.
        Returns (score, answer)
        Ref: https://huggingface.co/transformers/model_doc/bert.html#bertforquestionanswering
        """
        input_text = "[CLS] " + question + " [SEP] " + self.passage + " [SEP]"
        input_ids = self.tokenizer.encode(input_text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores = self.model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        score = self.compute_score(start_scores, end_scores)
        answer = ' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores) + 1])
        return score, answer

    def compute_score(self, start_scores, end_scores):
        start_scores = torch.nn.functional.softmax(start_scores, dim=1)
        end_scores = torch.nn.functional.softmax(end_scores, dim=1)
        score = torch.max(start_scores) + torch.max(end_scores)
        return round(score.item(), 3)


if __name__ == "__main__":
    """Example"""
    qa = QA()
    qa.load_passage("School fees for one student cost $300 a month.")
    score, answer = qa.query("How much do the school fees cost?")
    print("Answer:", answer)
    print("Score:", score)
    score, answer = qa.query("How much discount is given for school fees?")
    print("Answer:", answer)
    print("Score:", score)
