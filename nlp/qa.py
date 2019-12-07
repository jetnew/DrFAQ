import torch
from transformers import BertTokenizer, BertForQuestionAnswering


class QA:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    def load_passage(self, passage):
        """Load reference passage for querying later."""
        self.passage = passage

    def ask(self, question):
        """Ask question with reference to the previously given passage."""
        input_text = "[CLS] " + question + " [SEP] " + self.passage + " [SEP]"
        input_ids = self.tokenizer.encode(input_text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores = self.model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        return ' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores) + 1])

qa = QA()
qa.load_passage("The cost of one student's school fees is $90.")
answer = qa.ask("How much does school fees cost?")
print(answer)