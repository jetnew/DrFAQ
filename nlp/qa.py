import torch
from transformers import BertTokenizer, BertForQuestionAnswering


class QA:
    """
    HuggingFace BERT language model pre-trained on SQUAD.
    Ref: https://huggingface.co/transformers/index.html
    """
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    def load_passage(self, passage):
        """Load reference passage for querying later."""
        self.passage = passage

    def ask(self, question):
        """
        Ask question with reference to the previously given passage.
        Ref: https://huggingface.co/transformers/model_doc/bert.html#bertforquestionanswering
        """
        input_text = "[CLS] " + question + " [SEP] " + self.passage + " [SEP]"
        input_ids = self.tokenizer.encode(input_text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))]
        start_scores, end_scores = self.model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        return ' '.join(all_tokens[torch.argmax(start_scores): torch.argmax(end_scores) + 1])


if __name__ == "__main__":
    """Example"""
    qa = QA()
    qa.load_passage("The quick brown fox jumps over the lazy dog.")
    answer = qa.ask("How much does school fees cost?")
    print(answer)
