import datasets
import json

quac = datasets.load_dataset('quac')

quac_dataset_converted = {}

print(quac['validation'][0].keys())

for key in quac.keys():
  data_to_be_converted = quac[key]
  quac_dataset_converted[key] = []

  for source in data_to_be_converted:
    answers = source['answers']
    context = source['context']
    questions = source['questions']
    title = source['wikipedia_page_title']
    ids = source['turn_ids']

    if len(answers['answer_starts']) != len(questions) or len(answers['answer_starts']) != len(answers['texts']):
      print('something wrong')

    for i in range(0, len(answers['answer_starts'])):
      new_question_answer = {}

      answer_start = answers['answer_starts'][i]
      text = answers['texts'][i]
      answer = {}
      answer['answer_start'] = answer_start
      answer['text'] = text

      question = questions[i]

      new_question_answer['answers'] = answer
      new_question_answer['context'] = context
      new_question_answer['question'] = questions[i]
      new_question_answer['title'] = title
      new_question_answer['id'] = ids[i]

      quac_dataset_converted[key].append(new_question_answer)
    
with open('train.json', 'w') as f:
    json.dump({'data': quac_dataset_converted['train']}, f)
    
with open('test.json', 'w') as f:
    json.dump({'data': quac_dataset_converted['validation']}, f)


