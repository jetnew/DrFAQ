import spacy

spacy.prefer_gpu()

# nlp = spacy.load("en_core_web_sm") # For dev
nlp = spacy.load("en_core_web_lg") # For deploy

apples = nlp("How much is the cost of one student's school fees?")
oranges = nlp("How much do I need to pay for my child's school fees?")

apples_oranges = apples.similarity(oranges)
oranges_apples = oranges.similarity(apples)

print(apples_oranges)
print(oranges_apples)
