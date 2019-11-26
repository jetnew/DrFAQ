import re
from manual_faq import ManualFAQ


manual_faq = ManualFAQ()


def start(update, context):
    """Default /start message."""
    context.bot.send_message(chat_id=update.effective_chat.id, text="Hi, I'm Dr FAQ. Ask me anything! :)\n" \
                             + "Usage:\n" \
                             + "/add_faq \"<question>\" \"<answer>\"")


def answer(update, context):
    """Replies the message."""
    question = update.message.text
    # TODO: NLP QUESTION ANSWERING
    if manual_faq.query(question):
        answer = manual_faq.query(question)
    else:
        answer = "<TODO: NLP-BASED ANSWER>"
    context.bot.send_message(chat_id=update.effective_chat.id, text=answer)


def configure(update, context):
    """/configure"""
    text = ' '.join(context.args)
    # TODO: CONFIGURATION SETTINGS
    answer = "<TODO: ENABLE CONFIGURATION SETTINGS>"
    context.bot.send_message(chat_id=update.effective_chat.id, text=answer)


def add_faq(update, context):
    """/add_faq '<question>' '<answer>'"""
    text = re.split("\"", update.message.text)

    if len(text) != 5:
        answer = "Invalid format. Usage: /add_faq \"<question>\" \"<answer>\""
    else:
        question, answer = text[1], text[3]

        # Save question-answer in global manual_faq object
        if not manual_faq.query(question):
            manual_faq.save(question, answer)

        answer = "FAQ saved. Try asking!\n" \
            + "Question: \"" + text[1] + "\"\n" \
            + "Answer: \"" + text[3] + "\"\n"
    context.bot.send_message(chat_id=update.effective_chat.id, text=answer)


def unknown(update, context):
    """Unknown command."""
    context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I didn't understand that command.")
