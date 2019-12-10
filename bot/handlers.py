import re
from bot.manual_faq import ManualFAQ
from chat.interface import ChatInterface
from log.logger import Logger

manual_faq = ManualFAQ()
chat_interface = ChatInterface()
logger = Logger("log/logbook.xlsx")


def start(update, context):
    """Default /start message."""
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text=chat_interface.default_reply('start'))


def answer(update, context):
    """Replies the message."""
    message = update.message
    if manual_faq.query(message):
        answer = manual_faq.query(message.text)
    else:
        answer = chat_interface.reply(message.text)

    logger.log([message.date,
                message.from_user.username,
                message.text,
                answer])
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
    print(len(text))
    if len(text) != 5:
        reply = "Invalid format. Usage: /add_faq \"question\" \"answer\""
    else:
        question, answer = text[1], text[3]

        # Save question-answer in global manual_faq object
        if not manual_faq.query(question):
            reply = manual_faq.save(question, answer)
        else:
            reply = "Question exists. Override: /override_faq \"question\" \"answer\""

    context.bot.send_message(chat_id=update.effective_chat.id, text=reply)


def help(update, context):
    """/help"""
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text=chat_interface.default('help'))


def unknown(update, context):
    """Unknown command."""
    context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I didn't understand that command.")
