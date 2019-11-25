def start(update, context):
    """Default /start message."""
    context.bot.send_message(chat_id=update.effective_chat.id, text="Hi, I'm Dr FAQ. Ask me anything! :)")


def answer(update, context):
    """Replies the message."""
    question = update.message.text
    # TODO: NLP QUESTION ANSWERING
    answer = "<TODO: ANSWER THE QUESTION>"
    context.bot.send_message(chat_id=update.effective_chat.id, text=answer)


def configure(update, context):
    """/configure"""
    text = ' '.join(context.args)
    # TODO: CONFIGURATION SETTINGS
    answer = "<TODO: ENABLE CONFIGURATION SETTINGS>"
    context.bot.send_message(chat_id=update.effective_chat.id, text=answer)


def unknown(update, context):
    """Unknown command."""
    context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I didn't understand that command.")
