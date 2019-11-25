def start(update, context):
    """Default /start message."""
    context.bot.send_message(chat_id=update.effective_chat.id, text="Hi, I'm Dr FAQ! Ask me anything! :)")


def echo(update, context):
    """Echo user's message."""
    context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)


def caps(update, context):
    """/caps <message to return in caps>."""
    text_caps = ' '.join(context.args).upper()
    context.bot.send_message(chat_id=update.effective_chat.id, text=text_caps)


def unknown(update, context):
    """Unknown command."""
    context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I didn't understand that command.")
