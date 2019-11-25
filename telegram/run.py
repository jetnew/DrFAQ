import time
from telegram.ext import Updater
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler, Filters
import handlers
import config


# Initialise
updater = Updater(token=config.api_token, use_context=True)

# Set handlers
dispatcher = updater.dispatcher
dispatcher.add_handler(CommandHandler('start', handlers.start))
dispatcher.add_handler(MessageHandler(Filters.text, handlers.answer))
dispatcher.add_handler(CommandHandler('ask', handlers.configure))
dispatcher.add_handler(MessageHandler(Filters.command, handlers.unknown))

# Run
updater.start_polling()
print("Running...")
time.sleep(30)
print("Stopped.")
updater.stop()