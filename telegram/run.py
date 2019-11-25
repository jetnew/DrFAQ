import time
from telegram.ext import Updater
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler, Filters
from handlers import *


# Initialise
token = "919334819:AAE0ZYaMbYThsD7Zlv1boNMIKv5BG7YC1WA"
updater = Updater(token=token, use_context=True)

# Set handlers
dispatcher = updater.dispatcher
dispatcher.add_handler(CommandHandler('start', start))
dispatcher.add_handler(MessageHandler(Filters.text, echo))
dispatcher.add_handler(CommandHandler('caps', caps))
dispatcher.add_handler(MessageHandler(Filters.command, unknown))

# Run
updater.start_polling()
time.sleep(30)
updater.stop()