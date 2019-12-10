import time
import os
from telegram.ext import Updater
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler, Filters
from bot import handlers
from bot import config
import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)

# Initialise
# updater = Updater(token=config.api_token, use_context=True)
updater = Updater(token=os.environ['TOKEN'],
                 use_context=True)

# Set handlers
dispatcher = updater.dispatcher
dispatcher.add_handler(CommandHandler('start', handlers.start))
dispatcher.add_handler(CommandHandler('configure', handlers.configure))
dispatcher.add_handler(CommandHandler('add_faq', handlers.add_faq))
dispatcher.add_handler(CommandHandler('help', handlers.help))
dispatcher.add_handler(MessageHandler(Filters.text, handlers.answer))
dispatcher.add_handler(MessageHandler(Filters.command, handlers.unknown))

# Run
updater.start_polling()
print("Running...")
updater.idle()
