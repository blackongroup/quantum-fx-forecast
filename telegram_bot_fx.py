# telegram_bot_fx.py

import os
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
from forecast_and_trade import forecast_and_trade

updater = Updater(token=os.getenv("TELEGRAM_TOKEN"))


def start(update: Update, context: CallbackContext):
    update.message.reply_text("FX Forecast Bot online!")


def run(update: Update, context: CallbackContext):
    try:
        forecast_and_trade()
        update.message.reply_text("Forecast executed and trades placed.")
    except Exception as e:
        update.message.reply_text(f"Error: {e}")

updater.dispatcher.add_handler(CommandHandler("start", start))
updater.dispatcher.add_handler(CommandHandler("run", run))

if __name__ == "__main__":
    updater.start_polling()
