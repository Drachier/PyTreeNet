
import configparser

import requests

from ...consts import CONFIG_FILE as CONFIG_FILE_NAME

def send_telegram_message(bot_token: str,
                          chat_id: str,
                          message: str):
    """
    Send a message to a Telegram chat using the Telegram Bot API.

    Args:
        bot_token (str): The token of the Telegram bot.
        chat_id (str): The ID of the Telegram chat to send the message to.
        message (str): The message to send.
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    response = requests.post(url, data=payload)
    if response.status_code != 200:
        print(f"Failed to send message: {response.text}")

def load_telegram_config() -> tuple[str, str]:
    """
    Load Telegram bot token and chat ID from the config.ini file.
    """
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE_NAME)
    change = False
    try:
        bot_token = config['telegram']['bot_token']
    except KeyError:
        bot_token = input("Enter your Telegram bot token: ")
        change = True
    try:
        chat_id = config['telegram']['chat_id']
    except KeyError:
        chat_id = input("Enter your Telegram chat ID: ")
        change = True
    if change:
        configure_telegram(bot_token, chat_id)
        bot_token, chat_id = load_telegram_config()  # Reload to ensure it's correct
    return bot_token, chat_id

def configure_telegram(bot_token: str | None = None,
                       chat_id: str | None = None):
    """
    Configure Telegram bot token and chat ID by prompting the user for input.
    The values are saved to the config.ini file for future use.

    Args:
        bot_token (str | None): The Telegram bot token. If None, the user will
            be prompted to enter it. Defaults to None.
        chat_id (str | None): The Telegram chat ID. If None, the user will
            be prompted to enter it. Defaults to None.
    """
    if bot_token is None:
        bot_token = input("Enter your Telegram bot token: ")
    if chat_id is None:
        chat_id = input("Enter your Telegram chat ID: ")
    with open(CONFIG_FILE_NAME, 'a') as configfile:
        string = "[telegram]\n"
        string += f"bot_token = {bot_token}\n"
        string += f"chat_id = {chat_id}\n"
        configfile.write(string)
        print(f"Saved Telegram configuration to {CONFIG_FILE_NAME}")

def send_sim_finished_message():
    bot_token, chat_id = load_telegram_config()
    send_telegram_message(bot_token, chat_id, "The simulation has finished running!")
