import os
# import globals
import discord
from dotenv import load_dotenv
from agent.chain import load_chains
from agent.cache import LRUCache

def init():
    global THOUGHT_CHAIN, \
    RESPONSE_CHAIN, \
    THOUGHT_REVISION_CHAIN, \
    CACHE, \
    THOUGHT_CHANNEL
    
    CACHE = LRUCache(50)
    THOUGHT_CHANNEL = os.environ["THOUGHT_CHANNEL_ID"]
    ( 
        THOUGHT_CHAIN, 
        RESPONSE_CHAIN, 
        THOUGHT_REVISION_CHAIN,
    ) = load_chains()
    
load_dotenv()
token = os.environ['BOT_TOKEN']

# globals.init()
init()

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.members = True

bot = discord.Bot(intents=intents)


bot.load_extension("bot.core")


bot.run(token)
