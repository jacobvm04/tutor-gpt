# core functionality

import discord
from __main__ import (
    BLOOM_CHAIN,
    CACHE,
    THOUGHT_CHANNEL,
)
from discord.ext import commands
from typing import Optional
from agent.chain import ConversationCache


class Core(commands.Cog):
    def __init__(self, bot) -> None:
        self.bot = bot

    @commands.Cog.listener()
    async def on_member_join(self, member):
        welcome_message = """
Hello! Thanks for joining the Bloom server. 

I’m your Aristotelian learning companion — here to help you follow your curiosity in whatever direction you like. My engineering makes me extremely receptive to your needs and interests. You can reply normally, and I’ll always respond!

If I'm off track, just say so! If you'd like to reset our dialogue, use the /restart  command.

Need to leave or just done chatting? Let me know! I’m conversational by design so I’ll say goodbye 😊.

If you have any further questions, use the /help command or feel free to post them in https://discord.com/channels/1076192451997474938/1092832830159065128 and someone from the Plastic Labs team will get back to you ASAP!

Enjoy!
        """
        await member.send(welcome_message)

    @commands.Cog.listener()
    async def on_ready(self):
        await self.bot.sync_commands()
        print(f"We have logged in as {self.bot.user}: ID = {self.bot.user.id}")

    @commands.Cog.listener()
    async def on_message(self, message):
        # Don't let the bot reply too itself
        if message.author == self.bot.user:
            return

        # Get cache for conversation
        LOCAL_CHAIN = CACHE.get(message.channel.id)
        if LOCAL_CHAIN is None:
            LOCAL_CHAIN = ConversationCache()
            CACHE.put(message.channel.id, LOCAL_CHAIN)

        # Get the message content but remove any mentions
        inp = message.content.replace(str('<@' + str(self.bot.user.id) + '>'), '')
        n = 1800

        async def reply(forward_thought = True):
            "Generate response too user"
            async with message.channel.typing():
                thought, response = await BLOOM_CHAIN.chat(LOCAL_CHAIN, inp)

            thought_channel = self.bot.get_channel(int(THOUGHT_CHANNEL))

            # Thought Forwarding
            if (forward_thought):
                link = f"https://discord.com/channels/{message.guild.id}/{message.channel.id}/{message.id}"
                if len(thought) > n:
                    chunks = [thought[i:i+n] for i in range(0, len(thought), n)]
                    for i in range(len(chunks)):
                        await thought_channel.send(f"{link}\n```\nThought #{i}: {chunks[i]}\n```")
                else:
                    await thought_channel.send(f"{link}\n```\nThought: {thought}\n```")

            # Response Forwarding   
            if len(response) > n:
                chunks = [response[i:i+n] for i in range(0, len(response), n)]
                for chunk in chunks:
                    await message.channel.send(chunk)
            else:
                await message.channel.send(response)

        # if the message came from a DM channel...
        if isinstance(message.channel, discord.channel.DMChannel):
            await reply(forward_thought=False)

        if not isinstance(message.channel, discord.channel.DMChannel):
            if str(self.bot.user.id) in message.content:
                await reply(forward_thought=True)

        if not isinstance(message.channel, discord.channel.DMChannel):
            if message.reference is not None:
                reply_msg = await self.bot.get_channel(message.channel.id).fetch_message(message.reference.message_id)
                if reply_msg.author == self.bot.user:
                    if reply_msg.content.startswith("https://discord.com"):
                        return
                    if message.content.startswith("!no") or message.content.startswith("!No"):
                        return
                    await reply(forward_thought=True)
            

    @commands.slash_command(description="Help using the bot")
    async def help(self, ctx: discord.ApplicationContext):
        """
        Displays help message
        """
        help_message = """
Bloom is your digital learning companion. It can help you explore whatever you'd like to understand using Socratic dialogue 🏛️

Some possibilities:
🧠 Learn something new
🐇 Go down a rabbit hole
🚀 Have a stimulating conversation
⚔️ Challenge your beliefs & assumptions

You can also ask Bloom to help you with academic work, like:
✍️ Workshopping your writing
🔎 Doing research
📚 Reading comprehension
🗺️ Planning for assignments

**Instructions**
💬 You can chat with Bloom just like you'd chat with anyone else on Discord
🚧 If Bloom is going in a direction you don't like, just say so!
👋 When you're ready to end the conversation, say goodbye and Bloom will too
🔄 If you'd like to restart the conversation, use the /restart command.

**More Help**
If you're still having trouble, drop a message in https://discord.com/channels/1076192451997474938/1092832830159065128 and Bloom's builders will help you out!
        """
        await ctx.respond(help_message)

    @commands.slash_command(description="Restart the conversation with the tutor")
    async def restart(self, ctx: discord.ApplicationContext, respond: Optional[bool] = True):
        """
        Clears the conversation history and reloads the chains

        Args:
            ctx: context, necessary for bot commands
        """
        LOCAL_CHAIN = CACHE.get(ctx.channel_id)
        if LOCAL_CHAIN:
            LOCAL_CHAIN.restart()
        else:
            LOCAL_CHAIN = ConversationCache()
            CACHE.put(ctx.channel_id, LOCAL_CHAIN )

        if respond:
            msg = "Great! The conversation has been restarted. What would you like to talk about?"
            LOCAL_CHAIN.response_memory.add_ai_message(msg)
            await ctx.respond(msg)
        else:
            return




def setup(bot):
    bot.add_cog(Core(bot))
