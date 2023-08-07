import os
import validators

from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory

from agent.bloom_chain import BloomChain

from dotenv import load_dotenv

load_dotenv()


def load_memories():
    """Load the memory objects"""
    thought_memory: ChatMessageHistory
    response_memory: ChatMessageHistory

    # memory definitions
    thought_memory = ChatMessageHistory()
    response_memory = ChatMessageHistory()

    return (thought_memory, response_memory)


def load_chains():
    """Logic for loading the chain you want to use should go here."""
    llm = ChatOpenAI(model_name="gpt-4-0613", temperature=1.2)

    # define chain
    bloom_chain = BloomChain(llm=llm, verbose=True)

    return bloom_chain


async def chat(**kwargs):
    # if we sent a thought across, generate a response
    if kwargs.get("thought"):
        assert kwargs.get("bloom_chain"), "Please pass the bloom chain."
        response_chain: BloomChain = kwargs.get("bloom_chain")
        response_memory: ChatMessageHistory = kwargs.get("response_memory")
        inp = kwargs.get("inp")
        thought = kwargs.get("thought")

        # get the history into a string
        # history = response_memory.load_memory_variables({})['history']

        # response = response_chain.apredict(
        #     input=inp,
        #     thought=thought,
        #     history=history
        # )

        response = response_chain.respond(response_memory, thought, inp)

        if "Student:" in response:
            response = response.split("Student:")[0].strip()
        if "Studen:" in response:
            response = response.split("Studen:")[0].strip()

        return response

    # otherwise, we're generating a thought
    else:
        assert kwargs.get("bloom_chain"), "Please pass the thought chain."
        inp = kwargs.get("inp")
        thought_chain: BloomChain = kwargs.get("bloom_chain")
        thought_memory: ChatMessageHistory = kwargs.get("thought_memory")

        # get the history into a string
        # history = thought_memory.load_memory_variables({})['history']

        # response = await thought_chain.apredict(
        #     input=inp,
        #     history=history
        # )

        response = await thought_chain.think(thought_memory, inp)

        if "Tutor:" in response:
            response = response.split("Tutor:")[0].strip()

        tool_response = await thought_chain.collect_data(response, inp)

        return response


class ConversationCache:
    "Wrapper Class for storing contexts between channels. Using an object to pass by reference avoid additional cache hits"

    def __init__(self):
        self.thought_memory, self.response_memory = load_memories()

    def restart(self):
        self.thought_memory.clear()
        self.response_memory.clear()
