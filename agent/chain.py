import os
import validators
import faiss

from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import (
    VectorStoreRetrieverMemory,
    ConversationBufferMemory,
    CombinedMemory
) 
from langchain.prompts import load_prompt

from dotenv import load_dotenv

load_dotenv()

SYSTEM_THOUGHT = load_prompt(os.path.join(os.path.dirname(__file__), 'data/prompts/system/thought.yaml'))
SYSTEM_THOUGHT_REVISION = load_prompt(os.path.join(os.path.dirname(__file__), 'data/prompts/system/thought_revision.yaml'))
SYSTEM_RESPONSE = load_prompt(os.path.join(os.path.dirname(__file__), 'data/prompts/system/response.yaml'))
HUMAN_THOUGHT = load_prompt(os.path.join(os.path.dirname(__file__), 'data/prompts/human/thought.yaml'))
HUMAN_RESPONSE = load_prompt(os.path.join(os.path.dirname(__file__), 'data/prompts/human/response.yaml'))


def load_memories():
    """Load the memory objects"""
    thought_defaults = {
        "memory_key":"history",
        "input_key":"input",
        "ai_prefix":"Thought",
        "human_prefix":"User",
    }
    response_defaults = {
        "memory_key":"history",
        "input_key":"input",
        "ai_prefix":"Bloom",
        "human_prefix":"User",
    }
    thought_memory: ConversationBufferMemory
    response_memory: ConversationBufferMemory
    # memory definitions
    
    thought_memory = ConversationBufferMemory(
        **thought_defaults
    )

    response_memory = ConversationBufferMemory(
        **response_defaults
    )

    # add vector store-backed memory
    embedding_size = 1536 # Dimensions of the OpenAIEmbeddings
    index = faiss.IndexFlatL2(embedding_size)
    embedding_fn = OpenAIEmbeddings().embed_query
    vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})
    retriever = vectorstore.as_retriever(search_kwargs=dict(k=5))
    vector_memory = VectorStoreRetrieverMemory(retriever=retriever, memory_key="retrieved_vectors")

    # seed the vector store with some initial memories it should have (capabilities, etc)
    # When added to an agent, the memory object can save pertinent information from conversations or used tools
    vector_memory.save_context({"input": "Bloom is a subversive learning companion"}, {"output": "that's good to know"})
    vector_memory.save_context({"input": "Bloom can help with conceptual exploration, reading, writing, etc."}, {"output": "..."})

    thought_revision_memory = CombinedMemory(memories=[thought_memory, vector_memory])


    return (thought_revision_memory, thought_memory, response_memory)



def load_chains():
    """Logic for loading the chain you want to use should go here."""
    llm = ChatOpenAI(model_name = "gpt-4", temperature=1.2)

    # chatGPT prompt formatting
    system_thought = SystemMessagePromptTemplate(prompt=SYSTEM_THOUGHT)
    system_response = SystemMessagePromptTemplate(prompt=SYSTEM_RESPONSE)
    system_thought_revision = SystemMessagePromptTemplate(prompt=SYSTEM_THOUGHT_REVISION)


    human_thought = HumanMessagePromptTemplate(prompt=HUMAN_THOUGHT)
    human_response = HumanMessagePromptTemplate(prompt=HUMAN_RESPONSE)

    thought_chat_prompt = ChatPromptTemplate.from_messages([system_thought, human_thought])
    response_chat_prompt = ChatPromptTemplate.from_messages([system_response, human_response])
    thought_revision_prompt = ChatPromptTemplate.from_messages([system_thought_revision])


    # define chains
    thought_chain = LLMChain(
        llm=llm,
        prompt=thought_chat_prompt,
        verbose=True
    )

    response_chain = LLMChain(
        llm=llm,
        prompt=response_chat_prompt,
        verbose=True
    )

    thought_revision_chain = LLMChain(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        prompt=thought_revision_prompt,
        verbose=True
    )

    return ( 
        thought_chain, 
        response_chain,
        thought_revision_chain
    )


async def chat(**kwargs):
    # if we sent a thought across, generate a response
    if kwargs.get('thought'):
        assert kwargs.get('response_chain'), "Please pass the response chain."
        response_chain = kwargs.get('response_chain')
        response_memory = kwargs.get('response_memory')
        inp = kwargs.get('inp')
        thought = kwargs.get('thought')

        # get the history into a string
        history = response_memory.load_memory_variables({})['history']

        response = await response_chain.apredict(
            input=inp,
            thought=thought,
            history=history
        )
        if 'Student:' in response:
            response = response.split('Student:')[0].strip()
        if 'Studen:' in response:
            response = response.split('Studen:')[0].strip()

        return response

    # otherwise, we're generating a thought
    else:
        assert kwargs.get('thought_chain'), "Please pass the thought chain."
        inp = kwargs.get('inp')
        thought_chain = kwargs.get('thought_chain')
        thought_memory = kwargs.get('thought_memory')

        # get the history into a string
        history = thought_memory.load_memory_variables({})['history']

        
        response = await thought_chain.apredict(
            input=inp,
            history=history
        )

        if 'Tutor:' in response:
            response = response.split('Tutor:')[0].strip()

        print(f"OG THOUGHT: {response}")

        # revise the thought w/ personal context
        thought_revision_chain = kwargs.get('thought_revision_chain')
        thought_revision_memory = kwargs.get('thought_revision_memory')

        # not sure if the prompt is right but we can mess with that later
        retrieved_vectors = thought_revision_memory.load_memory_variables({"prompt": f"User Input: {inp} Generated Thought: {response}"})['retrieved_vectors']

        revision = await thought_revision_chain.apredict(
            thought=response,
            history=history, 
            retrieved_vectors=retrieved_vectors
        )

        print(f"REVISED THOUGHT: {revision}")

        return response, revision


class ConversationCache:
    "Wrapper Class for storing contexts between channels. Using an object to pass by reference avoid additional cache hits"
    def __init__(self):
        (
            self.thought_revision_memory,
            self.thought_memory, 
            self.response_memory, 
        )= load_memories()


    def restart(self):
       self.thought_memory.clear()
       self.response_memory.clear()
       self.thought_revision_memory.clear()
       self.convo_type = None
