import os
import json

from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain.prompts import (
    SystemMessagePromptTemplate,
)
from langchain.prompts import load_prompt
from langchain.schema import AIMessage, HumanMessage, SystemMessage, FunctionMessage
from langchain.agents import load_tools
from langchain.tools import format_tool_to_openai_function


SYSTEM_THOUGHT = load_prompt(os.path.join(os.path.dirname(__file__), 'data/prompts/system/thought.yaml'))
SYSTEM_RESPONSE = load_prompt(os.path.join(os.path.dirname(__file__), 'data/prompts/system/response.yaml'))
SYSTEM_TOOLS = load_prompt(os.path.join(os.path.dirname(__file__), 'data/prompts/system/tools.yaml'))


class BloomChain:
    def __init__(self, llm: ChatOpenAI, verbose: bool = False):
        self.llm = llm
        self.verbose = verbose

        # setup prompts
        self.system_thought = SystemMessagePromptTemplate(prompt=SYSTEM_THOUGHT).format()
        self.system_tools = SystemMessagePromptTemplate(prompt=SYSTEM_TOOLS).format()
        self.system_response = SystemMessagePromptTemplate(prompt=SYSTEM_RESPONSE)

        # setup tools
        # TODO: setup tool retreival
        self.tools = load_tools(["google-serper", "wolfram-alpha", "wikipedia", "arxiv"], llm=llm)
        self.tools_openai = [format_tool_to_openai_function(tool) for tool in self.tools]
        self.tool_names = [tool.name for tool in self.tools]
        

    def think(self, thought_memory: ChatMessageHistory, input: str, max_turns: int = 10):
        """Generate Bloom's thought on the user."""

        # load message history
        messages = [self.system_thought, *thought_memory.messages, HumanMessage(content=input), self.system_tools]

        # Builds the thought, makes sure it encompasses the student's needs as well as the necessary information to help the student
        # 
        # It works by letting the model first use tools to gain the necessary knowledge to answer the question, and then forming the thought
        # More technically, this loop is just checking for each of the 3 possible messages from the model: Reasoning for using a tool, a tool call, and the final thought
        turns = 0
        while True:
            # Check if we've reached the max number of turns
            turns += 1
            if turns > max_turns:
                # reset messages and come up with thought without tools
                messages = [self.system_thought, *thought_memory.messages, HumanMessage(content=input)]
                thought_message = self.llm.predict_messages(messages)

                # verbose logging
                if self.verbose:
                    # Seralize messages to strings
                    message_strings = [f"{message.type}: {message.content}" for message in messages]
                    print("Thought Conversation: ```\n", "\n\n".join(message_strings), "\n```\n")

                    print("New Thought (Tools Failed): ```\n", thought_message.content, "\n```\n")

                # update chat memory and return
                thought_memory.add_message(HumanMessage(content=input))
                thought_memory.add_message(thought_message)

                return thought_message.content

            thought_message = self.llm.predict_messages(messages, functions=self.tools_openai)

            # Check if the model is reasoning about a function call
            if thought_message.content.startswith("Reasoning:"):
                if self.verbose:
                    print("Reasoning: ```", thought_message.content, "```")

                if "function_call" in thought_message.additional_kwargs:
                    pass
                else:
                    messages.append(thought_message)
                    continue

            # Handle function calls
            if "function_call" in thought_message.additional_kwargs:
                # get the tool
                tool_name = thought_message.additional_kwargs["function_call"]["name"]
                tool = [tool for tool in self.tools if tool.name == tool_name][0]
                arguments = json.loads(thought_message.additional_kwargs["function_call"]["arguments"])
                
                # call the tool
                result = tool(arguments["__arg1"])
                result_message = FunctionMessage(name=tool_name, content=result)

                messages.append(thought_message)
                messages.append(result_message)

                # verbose logging
                if self.verbose:
                    print(f"Function call: ```{tool_name}({arguments['__arg1']})```")
                    print(f"Result: ```{result}```\n")

                continue

            # verbose logging
            if self.verbose:
                # Seralize messages to strings
                message_strings = [f"{message.type}: {message.content}" for message in messages]
                print("Thought Conversation: ```\n", "\n\n".join(message_strings), "\n```\n")

                print("New Thought: ```\n", thought_message.content, "\n```\n")

            break

        # update chat memory
        thought_memory.add_message(HumanMessage(content=input))
        thought_memory.add_message(thought_message)

        return thought_message.content
    

    def respond(self, response_memory: ChatMessageHistory, thought: str, input: str):
        """Generate Bloom's response to the user."""

        # load message history
        # TODO: Is the "thought: " prefix necessary?
        messages = [self.system_response.format(tools=self.tool_names), *response_memory.messages, HumanMessage(content=input), AIMessage(content=thought)]
        response_message = self.llm.predict_messages(messages)

        # verbose logging
        if self.verbose:
            # Seralize messages to strings
            message_strings = [f"{message.type}: {message.content}" for message in messages]
            print("Response Conversation: ```\n", "\n\n".join(message_strings), "\n```\n")

            print("New Response: ```\n", response_message.content, "\n```\n")

        # update chat memory
        response_memory.add_message(HumanMessage(content=input))
        response_memory.add_message(response_message)

        return response_message.content
    