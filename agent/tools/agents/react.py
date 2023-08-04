from langchain.chains.openai_functions.base import convert_to_openai_function
from langchain.tools import format_tool_to_openai_function
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.prompts import AIMessagePromptTemplate
from langchain.schema import BaseMessage
from langchain.schema.messages import AIMessage, HumanMessage

import json
from typing import List


def process_openai_function_call(message: BaseMessage):
    if not "function_call" in message.additional_kwargs:
        # raise Exception("The LLM didn't return a function call in the action step.")
        return None, None
    
    function_call = message.additional_kwargs["function_call"]
    # print("functioN: ", function_call)
    tool_name = function_call["name"]

    try:
        # function_call["arguments"] = fix_code(function_call["arguments"])
        arguments = json.loads(function_call["arguments"])
    except json.decoder.JSONDecodeError:
        raise Exception(f"The LLM didn't return a valid json blob for the arguments: {function_call}")

    return tool_name, arguments


def think(thought: str):
    """Use to reason about what you need to do properly assist the student. Especially use these thoughts to reason about how you can use the provided tools to help the student."""
    return thought

def respond(message: str):
    """Use to finalize your response to the student"""
    return message

def extract_response(message: str):
    """Use to extract the response from the LLM's message"""
    
    return message.split("Response:")[1].strip() if "Response:" in message else None


class ReACTAgentOpenAI:
    def __init__(self, llm: ChatOpenAI, tools: List[Tool], max_steps: int = 10):
        self.llm = llm
        self.max_steps = max_steps

        # tools that be used to reason before taking an action
        self.reasoning_tools = [think, respond]
        self.reasoning_tools_openai = [convert_to_openai_function(tool) for tool in self.reasoning_tools]

        # tools that can be used as actions
        self.tools = tools
        self.tools_openai = [format_tool_to_openai_function(tool) for tool in self.tools]

    def load_prompts(self, prompts: List[BaseMessage]):
        """Load the prompt messages to instruct the agent"""
        self.conversation_history = prompts

    def plan(self):
        """Plan and take an action. Returns the response if one is generated this step."""
        
        # Step 1: Reason about what to do
        self.conversation_history.append(HumanMessage(content="Reason about what you are going to do and which tool you are going to use, or just return a response if you do are done."))

        message = self.llm.predict_messages(self.conversation_history)
        self.conversation_history.append(message)
        print(message)
        

        if response := extract_response(message.content):
            return response


        
        # Step 2: Take an action
        self.conversation_history.append(HumanMessage(content="Use a tool, or respond."))

        message = self.llm.predict_messages(self.conversation_history, functions=self.tools_openai)
        tool_name, arguments = process_openai_function_call(message)
        if tool_name is None:
            # Check if the message is prefixed with "Response:"
            if response := extract_response(message.content):
                return response
            
        # find the tool that matches the tool name
        for tool in self.tools:
            if tool.name == tool_name:
                action_message = AIMessage(content=f"Tool Usage: Used \"{tool_name}\" with arguments: {arguments}")
                self.conversation_history.append(action_message)

                observation = tool(arguments["__arg1"])
                observation_message = AIMessage(content=f"Tool Result: ```{observation}```")
                self.conversation_history.append(observation_message)

                print(f"Tool: {tool_name}({arguments})")
                print(f"Action: {observation}")

                break

        return None
    
    def run(self):
        if not self.conversation_history:
            raise Exception("You must load prompts before running the agent.")
        
        for _ in range(self.max_steps):
            response = self.plan()
            if response:
                return response
            
        return "Error: Agent did not finish in time."

        
        
        
