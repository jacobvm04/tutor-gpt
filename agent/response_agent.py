
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate, ChatPromptTemplate, BasePromptTemplate, load_prompt, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain import SerpAPIWrapper, LLMChain
from langchain.chat_models import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re
import os

OBJECTIVE_SYSTEM_RESPONSE = load_prompt(os.path.join(os.path.dirname(__file__), 'data/prompts/objective/system/response.yaml'))
OBJECTIVE_HUMAN_RESPONSE = load_prompt(os.path.join(os.path.dirname(__file__), 'data/prompts/objective/human/response.yaml'))

objective_system_response = SystemMessagePromptTemplate(prompt=OBJECTIVE_SYSTEM_RESPONSE)
objective_human_response = HumanMessagePromptTemplate(prompt=OBJECTIVE_HUMAN_RESPONSE)

objective_response_chat_prompt = ChatPromptTemplate.from_messages([objective_system_response, objective_human_response])

# Set up a prompt template
class ReponseAgentPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: ChatPromptTemplate
    # The list of tools available
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "

        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts

        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])

        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format_messages(**kwargs)

        return formatted
    

class ResponseAgentOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
    
def initialize_response_agent(llm: ChatOpenAI, tools: List[Tool]):
    prompt = ReponseAgentPromptTemplate(template=objective_response_chat_prompt, tools=[], input_variables=["intermediate_steps", "input", "thought", "history"])
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    tool_names = [tool.name for tool in tools]

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=ResponseAgentOutputParser(),
        stop=['\nObservation:'],
        allowed_tools=tool_names
    )
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

    return agent_executor
