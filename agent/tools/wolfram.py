from typing import Type

from langchain.tools import BaseTool
from langchain import WolframAlphaAPIWrapper
from pydantic import BaseModel, Field

from dotenv import load_dotenv

load_dotenv()

class WolframAlphaInputSchema(BaseModel):
    query: str = Field(..., title="query", description="The query to send to WolframAlpha")

class WolframAlphaTool(BaseTool):
    name = "wolframalpha"
    description = "WolframAlpha understands natural language queries about math, chemistry, physics, geography, history, art, astronomy, and more. Use to perform any work that requires calculations, do not attempt to calculate anything that's not trivial by hand."
    # args_schema: Type[BaseModel] = WolframAlphaInputSchema

    def _run(self, query: str):
        wolfram = WolframAlphaAPIWrapper()
        return wolfram.run(query)

    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
