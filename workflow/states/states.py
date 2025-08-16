from typing import TypedDict, Annotated, List
from langchain_core.documents import Document
import operator
from pydantic import BaseModel, Field
from typing import List,Literal

class AgentState(TypedDict):
    question: str
    rewrite_question: str
    context: Annotated[List[Document], operator.add]
    reranked_context: Annotated[List[Document], operator.add]
    answer: str
    sources : List[str]
    error: str = None
    count:int 



class RewriterQuery(BaseModel):
    """
    This model is used exclusively to structure the refined query
    that will be used to retrieve more relevant content from the knowledge base.

    Always return only the improved query string inside 'refine_query'.
    """
    refine_query: str = Field(
        ...,
        description="A refined user query for retrieving the most relevant context or knowledge."
    )


class QueryDecomposer(BaseModel):
    """
    This model is used to structure the decomposed or refined queries
    for retrieving more relevant content from the knowledge base.

    Always return the list of improved sub-queries inside 'compose_query'.
    """
    compose_query: List[str] = Field(
        ...,
        description="A list (max 5) of refined user sub-queries for retrieving the most relevant context or knowledge."
    )


class Evaluation(BaseModel):
    evaluation_result: Literal["yes", "no"] = Field(
        description="Evaluate whether the response meets the user query requirements. Use 'yes' if it meets the requirements, otherwise 'no'."
    )