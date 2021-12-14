
from typing import List, Tuple, Dict

from pydantic import BaseModel, Field
from transformers import GPT2Config, PreTrainedTokenizerBase

class Args(BaseModel):
    args: Tuple = Field(None, description="The args passed to `from_pretrained`")
    kwargs: Dict = Field(None, description="The kwargs dict passed to `from_pretrained`")

class PereiraResult(BaseModel):
    scores: list = Field(..., description="The scores for this result")
    model_args: Args = Field(None, description="The args and kwargs passed to `from_pretrained` for the model")
    model_config: GPT2Config = Field(None, description="The model's config")
    tokenizer_args: Args = Field(..., description="The args and kwargs passed to `from_pretrained` for the tokenizer")

    class Config:
        arbitrary_types_allowed = True

class PereiraResultSet(BaseModel):
    results: List[PereiraResult]
