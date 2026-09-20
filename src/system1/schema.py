"""Request validation and question options."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

State = str | dict[str, JsonValue] | list[JsonValue]
Label = Annotated[str, Field(min_length=1)]


class QuestionBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instructions: str


class Choice(QuestionBase):
    type: Literal["choice"]
    criteria: dict[Label, str | None] = Field(min_length=1, max_length=255)


class Score(QuestionBase):
    type: Literal["score"]
    levels: list[Label] = Field(min_length=1, max_length=255)

    @field_validator("levels")
    @classmethod
    def unique_levels(cls, levels: list[str]) -> list[str]:
        if len(set(levels)) != len(levels):
            raise ValueError("Score levels must be unique")
        return levels


class Noul(QuestionBase):
    type: Literal["noul"]


Question = Annotated[Choice | Score | Noul, Field(discriminator="type")]


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: State
    questions: dict[Label, Question] = Field(min_length=1)
    permutations: int = Field(default=3, ge=1, strict=True)


def options(question: QuestionBase) -> dict[str, str | None]:
    """Return labels in their public order."""
    if isinstance(question, Choice):
        return question.criteria
    if isinstance(question, Score):
        return dict.fromkeys(question.levels)
    return {"yes": None, "no": None}
