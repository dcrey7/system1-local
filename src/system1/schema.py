"""Request validation and question options."""

import re
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

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
    levels: list[Label] | None = Field(default=None, min_length=1, max_length=255)
    criteria: dict[Label, str | None] | None = Field(
        default=None, min_length=1, max_length=255
    )

    @field_validator("levels")
    @classmethod
    def unique_levels(cls, levels: list[str] | None) -> list[str] | None:
        if levels is not None and len(set(levels)) != len(levels):
            raise ValueError("Score levels must be unique")
        return levels

    @model_validator(mode="after")
    def resolve_levels(self) -> Self:
        if self.criteria is not None:
            levels = list(self.criteria)
            if all(re.fullmatch(r"[+-]?[0-9]+", key) for key in levels):
                levels.sort(key=int)
            if self.levels is not None and self.levels != levels:
                raise ValueError("Score levels must match the ordered criteria")
            self.levels = levels
        if self.levels is None:
            raise ValueError("Score requires levels or criteria")
        return self


class Noul(QuestionBase):
    type: Literal["noul"]
    criteria: dict[Literal["true", "false"], str | None] | None = Field(
        default=None, min_length=2, max_length=2
    )


Question = Annotated[Choice | Score | Noul, Field(discriminator="type")]


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = None
    state: State
    questions: dict[Label, Question] = Field(min_length=1)
    permutations: int = Field(default=3, ge=1, strict=True)


def options(question: QuestionBase) -> dict[str, str | None]:
    """Return labels in their public order."""
    if isinstance(question, Choice):
        return question.criteria
    if isinstance(question, Score):
        return {
            level: (question.criteria or {}).get(level) for level in question.levels
        }
    if isinstance(question, Noul) and question.criteria is not None:
        return {"yes": question.criteria["true"], "no": question.criteria["false"]}
    return {"yes": None, "no": None}
