from pathlib import Path
from typing import Any, Callable, Optional, Self

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from langchain_openai.chat_models.base import BaseChatModel
from pydantic import model_validator

AGENTS_PATH = Path(__file__).parent / "agents"
OPENAI_BASE_URL = "http://localhost:8750/v1"


BotInput = str
BotOutput = str


class Bot(RunnableSerializable[BotInput, BotOutput]):
    prologue: Optional[str] = None
    model_id: Optional[str] = None
    model_url: Optional[str] = None
    model_provider: Optional[str] = None
    model_api_key: Optional[str] = None
    model_kwargs: Optional[dict[str, Any]] = None
    messages: Optional[list[tuple[str, str]]] = None
    model: Optional[BaseChatModel] = None

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        from yaml import safe_load

        path = AGENTS_PATH / "assistant.yaml"
        data = safe_load(path.read_text(encoding="utf-8"))

        if isinstance(data, dict):
            self.prologue = data.get("prologue", None)
            model_kwargs = data.get("model_kwargs", {})
            if not isinstance(model_kwargs, dict):
                raise ValueError("Invalid model_kwargs")

            self.model_id = model_kwargs.pop("model", "gpt-3.5-turbo")
            self.model_url = model_kwargs.pop("base_url", OPENAI_BASE_URL)
            self.model_provider = model_kwargs.pop("model_provider", "openai")
            self.model_api_key = model_kwargs.pop("api_key", None)
            self.model_kwargs = model_kwargs

            messages = (
                data["messages"]
                if data["messages"]
                else [{"system": "You are a helpful assistant."}]
            )
            if not isinstance(messages, list):
                raise ValueError("Invalid messages")

            self.messages = []
            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError("Invalid message")
                role = message["role"]
                self.messages.append(
                    ("human" if role == "user" else role, message["content"])
                )
        else:
            raise ValueError(f"Invalid {path}")
        return self

    def _create_model(self) -> BaseChatModel:
        if not self.model:
            if not self.model_api_key:
                raise ValueError("No model_api_key")

            from langchain.chat_models.base import init_chat_model

            self.model = init_chat_model(
                self.model_id,
                model_provider=self.model_provider,
                base_url=self.model_url,
                api_key=self.model_api_key,
                **self.model_kwargs,
            )
        return self.model

    def invoke_stream(
        self,
        input: BotInput,
        write_function: Callable[[BotOutput], None],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ):
        model = self._create_model()

        self.messages.append(("human", input))
        prompt = ChatPromptTemplate.from_messages(self.messages)
        chain = prompt | model | StrOutputParser()

        full_messages = []
        for chunk in chain.stream({}, config=config, **kwargs):
            full_messages.append(chunk)
            for c in chunk:
                write_function(c)

        self.messages.append(("ai", full_messages))

    def invoke(
        self,
        input: BotInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> BotOutput:
        model = self._create_model()

        self.messages.append(("human", input))
        prompt = ChatPromptTemplate.from_messages(self.messages)
        chain = prompt | model | StrOutputParser()

        ai_output = chain.invoke({}, config=config, **kwargs)
        self.messages.append(("ai", ai_output))

        return ai_output
