import asyncio
from pathlib import Path
from typing import Any

import aiohttp
from aioconsole import aprint
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai.chat_models.base import ChatOpenAI
from pydantic import BaseModel

from tool import read_yaml_file


class Prompt(BaseModel):
    model_args: dict[str, Any]
    messages: list[dict[str, Any]]

    @staticmethod
    def from_yaml(path: Path | str) -> "Prompt":
        return Prompt(**read_yaml_file(path))


async def supported_models(base_uri: str, api_key: str) -> list[str]:
    async with aiohttp.ClientSession() as session:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        async with session.get(f"{base_uri}/models", headers=headers) as response:
            if response.status == 200:
                data: dict[str, Any] = await response.json()
                return [model["id"] for model in data.get("data", [])]
            return []


async def query(
    template: ChatPromptTemplate,
    variables: dict[str, Any],
    **kwargs,
) -> str:
    model = ChatOpenAI(streaming=True, **kwargs) | StrOutputParser()

    prompt_value = template.invoke(variables)
    response = []

    # response = await model.ainvoke(
    #     [(m.type, m.content) for m in prompt_value.to_messages()]
    # )
    async for chunk in model.astream(
        [(m.type, m.content) for m in prompt_value.to_messages()]
    ):
        response.append(chunk)
        for c in chunk:
            await aprint(c, end="", flush=True)
            await asyncio.sleep(0.05)

    return "".join(response)
