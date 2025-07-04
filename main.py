import asyncio
from pathlib import Path

from aioconsole.stream import ainput, aprint
from langchain_core.prompts import ChatPromptTemplate

from bot import Prompt, query, supported_models

PROMPT_FILE = Path(__file__).parent / "prompt.yaml"

model_cache: dict[int, str] = {}
prompt = Prompt.from_yaml(PROMPT_FILE)
openai_api_url = prompt.model_args.get("base_url", "http://localhost:8000/v1")
openai_api_key = prompt.model_args.get("api_key", "")


async def print_supported_models():
    global model_cache
    model_cache = model_cache or {
        i: m
        for i, m in enumerate(await supported_models(openai_api_url, openai_api_key))
    }

    if not model_cache:
        await aprint("No models available.")
    else:
        sorted_models = sorted(model_cache.items(), key=lambda x: x[0])

        max_widths = [0, 0, 0]
        for idx, (i, model) in enumerate(sorted_models):
            col = idx % 3
            s = f"{i}. {model}"
            if len(s) > max_widths[col]:
                max_widths[col] = len(s)

        for row_idx in range(0, len(sorted_models), 3):
            line_parts = []
            for col in range(3):
                item_idx = row_idx + col
                if item_idx < len(sorted_models):
                    i, model = sorted_models[item_idx]
                    s = f"{i}. {model}"
                    line_parts.append(s.ljust(max_widths[col]))
                else:
                    line_parts.append(" " * max_widths[col])
            await aprint("    ".join(line_parts))


async def multiline_input(end_marker: str = "") -> str:
    lines = []
    while True:
        line: str = await ainput("> ")
        if line.strip() == end_marker:
            break
        lines.append(line)
    return "\n".join(lines)


async def setup_chat():
    template = ChatPromptTemplate.model_validate(prompt.model_dump())

    model_args = {**prompt.model_args}
    variables = {
        input_variable: model_args.pop(input_variable)
        for input_variable in template.input_variables
    }

    await aprint("🤖: ", end="", flush=True)
    response = await query(template, variables, **model_args)
    prompt.messages.append({"role": "ai", "content": response})


async def chat_loop():
    await setup_chat()
    while True:
        await aprint()
        user_input: str = await ainput("😀: ")
        if not user_input:
            continue

        match user_input:
            case x if x in ["exit", "quit", "q"]:
                await aprint("Goodbye!")
                break
            case "/model":
                await print_supported_models()
                continue
            case "/setmodel":
                global model_name
                model_name = user_input.split(" ")[1]
                continue
            case x if x in ["/multiline", "/ml"]:
                user_input = await multiline_input()

        prompt.messages.append({"role": "human", "content": user_input})
        template = ChatPromptTemplate.model_validate(prompt.model_dump())

        model_args = {**prompt.model_args}
        variables = {
            input_variable: model_args.pop(input_variable)
            for input_variable in template.input_variables
        }

        await aprint("🤖: ", end="", flush=True)
        response = await query(
            template,
            {**variables, "user_input": user_input},
            **model_args,
        )
        prompt.messages.append({"role": "ai", "content": response})


async def main():
    await aprint(
        "Run /model for more model, /setmodel <model_name> to set model, /ml to input multi-line."
    )
    await aprint()
    await chat_loop()


if __name__ == "__main__":
    asyncio.run(main())
