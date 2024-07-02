from transformers.models.gemma.tokenization_gemma_fast import GemmaTokenizerFast
from typing import List, Dict, Union
from aiohttp import ClientSession
import os

HOST = os.environ.get("API_HOST") or "http://localhost:8080"

tok: GemmaTokenizerFast = GemmaTokenizerFast.from_pretrained("google/gemma-2-27b-it")


async def chat(prompt: List[Dict[str, str]]) -> Union[str, None]:
    comp = tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)[
        len(tok.bos_token) :
    ]
    async with ClientSession() as session:
        async with session.post(
            f"{HOST}/completions",
            json={"prompt": comp, "max_tokens": 1024},
        ) as resp:
            resp.raise_for_status()
            return (await resp.json())["content"]


async def gen(prompt: str):
    res = await chat(
        [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )
    if res is None:
        return None
    return res.strip()
