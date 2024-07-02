from tqdm import tqdm
import json
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


async def evol_width(prompt: str):
    return await gen(
        f"""プロのAIへの指示プロンプトの作成者として行動してください。
あなたの目標は、#Given Prompt#からインスピレーションを得て、まったく新しいプロンプトを作成することです。
この新しいプロンプトは#Given Prompt#と同じジャンルに属している必要がありますが、もっと珍しいものでなくてはなりません。
#Created Prompt#の長さと難易度は#Given Prompt#と同等であるべきです。
#Created Prompt#は合理的、高品質であり、破綻があってはなりません。
#Created Prompt#は理性のないテキストしか扱えない機械が回答できるものでなくてはなりません。
#Given Prompt#:
{prompt}
#Created Prompt#:(新しいpromptのみを応答してください。)"""
    )


async def evol_depth(prompt: str):
    return await gen(
        f"""あなたはAIへの指示プロンプトの書き換えに特化したプロです。
あなたの目的は、与えられたプロンプトをより複雑なバージョンに書き換えて、プロンプト応答の難易度を上昇させることです。
書き換える際には、#Given Prompt#の非テキスト部分（例:表やコード）を省略しないでください。また、#Given Prompt#の入力も省略しないでください。
以下の方法を使用して与えられたプロンプトを複雑にしてください。

```
- #Given Prompt#にもう一つ、制約/要件を加える。
- #Rewritten Prompt#が冗長にならないように最善を尽くす
- #Rewritten Prompt#は#Given Prompt#に10から20語追加したものとする。
```

#Given Prompt#:
{prompt}
#Rewritten Prompt#(改善されたpromptのみを応答してください。):"""
    )


async def evol_judge(prompt: str):
    return await gen(
        f"""以下のプロンプトは十分に質の高いプロンプトであるといえますか?
質が高いとは日本語として正しく、破綻なく、理性のない機械が正確に回答できる質問であることを指します。
例えば、質問形式でないものは質の高いプロンプトとは言えません。
ほかにも、ユーザー独自の状況にアクセスする必要のある質問は、正確に回答できないため質が高いとは言えません。
質問として成立していないものも、質が高いとは言えません。
このAIはテキストしか扱えないことにも留意してください。

プロンプト：{prompt}

あなたの判断(そう思う場合は**\"はい\"**とだけ応答。思わない場合は理由を応答してください): """
    )


async def evol_null():
    return await gen(
        f"""あなたはプロのAIアシスタントに指示するプロンプトの作成者です。
あなたの目標は、無から人間が考えそうな新しいプロンプトを100文字以内で作成することです。
#Created Prompt# は長さと難易度は短く、答えるのは極めて簡単です。しかし、簡単に思いつくようなものではありません。
#Created Prompt# はテキストしか扱えない理性のない機械が自身の知識や判断力で答えられるものです。その機械は外部情報にアクセスできません。
#Created Prompt# は現実的です。
生成されたAIに対する指示プロンプトだけを応答してください。
#Created Prompt#:"""
    )


async def evol_flatten(prompt: str):
    return await gen(
        f"""AIへの指示プロンプトの書き換えをしてください。
あなたの目的は、以下の方法を使用して与えられたプロンプトをより正確な形に修正することです。
```
- 誤字、脱字がある場合それを修正する。
- 質問として破綻している場合、質問の意図を類推し、要求される回答が生成できるように質問を修正する。
- 理性のない機械が回答できるようにする。
```

#Given Prompt#:
{prompt}
#Rewritten Prompt#(改善されたpromptのみを応答してください。):
"""
    )


async def length(q: str, a: str):
    return await gen(
        f"""あなたはとても賢く饒舌なAIとして、以下の質問および饒舌でないAIが生成した回答を基に、より饒舌な回答を生成します。
しかしながら、饒舌で賢いあなたは回答しているうちに、回答が不正確であることに気が付くかもしれません。その場合は\"不正確です\"と応答してください。
それ以外の場合は饒舌な回答だけを応答してください。

質問: {q}
饒舌でないAIが生成した回答: {a}

より饒舌に修正された回答または\"不正確です\": """
    )


async def response(prompt: str):
    return await gen(
        f"""あなたはとても賢いAIとして、以下の質問に対して回答します。
しかしながら、質問の質はよいとは限らず、回答が不可能なものも含まれます。
回答が可能である場合は質問に対して回答し、回答が不可能、または正確な回答を導き出すことが不可能である場合は、\"不可能\"とだけ回答してください。
回答が可能である場合は回答のみを出力してください。

質問: {prompt}

回答もしくは\"不可能\": """
    )


async def check(q: str, a: str):
    return await gen(
        f"""以下の質問に対して、以下の回答が十分質が高いかを\"はい\"か\"いいえ\"で回答してください。
質が高いとは日本語として正しく、破綻ないことを指します。
回答として成立していないものは、質が高いとは言えません。
このAIはテキストしか扱えないことにも留意してください。

質問: {q}
回答: {a}"""
    )


async def evol(prompt: str, steps: int = 3):
    prompt = await evol_width(prompt)
    if prompt is None:
        return {"failed": "width"}
    result = await evol_judge(prompt)
    if result != "はい" and result != "はい。":
        return {"failed": "judge", "reason": result}
    for _ in range(steps):
        prompt = await evol_depth(prompt)
        if prompt is None:
            return None
    prompt = await evol_flatten(prompt)
    result = await evol_judge(prompt)
    if result != "はい" and result != "はい。":
        return {"failed": "judge2", "reason": result}
    r = await response(prompt)
    if "不可能" in r:
        return {"failed": "response", "reason": r}
    r = await length(prompt, r)
    if "不正確です" in r:
        return {"failed": "length", "reason": r}
    c = await check(prompt, r)
    if c != "はい" and c != "はい。":
        return {"failed": "check", "reason": c}
    return {"user": prompt, "model": r}


async def evol_lot(
    max_count_for_evol: int = 1, max_count_for_null: int = 1, evol_steps: int = 3
) -> List[str]:
    res = []
    with tqdm(total=max_count_for_evol * max_count_for_null) as pbar:
        for _ in range(max_count_for_null):
            prompt = await evol_null()
            if prompt is None:
                pbar.update(max_count_for_evol)
                continue
            for _ in range(max_count_for_evol):
                r = await evol(prompt, evol_steps)
                if "user" in r:
                    m = r["model"]
                    u = r["user"]
                    print(f"Question: {u}\nAnswer: {m}")
                    res.append(r)
                else:
                    f = r["failed"]
                    reason = r["reason"] if "reason" in r else "理由不明"
                    print(f"Failed to generate in {f} phaze with reason {reason}")
                pbar.update(1)
    return res


async def main():
    import time
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--max_count_for_evol", default=3, type=int)
    parser.add_argument("--max_count_for_null", default=300, type=int)
    parser.add_argument("--evol_steps", default=1, type=int)
    args = parser.parse_args()
    res = await evol_lot(
        max_count_for_evol=args.max_count_for_evol,
        max_count_for_null=args.max_count_for_null,
        evol_steps=args.evol_steps,
    )
    with open(f"./result/prompt_{time.time()}.json", "w") as w:
        w.write(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
