from tqdm import tqdm
import json
from gen import gen
from typing import List


async def evol_width(prompt: str):
    return await gen(
        f"""AIへの指示プロンプトの作成者として行動してください。
あなたの目標は、#Given Prompt#からインスピレーションを得て、まったく新しいプロンプトを作成することです。
この新しいプロンプトは#Given Prompt#と同じジャンルに属している必要がありますが、もっと珍しいものでなくてはなりません。
#Created Prompt#の長さと難易度は#Given Prompt#と同等であるべきです。
#Created Prompt#は合理的であり、破綻があってはなりません。
#Given Prompt#:
{prompt}
#Created Prompt#:(新しいpromptのみを応答してください。)"""
    )


async def evol_depth(prompt: str):
    return await gen(
        f"""AIへの指示プロンプトの書き換えをしてください。
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
#Rewritten Prompt#(改善されたpromptのみを応答してください。):
"""
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

あなたの判断(そう思う場合は"はい"とだけ、思わない場合は理由を出力してください): """
    )


async def evol_null():
    return await gen(
        f"""AIアシスタントに指示するプロンプトの作成者として行動してください。
あなたの目標は、無から人間が考えそうな新しいプロンプトを100文字以内で作成することです。
#Created Prompt# は長さと難易度は短く、答えるのは極めて簡単です。しかし、簡単に思いつくようなものではありません。
#Created Prompt# はテキストしか扱えないAIが自身の知識や判断力で答えられるものです。AIは外部情報にアクセスできません。
#Created Prompt# は人間の立場に立った思考や動作を必要としません。
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
```

#Given Prompt#:
{prompt}
#Rewritten Prompt#(改善されたpromptのみを応答してください。):
"""
    )


async def evol(prompt: str, steps: int = 3):
    prompt = await evol_width(prompt)
    if prompt is None:
        return None
    result = await evol_judge(prompt)
    if result == "はい" or result == "はい。":
        pass
    else:
        print(f"Failed: {result}")
        return None
    for _ in range(steps):
        prompt = await evol_depth(prompt)
        if prompt is None:
            return None
    prompt = await evol_flatten(prompt)
    result = await evol_judge(prompt)
    if result == "はい" or result == "はい。":
        pass
    else:
        print(f"Failed: {result}")
        return None
    return prompt


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
                if r:
                    print(f"Generated: {r}")
                    res.append(r)
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
