import glob
import json
from gen import gen


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
        f"""以下の質問に対して、以下の回答が正確もしくは合理的であるかを\"はい\"か\"いいえ\"で回答してください。
質問: {q}
回答: {a}"""
    )


async def main():
    import time
    from tqdm import tqdm

    sentences = []

    for f in glob.glob("./result/prompt_*.json"):
        with open(f, "r") as r:
            sentences.extend(json.loads(r.read()))
    res = []
    for s in tqdm(sentences):
        r = await response(s)
        if "不可能" in r:
            print(f"Skip: {s}")
            continue
        c = await check(s, r)
        if c == "はい" or c == "はい。":
            print(f"質問: {s}\n回答: {r}")
            res.append({"user": s, "model": r})
        else:
            print(f"Skip: {s}")
    with open(f"./result/response_{time.time()}.json", "w") as w:
        w.write(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
