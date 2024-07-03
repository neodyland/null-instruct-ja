from typing import List, Dict, Union
from aiohttp import ClientSession
from argparse import ArgumentParser
from lora_infer import infer, load_model
from datasets import load_dataset
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument("--host", default="http://localhost:8080")
parser.add_argument("--steps", type=int, required=True)
parser.add_argument("--max", type=int, default=256)
args = parser.parse_args()


async def chat(prompt: List[Dict[str, str]]) -> Union[str, None]:
    async with ClientSession() as session:
        async with session.post(
            f"{args.host}/v1/chat/completions",
            json={
                "messages": prompt,
                "max_tokens": 1,
                "grammar": 'root ::= "1"|"2"|"3"|"4"|"5"',
            },
        ) as resp:
            resp.raise_for_status()
            return (await resp.json())["choices"][0]["message"]["content"].replace(
                "<end_of_turn>", ""
            )


async def eval_one(q: str, a: str, aspect: str, max: int):
    pred = infer(q, max, False)
    res = await chat(
        [
            {
                "role": "user",
                "content": f"""あなたは採点者です。

問題, 正解例, 採点基準, 回答 が与えられます。

採点基準と正解例を参考にして、回答を1,2,3,4,5の5段階で採点し、数字のみを出力してください。

# 問題
{q}

# 正解例
{a}

# 採点基準
基本的な採点基準
- 1点: 誤っている、 指示に従えていない
- 2点: 誤っているが、方向性は合っている
- 3点: 部分的に誤っている、 部分的に合っている
- 4点: 合っている
- 5点: 役に立つ

基本的な減点項目
- 不自然な日本語: -1点
- 部分的に事実と異なる内容を述べている: -1点
- 「倫理的に答えられません」のように過度に安全性を気にしてしまっている: 2点にする

問題固有の採点基準
{aspect}

# 回答
{pred}""",
            }
        ]
    )
    if res is None:
        return None
    return float(res.strip()), pred


async def main():
    import time
    import json

    load_model(args.steps)
    ds = load_dataset("elyza/ELYZA-tasks-100", split="test")
    score = 0.0
    res = []
    for entry in tqdm(ds):
        q = entry["input"]
        a = entry["output"]
        aspect = entry["eval_aspect"]
        s, pred = await eval_one(
            q,
            a,
            aspect,
            args.max,
        )
        score += s
        print(f"Score: {s}\nQuestion: {q}\nAnswer: {a}\nPred: {pred}")
        res.append(
            {"score": int(s), "question": q, "pure_answer": a, "pred_answer": pred}
        )
    print(f"Score: {score / len(ds)}")
    with open(f"./result/eval_{time.time()}.json", "w") as w:
        w.write(
            json.dumps(
                {"progress": res, "score": score / len(ds), "steps": args.steps},
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
