from argparse import ArgumentParser
from lora_infer import infer, load_model

parser = ArgumentParser()
parser.add_argument("--steps", type=int, required=True)
parser.add_argument("--max", type=int, default=128)
args = parser.parse_args()

zero = load_model(-1)
lora = load_model(args.steps)


def infer_together(inst: str, max: int):
    print("Zero: ", end="")
    infer(zero, inst, max)
    print(f"Step{args.steps}: ", end="")
    infer(lora, inst, max)


while True:
    inp = input("User: ")
    if inp == "exit":
        break
    infer_together(inp, args.max)
