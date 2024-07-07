from ds import ds
import gradio as gr

index = 0


def submit_user(content):
    ds["user"][index] = content


def submit_model(content):
    ds["user"][index] = content


def submit_reject(content):
    ds["reject"][index] = content


def get_user():
    return ds["user"][index]


def get_model():
    return ds["model"][index]


def get_reject():
    return ds["reject"][index]


def get_title():
    return f"{index + 1}/{len(ds)}"


def submit_next():
    global index
    index += 1
    if len(ds) == index:
        index = 0
    return ds["user"][index], ds["model"][index], ds["reject"][index], get_title()


def submit_back():
    global index
    index -= 1
    if len(ds) == -1:
        index = len(ds) - 1
    return ds["user"][index], ds["model"][index], ds["reject"][index], get_title()


def submit_save():
    ds.save_to_disk("./result/edited")


with gr.Blocks() as demo:
    title = gr.Label(value=get_title)
    with gr.Row():
        i1 = gr.TextArea(get_user, label="User Input")
        i2 = gr.TextArea(get_model, label="Model Output")
        i3 = gr.TextArea(get_reject, label="Model Reject Output")
    with gr.Row():
        back = gr.Button(
            value="Back",
        )
        next = gr.Button(
            value="Next",
        )
        gr.Button(value="Save").click(submit_save)
    back.click(submit_back, outputs=[i1, i2, i3, title])
    next.click(submit_next, outputs=[i1, i2, i3, title])

demo.launch(server_name="0.0.0.0")
