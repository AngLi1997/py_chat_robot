import base64
from io import BytesIO

from gradio import Interface, Image
from gradio.components.textbox import Textbox
from PIL.Image import Image as PILImage
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from bmos.model_constants import LLAVA

llm = ChatOllama(model=LLAVA, verbose=True, base_url="192.168.110.19:11434")

def image_to_base64(image:PILImage, suffix="png"):
    buffer = BytesIO()
    image.save(buffer, format=suffix)
    bin_data = buffer.getvalue()
    base64_str = base64.b64encode(bin_data).decode()
    buffer.close()
    return base64_str


def extract_image(image:PILImage):
    result = ""
    base64_str = image_to_base64(image)
    messages = [
        SystemMessage(f"""
        你是一个图像识别机器人，请根据输入的图片base64信息回答输入的问题
        """),
        HumanMessage(f"""
            图片base64信息： 
            {base64_str}
            使用中文描述图片的内容是什么
        """)
    ]
    for e in llm.stream(messages):
        result += e.content
        yield result


if __name__ == '__main__':
    gr = Interface(fn=extract_image,
                   title="图像OCR",
                   inputs=[Image(label="上传图片", type="pil")],
                   outputs=[Textbox(label="说明")])
    gr.launch()