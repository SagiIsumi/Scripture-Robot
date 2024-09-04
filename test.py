from GPTpackages.GPTopenai import GPTopenai
from GPTpackages.PromptTemplate import PromptTemplate
from GPTpackages.ImageBufferMemory import ImageBufferMemory, encode_image
from GPTpackages.TextBuffer import TextBuffer
from datetime import datetime
from pathlib import Path
import configparser
from langchain.memory import ConversationBufferMemory

API_KEY="sk-proj-ebjcEZYloUZnX0nLGaXdT3BlbkFJAeGVT80bSncVYWOqmaFZ"
Model="gpt-4o"

def showtime():
    currentDateAndTime = datetime.now()
    currentTime = currentDateAndTime.strftime("%H-%M-%S")
    print(currentTime)

if __name__ == '__main__':
    path = Path('./prompts') / Path('vision_prompt.txt')
    agent = GPTopenai(
        openai_api_key=API_KEY, 
        #model=Model,
        prompt=PromptTemplate(path), 
        text_memory=TextBuffer(),
        img_memory=ImageBufferMemory()
    )
    text_dict = {'what': '你看到了幾個杯子?他們是什麼顏色的?'}
    path = Path('./input_pictures') / Path('IMG_8405.jpg')
    showtime()
    img_list = [encode_image(path)]
    result = agent.run(text_dict, img_list)
    showtime()
    print(result)
