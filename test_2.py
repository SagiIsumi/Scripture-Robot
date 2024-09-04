from pathlib import Path
from openai import OpenAI
from datetime import datetime

time_object=datetime.now()
currentTime = time_object.strftime("%d-%m-%y_%H-%M-%S")
audio_path=Path("./audio_file")/Path(currentTime+".mp3")
txt_file=Path("./conversation_history/response_23-21-11.txt")

with open(txt_file,encoding="utf-8") as f:
    a=f.readline()
    input_text=f.read()
client= OpenAI()
with client.audio.speech.with_streaming_response.create(
  model="tts-1",
  voice="nova",
  input=input_text,
) as response:
    print(type(response))
    response.stream_to_file(audio_path)