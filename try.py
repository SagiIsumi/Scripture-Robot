import pyaudio
import speech_recognition as sr
import queue
from datetime import datetime
from pathlib import Path
import wave
recognizer = sr.Recognizer()

rate = 16000
chunk = 1024

class audio_record:
    def __init__(self,rate,chunk):
        self.rate=rate
        self.chunk=chunk
        self.buffer=queue.Queue()
        self.close=True
    def __enter__(self):
        self.audio_interface = pyaudio.PyAudio()
        self.audio_stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=self.fill_buffer
        )
        self.close=False
        return self
    def __exit__(self,type,value,traceback):
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.closed = True
        self.buffer.put(None)
        self.audio_interface.terminate()
    def fill_buffer(self,in_data,frame_count,time_info:dict,status_flags)->tuple:
        #the callback function must return tuple
        self.buffer.put(in_data)
        # print("in_data:", in_data)
        # print("frame_count:", frame_count)
        # print("time_info:", time_info)
        # print("status_flags:", status_flags)
        return (None, pyaudio.paContinue)
    def generator(self):
        while not self.close:
            chunk=self.buffer.get()
            if chunk==None:
                return
            yield chunk


def main():
    print("listening...")
    with audio_record(rate=rate,chunk=chunk) as ar:
        audio_generator = ar.generator()
        frames=[]
        for audio_data in audio_generator:
            frames.append(audio_data)
            if len(frames)<50:
                continue
            audio_chunk = b"".join(frames)
            audio_chunk=sr.AudioData(audio_chunk, rate, 2)
            
            try:
                # time_object=datetime.now()
                # currentTime = time_object.strftime("%d-%m-%y_%H-%M-%S")
                # audio_path=Path("./audio_file")/Path(currentTime+".wav")
                # audio_path=str(audio_path)
                # with wave.open(audio_path,"wb") as wf:
                #     wf.setnchannels(1)
                #     wf.setsampwidth(2)
                #     wf.setframerate(rate)
                #     wf.writeframes(b"".join(frames))

                text = recognizer.recognize_google(audio_chunk,language='zh-TW')
                frames=[]
                print(f"You said: {text}")
            except sr.UnknownValueError:
                print("hi")
                frames=[]
                pass  
            except sr.RequestError as e:
                print(f"Error: {e}")
if __name__ == "__main__":
    main()
