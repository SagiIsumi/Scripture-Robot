class TextBuffer():
    def __init__(self, buffer_size=1)->None:
        self.buffer_size = buffer_size
        self.buffer = []
    
    def set(self, con:list) -> None:
        self.buffer.append(con)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def get(self) ->str:
        text = ''
        for b in self.buffer:
            text = text + str(b) + '\n'
        return text
