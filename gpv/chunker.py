import semchunk


class Chunker:
    def __init__(self, chunk_size: int, model_name: str = "gpt-4"):
        self.model_name = model_name
        self.chunk_size = chunk_size

    def chunk(self, text: list[str]) -> list[list[str]]:
        chunker = semchunk.chunkerify(self.model_name, self.chunk_size)
        return chunker(text)


if __name__ == "__main__":
    chunker = Chunker(20)
    texts = [
        "I have been at work since 7 this morning.  I got up at 6 this morning.  I went to bed at 11 last night.  I am pooped, and its a long day.  And, no word from TD since Wednesday (feeling a little insecure about that).",
        "I had some good time with Renada last night... I went up to her house to help her with a school project.  It was fun!  We spraypainted some stuff, went on the regular Thursday night Walmart run, chatted a lot.",
    ]
    chunks = chunker.chunk(texts)
    print(chunks)