import sys
from pathlib import Path


class Unbuffered:
    _instance = None

    def __new__(cls, file_path: Path = None):
        if cls._instance is None:
            if file_path is None:
                raise Exception("Please provide file_path")

            cls._instance = super().__new__(cls)
            cls._instance.file_stream = open(file_path, "a")
            cls._instance.stream = sys.stdout
        return cls._instance

    def write(self, data: str):
        self.stream.write(data)
        self.stream.flush()
        self.file_stream.write(data)
        self.file_stream.flush()

    def close(self):
        self.file_stream.flush()
        self.file_stream.close()
