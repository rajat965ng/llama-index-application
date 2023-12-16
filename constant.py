from enum import Enum


class Model(Enum):
    LLM_HUGGING_FACE = "Writer/camel-5b-hf"
    EMBEDDINGS_HUGGING_FACE = "BAAI/bge-small-en-v1.5"


class Quiz(Enum):
    QUES_1 = "Who is Shah Rukh Khan?"
    QUES_2 = "What is the full form of MGNREGA?"
    QUES_3 = "Who was the first Vice President of India?"
    QUES_4 = "What is the full form of ICICI Bank?"


class Dir(Enum):
    OFFLOAD_DIR = "offload"
    STORAGE = "./storage"
    DATA = "./data"
