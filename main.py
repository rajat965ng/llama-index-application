import logging, sys
import os.path
from constant import Model, Quiz, Dir
import torch
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    PromptTemplate,
    ServiceContext,
    StorageContext,
    load_index_from_storage, Response
)
from llama_index.llms import HuggingFaceLLM
from llama_index.embeddings import HuggingFaceEmbedding
from transformers import AutoModelForCausalLM

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class LocalLLM:
    def __init__(self, dir: str):
        print("Initialize PromptTemplate\n")
        query_wrapper_prompt = PromptTemplate(
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{query_str}\n\n### Response:"
        )

        print("Initialize AutoModelForCausalLM\n")
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=Model.LLM_HUGGING_FACE.value,
            device_map="auto",
            offload_folder=Dir.OFFLOAD_DIR.value,
            torch_dtype=torch.float16,
            cache_dir="$HOME/.cache",
            local_files_only=True
        )

        print("Initialize HuggingFaceLLM\n")
        hf_llm = HuggingFaceLLM(
            context_window=2048,
            max_new_tokens=256,
            generate_kwargs={"temperature": 0.25, "do_sample": False},
            query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name=Model.LLM_HUGGING_FACE.value,
            model_name=Model.LLM_HUGGING_FACE.value,
            device_map="auto",
            tokenizer_kwargs={"max_length": 2048},
            model=model,
        )

        print("Initialize ServiceContext\n")
        service_context = ServiceContext.from_defaults(
            chunk_size=1024,
            llm=hf_llm,
            embed_model=HuggingFaceEmbedding(Model.EMBEDDINGS_HUGGING_FACE.value),
        )

        print("Validate Storage Path\n")
        if not os.path.exists(Dir.STORAGE.value):
            print("Storage Path don't Exists\n")
            self.docs = SimpleDirectoryReader(dir).load_data()
            print("Initialize VectorStoreIndex\n")
            self.index = VectorStoreIndex.from_documents(
                self.docs, service_context=service_context
            )
            print("Persist storage_context\n")
            self.index.storage_context.persist()
        else:
            print("Storage Path Exists\n")
            storage_context = StorageContext.from_defaults(persist_dir=Dir.STORAGE.value)
            print("load_index_from_storage\n")
            self.index = load_index_from_storage(storage_context, service_context=service_context)

        print("Init query_engine\n")
        self.query_engine = self.index.as_query_engine()

    def ask(self, text: str) -> Response:
        print("Ask: "+text+"\n")
        return self.query_engine.query(text)


def root():
    llm = LocalLLM(Dir.DATA.value)
    for ques in Quiz:
        result = llm.ask(ques.value)
        print("\nComplete Text: \n")
        print(result)
