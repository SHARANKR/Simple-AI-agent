from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

prompt_1 = PromptTemplate(
    template= "Generate a short notes from the following {text}",
    input_variables= ["text"]
)

prompt_2 = PromptTemplate(
    template= "Generate 5 short question answers from the following text \n {text}",
    input_variables= ["text"]
)

prompt_3 = PromptTemplate(
    template= "Merge the provided notes and quiz into a single document \n notes -> {notes} and {quiz}",
    input_variables= ["notes", "quiz"]
)

model = ChatGoogleGenerativeAI(model = "gemini-3-flash-preview")

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes": prompt_1 | model | parser,
    "quiz": prompt_2 | model | parser
})

merge_chain = prompt_3 | model | parser 

chain = parallel_chain | merge_chain

text = """
The Transformer architecture, introduced in the groundbreaking paper Attention Is All You Need by researchers at Google, has fundamentally transformed the field of Artificial Intelligence and Natural Language Processing. Unlike traditional sequential models such as RNN and LSTM, Transformers rely on a powerful mechanism called self-attention, which allows the model to understand relationships between words regardless of their distance in a sentence while also enabling parallel processing, making training significantly faster and more efficient. The architecture typically consists of components such as token embeddings, positional encoding to preserve word order, multi-head self-attention to capture different contextual relationships, feed-forward neural networks for deeper representation learning, and residual connections with layer normalization to stabilize training. This architecture has become the backbone of many modern AI systems, powering models like BERT, GPT, and T5, which are widely used in applications such as conversational AI, intelligent search, code generation, AI assistants, and autonomous AI agents, making Transformers one of the most important innovations driving today’s AI revolution.
"""

result = chain.invoke({"text":text})

print(result)

# chain.get_graph().print_ascii()