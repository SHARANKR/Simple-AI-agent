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

chain = prompt_1 | model | parser | prompt_2 | model | parser

result = chain.invoke({"topic":"vegetables"})

print(result)

# chain.get_graph().print_ascii()