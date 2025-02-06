import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from flask import Flask, request
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

endpoint = os.getenv("OTEL_EXPORTER_OTLP_HTTP_ENDPOINT", "http://127.0.0.1:4318")
tracer_provider = trace_sdk.TracerProvider()
trace_api.set_tracer_provider(tracer_provider)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

app = Flask(__name__)
LangChainInstrumentor().instrument()


# use a local model
model = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    model="deepseek-r1-distill-qwen-7b",
    temperature=0,
    api_key="not-needed"
)

# model = ChatOpenAI(model="gpt-3.5-turbo")

embeddings_model = OpenAIEmbeddings(
    check_embedding_ctx_length=False,
    model="text-embedding-nomic-embed-text-v1.5",
    base_url="http://localhost:1234/v1",
    openai_api_key="not-needed"
)

db = Chroma(
    persist_directory="../my_embeddings",
    embedding_function=embeddings_model
)

store = {}
config = {"configurable": {"session_id": "test"}}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(model, get_session_history)

@app.route("/askquestion", methods=['POST'])
def ask_question():

    data = request.json
    question = data.get('question')

    # find the documents most similar to the question that we can pass as context
    context = db.similarity_search(question)

    response = with_message_history.invoke(
        [
            SystemMessage(
                content=f'Use the following pieces of context to answer the question: {context}'
            ),
            HumanMessage(
                content=question
            )
        ],
        config=config
    )

    return response.content

