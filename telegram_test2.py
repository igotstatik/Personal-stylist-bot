from langchain_community.llms import Ollama
from langchain.schema import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#Для RAG'а
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

#Для генерации картинки
import sd_test3

#Для генерации презы
import ppt_utility2

# Инициализация памяти
memory = ConversationBufferMemory ()

responces_save=[]

import telebot;
import os
from dotenv import load_dotenv
load_dotenv ()

telegram_key = os.environ.get('TELEGRAM_BOT')
bot = telebot.TeleBot(telegram_key);

model = Ollama ( model="llama3.1" )
prompt = ChatPromptTemplate.from_messages (
    [
        (
            "system",
            """You are should be ask, in which city human is now, how old he is, and what gender it have.
            When you will understand, in which city he is, how old he is, and what gender he have (strictly only two options: woman or men), you say, what now you are need all what you are need, and will finish dialogue. For this you are should add word FINISH in your response.
            Your questions to human must to be in Russian language.
            """,
        ),
        ("human", "{question}"),
    ]
)
runnable = prompt | model

responces_save = []

@bot.message_handler(commands=['start'])
def send_welcome(message):
    replika = {}
    replika['role'] = 'assistant'
    replika['content'] = "Привет! Я - твой личный стилист, но мне надо бы немного узнать о тебе. Где ты живешь, сколько тебе лет, и какого ты пола?"
    responces_save.append(replika)

    bot.reply_to(message, "Привет! Я - твой личный стилист, но мне надо бы немного узнать о тебе. Где ты живешь, сколько тебе лет, и какого ты пола?")

    #Тут надо устроить бесконечный обмен сообщениями, пока мы не увидим "FINISH"


@bot.message_handler(func=lambda message: True)
def echo_message(message):

    msg = message.text

    replika = {}
    replika['role'] = 'user'
    replika[
        'content'] = msg
    responces_save.append ( replika )

    stroka = ""
    for event in runnable.stream (
        {"question": msg}
    ):
        print (event)
        stroka = stroka+event

        if "FINISH" in stroka:

            print ( "Ура!" )

            print(responces_save)

            know_weather ( responces_save, message )

            break

    replika = {}
    replika['role'] = 'assistant'
    replika[
        'content'] = stroka
    responces_save.append ( replika )

    bot.reply_to ( message, stroka )

class UserData():
    city = ''
    age = ''
    gender = ''
    temperature = ''

def know_weather(msgs, last_answer):

    def simple_message(content_to_send):
        bot.reply_to (last_answer, content_to_send )

    #Обработаем полученную инфу
    class Data(BaseModel):
        """Base data about user"""

        city: str = Field(description="Город, где живет человек")
        age: str = Field(description="Возраст человека")
        gender: str = Field(description="Пол человека - мужчина или женщина")

    model = ChatOllama(base_url='http://localhost:11434', model="llama3.1")
    structured_model = model.with_structured_output(Data)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Ты должен ответить на вопрос: в каком городе живет человек, сколько ему лет, и какого он пола - мужчина или женщина. Ответ должен быть строго в формате PLACE:город, AGE:возраст, GENDER:пол",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    #print(prompt.invoke({"messages": messages}))

    chain = prompt | structured_model
    result = chain.invoke({"messages": msgs})

    print(result)
    UserData.city = result.city
    UserData.age = result.age
    UserData.gender = result.gender

    simple_message("Данные обработаны")

    #Займемся погодой

    import os
    from dotenv import load_dotenv
    load_dotenv ()
    os.environ['USER_AGENT'] = 'myagent'

    from langchain_community.tools.bing_search.tool import BingSearchRun, BingSearchResults
    from langchain_community.utilities.bing_search import BingSearchAPIWrapper

    api_wrapper = BingSearchAPIWrapper()
    tool = BingSearchRun(api_wrapper=api_wrapper, max_results= 10)
    tools = [tool]
    model = ChatOllama ( base_url='http://localhost:11434', model="llama3.1" )
    model_with_tools = model.bind_tools(tools)

    from langgraph.prebuilt import create_react_agent
    agent_executor = create_react_agent(model_with_tools, tools)
    response = agent_executor.invoke({"messages": [HumanMessage(content="Respond, what the weather in the city " +result.city)]})
    last_msg = response['messages'][-1].content

    class WeatherData(BaseModel):
        """Weather data model"""
        temperature: str = Field(description="Current temperature in the city in Celsius degrees")

    model = ChatOllama(base_url='http://localhost:11434', model="llama3.1")
    structured_model = model.with_structured_output(WeatherData)

    print(last_msg)

    prompt = ChatPromptTemplate.from_messages (
        [
            (
                "system",
                """From this information, you are must to understand, what is the high daily temperature in city today.                  
                Your response MUST to be EXACT SINGLE number""",
            ),
            (
                "human",
                last_msg,
            ),
        ]
    )

    chain = prompt | structured_model
    rst = chain.invoke({"messages": msgs})

    print ( rst )
    UserData.temperature = rst.temperature
    print ( "Город - ", UserData.city)
    print ( "Возраст - ", UserData.age)
    print ( "Пол - ", UserData.gender )
    print ( "Температура днем - ", UserData.temperature)

    simple_message("Прогноз погоды узнал")

    model = ChatOllama ( base_url='http://localhost:11434', model="gemma2" )

    #Часть RAG'а
    #Имитируем работу с внутренней информацией
    loader = WebBaseLoader ( "https://bowandtie.ru/35-sovetov-po-muzhskomu-stilyu/" )
    data = loader.load ()
    text_splitter = RecursiveCharacterTextSplitter ( chunk_size=500, chunk_overlap=0 )
    all_splits = text_splitter.split_documents ( data )
    local_embeddings = OllamaEmbeddings ( model="nomic-embed-text" )
    vectorstore = Chroma.from_documents ( documents=all_splits, embedding=local_embeddings )

    def format_docs(docs):
        return "\n\n".join ( doc.page_content for doc in docs )

    RAG_TEMPLATE = """
    You are great fashion designer. You are MUST to use the following pieces of retrieved context to answer the question.
    <context>
    {context}
    </context>    
    Beside your own recommendations, you are should to choose one recommendation from context, ensure, what your recommendations is aligned with it, and make remarks about this recommendation.
    Your response must to be in Russian and have note about current weather
    What you can suggest to wear in this situation:
    {question}"""

    rag_prompt = ChatPromptTemplate.from_template ( RAG_TEMPLATE )

    retriever = vectorstore.as_retriever ()

    qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough ()}
            | rag_prompt
            | model
            | StrOutputParser ()
    )

    question = "My age is "+UserData.age+" my gender is "+ UserData.gender + " and temperature of air at my place about "+UserData.temperature + " degrees Celsius"
    #question = "My age is " + UserData.age + " my gender is " + UserData.gender + " and temperature of air at my place about 39 degrees Celsius"
    fashion_verdict = qa_chain.invoke(question)

    simple_message ( "Совет по стилю сформировал" )

    img_request = "Нарисуй ОДНОГО человека - "+UserData.gender+" - красивой внешности. Человеку "+ UserData.age+ " лет. На картинке ОБЯЗАТЕЛЬНО ТРЕБУЕТСЯ изобразить "+UserData.gender+". "+fashion_verdict
    print (img_request)
    #А теперь для фешен-вердикта (Ну и плюс "Красивый gender + age) вызываем gen(prompt) из sd_test3
    sd_test3.gen(img_request)
    #Пока забудем про картинку

    simple_message("Иллюстрацию подобрал")

    #А теперь перепиши вердикт под powerpoint
    PRESENTATION_TEMPLATE = """
    "You are a power point presentation specialist. You are asked to create
    the content for a slide for a presentation about fashion style.  
    Structure the information in a way that it can be put in a power point presentation slide. 
    Each slide should have a main text box for the main ideas and smaller text box for comments about them. 
    Return the structured information as a JSON as follows. 
    JSON must to contain two keys: main_box and small_box. In each you must to put strictly list of paragraphs as strings.
    Your answer MUST only contain the JSON - no any markdown formatting. 
    Here is information for the slide:
    {information}"""

    presentation_prompt = ChatPromptTemplate.from_template ( PRESENTATION_TEMPLATE )

    chain = presentation_prompt | model | StrOutputParser ()
    information = fashion_verdict
    res = chain.invoke(information)
    print (res)

    ppt_utility2.create_ppt(res)

    simple_message("В презентацию упаковал")

    bot.reply_to (last_answer, fashion_verdict)
    doc = open ( 'test.pptx', 'rb' )
    bot.send_document ( last_answer.from_user.id, doc)

bot.polling(none_stop=True, interval=0)


