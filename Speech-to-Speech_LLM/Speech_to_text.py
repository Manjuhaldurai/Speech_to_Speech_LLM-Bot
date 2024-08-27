import speech_recognition as sr
import pyttsx3
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import chainlit as cl

HUGGINGFACEHUB_API_TOKEN = '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'
print("API Token:", HUGGINGFACEHUB_API_TOKEN)

model_id = 'tiiuae/falcon-7b-instruct'
falcon_llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN, repo_id=model_id, model_kwargs={"temperature": 0.8, "max_new_tokens": 2000})

prompt_template = """
You are a helpful and informative chatbot. Please provide a comprehensive and informative response to the following question:
{question}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["question"])

falcon_chain = LLMChain(llm=falcon_llm, prompt=prompt, verbose=True)

recognizer = sr.Recognizer()
engine = pyttsx3.init()

def listen_and_respond():
    with sr.Microphone() as source:
        print("Speak now...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)

        response = falcon_chain.run(text)
        print("Response:", response)

        engine.say(response)
        engine.runAndWait()

    except sr.UnknownValueError:
        print("Could not understand audio")
        cl.error("Sorry, I couldn't understand what you said.")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
        cl.error("An error occurred while processing your request.")

# Run the function directly
listen_and_respond()