import speech_recognition as sr
import pyttsx3
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import os  


HUGGINGFACEHUB_API_TOKEN = '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'
print("API Token:", HUGGINGFACEHUB_API_TOKEN)

model_id = 'tiiuae/falcon-7b-instruct'
falcon_llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN, repo_id=model_id, model_kwargs={"temperature": 0.8, "max_new_tokens": 2000})

prompt_template = """
Hi! Here is your answer
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

        # Text Output
        with open("response.txt", "w") as f:
            f.write(response)

        # Text-to-Speech (Audio Output)
        engine.say(response)
        engine.runAndWait()

        # Save Audio Output to MP3
        filename = "response.mp3"
        engine.save_to_file(response, filename)
        engine.runAndWait()
        print(f"Audio saved to: {filename}")

    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))


# Run the function directly
listen_and_respond()