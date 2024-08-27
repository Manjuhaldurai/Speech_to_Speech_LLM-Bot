import os
import chainlit as cl
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

HUGGINGFACEHUB_API_TOKEN = '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'
print("API Token:", HUGGINGFACEHUB_API_TOKEN)

# # Loading the Falcon-7B-Instruct model using HuggingFaceHub
# llm = HuggingFaceHub(model_name="tiiuae/falcon-7b-instruct", api_key=os.environ['HUGGINGFACEHUB_API_TOKEN'])

model_id = 'tiiuae/falcon-7b-instruct'
falcon_llm = HuggingFaceHub(huggingfacehub_api_token='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',repo_id=model_id,model_kwargs={"temperature": 0.8, "max_new_tokens":2000})


prompt_template = """
You are a helpful and informative chatbot. Please provide a comprehensive and informative response to the following question:
{question}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["question"])

falcon_chain = LLMChain(llm=falcon_llm, prompt=prompt, verbose=True)

query = "What are the colors in the Rainbow?"
response = falcon_chain.run(query)
print(response)

# @cl.langchain_factory(use_async=False)

# def factory():

#     prompt = PromptTemplate(template=template, input_variables=['question'])
#     falcon_chain = LLMChain(llm=falcon_llm,
#                         prompt=prompt,
#                         verbose=True)

#     return falcon_chain