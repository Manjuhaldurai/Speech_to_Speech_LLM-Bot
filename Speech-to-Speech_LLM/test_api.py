import os
import requests

def test_api_key():
    api_key = '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'
    url = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
    headers = {"Authorization": f"Bearer {api_key}"}

    data = {
        "inputs": "What is the capital of France?",
        "max_length": 100,
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        print("API key is valid.")
        print("Response:", response.json())
    else:
        print("API key is invalid.")
        print("Error:", response.text)

if __name__ == "__main__":
    test_api_key()