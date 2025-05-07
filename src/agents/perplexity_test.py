from openai import OpenAI
import dotenv 
import os

dotenv.load_dotenv()

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

messages = [
    {
        "role": "system",
        "content": (
            "You are an artificial intelligence assistant and you need to "
            "search the web and summarize the results. "
        ),
    },
    {   
        "role": "user",
        "content": (
            "Quais as últimas nottícias sobre o Paraná Clube? "
        ),
    },
]

client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")

# chat completion without streaming
response = client.chat.completions.create(
    model="sonar-pro",
    messages=messages,
)
print(response)

# chat completion with streaming
response_stream = client.chat.completions.create(
    model="sonar-pro",
    messages=messages,
    stream=True,
)

for response in response_stream:
    print(response)
