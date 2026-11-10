from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:18005/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="GLM-5.3-Flash-Mix",
    messages=[{"role": "user", "content": "请说明AI加速芯片的软件栈的作用。"}],
    temperature=0,
    max_tokens=1024,
)
print(response.choices[0].message.content)