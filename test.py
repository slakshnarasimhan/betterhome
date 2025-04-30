from ollama import Client
client = Client(host="http://localhost:11434")

response = client.embed(model="nomic-embed-text", input=["hello world"])
print(response)
