import requests

url = "https://8000-dep-01jpjqmb6cpgmyk2f97rsm9djc-d.cloudspaces.litng.ai/v1/chat/completions"
s = requests.Session()
s.headers.update({"Authorization": "Bearer 51abcce7-5a95-4bcb-b288-046b83f575ef"})
response = s.post(url, json={
  "model": "",
  "messages": [
    {
      "role": "user",
      "content": "You are a helpful assistant who provides short and concise answer. How many 'R's in strawbery?"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 500
})
print(response.content)
