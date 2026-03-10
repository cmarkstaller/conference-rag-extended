import requests

url = "https://www.churchofjesuschrist.org/study/general-conference/2024/04?lang=eng"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
}

response = requests.get(url, headers=headers, timeout=10)
response.raise_for_status()

print("Status Code:", response.status_code)
print("HTML Length:", len(response.text))
print("HTML:", response.text)

# response.text is the page body