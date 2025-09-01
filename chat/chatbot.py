import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyC3F3krGi4b1fW9Ne8w32yUB4Ez4CBffkI") # paste key directly

model = genai.GenerativeModel("gemini-2.0-flash")
response = model.generate_content("Hello Gemini! Can you hear me?")
print(response.text)