

# Exploring the Visual Intelligence of OpenAI's GPT-4 Vision with Python in Google Colab


![](./images/0_RB-mznpMWzQ9-Mmb.jpg)

In the ever-evolving landscape of artificial intelligence, OpenAI's
GPT-4 Vision (GPT-4V) emerges as a groundbreaking innovation, blending
the advanced capabilities of GPT-4 with the ability to interpret and
analyze visual information. This integration marks a significant leap
forward, transcending the traditional text-only limitations of language
models. In this detailed exploration, we'll delve into the practical
applications of GPT-4V, showcasing how it can be used to unlock new
dimensions of image understanding in Google Colab.

# Introduction: A New Frontier in AI

GPT-4 with Vision, colloquially known as GPT-4V or gpt-4-vision-preview
in the API, represents a monumental step in AI's journey. This model
transcends the boundaries of traditional language models by
incorporating the ability to process and interpret images, thereby
broadening the scope of potential applications. GPT-4V is available
through the gpt-4-vision-preview model and the Chat Completions API,
which now supports image inputs, expanding the horizons of what AI can
achieve.

# Key Points about GPT-4V:

- Augmentative Capabilities: GPT-4V enhances the already robust GPT-4
    model by adding visual understanding without compromising its
    textual proficiency.
- Image Processing: The model can process images in two primary ways ---
    through direct URL links or by handling base64 encoded
    images.
-  Limitations and Use Cases: While GPT-4V excels at general image
    understanding and object relationships, it is not optimized for
    precise spatial localization tasks.

# Using GPT-4V: From URLs to Local Images

# Analyzing Images from URLs

One of the most straightforward methods to leverage GPT-4V is by using
image URLs. This approach is ideal for easily accessible images hosted
on platforms like GitHub or Wikimedia Commons. Here's how you can use
it:

```
!pip install openai
import os

# Prompt for the API key
api_key = input("Enter your OpenAI API key: ")

# Set the environment variable
os.environ["OPENAI_API_KEY"] = api_key
```

Then call the API:

```
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://raw.githubusercontent.com/fenago/datasets/main/IQR.png",
                    },
                },
            ],
        }
    ],
    max_tokens=300,
)

print(response.choices[0])
```

Replace `"https://your-image-url.jpg"` with your
image URL to get insights about the image.

Here is the image:

![](./images/1_DmHDqhYtq9vU1fazXx5jxQ.png)

Here is the analysis:

![](./images/1_5SMP6ZhAoHbsUSPm-5ScAw.png)


# Uploading and Analyzing Local Images

For local images, GPT-4V can analyze them using base64 encoding. This
method is particularly useful for personal or sensitive images that are
not hosted online:

```
import base64
import requests

# OpenAI API Key
# api_key = "YOUR_OPENAI_API_KEY"

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = "/content/drlee_dabi.jpg"

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "What’s in this image?"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{base64_image}"
            }
          }
        ]
      }
    ],
    "max_tokens": 300
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

print(response.json())
```

![](./images/1_0xGjy_YZgPRYfklOfoSO6A.png)

In this script, replace `"path_to_your_image.jpg"` with the path to your local image.

# Understanding the Limitations and Strengths

While GPT-4V opens new avenues in AI, it's crucial to recognize its
limitations. For instance, it's not suitable for interpreting
specialized medical images or understanding text in non-Latin alphabets.
Additionally, it may struggle with tasks requiring detailed spatial
reasoning or interpreting complex visual elements like graphs.

On the flip side, GPT-4V excels in general image understanding,
recognizing relationships between objects, and generating creative
responses based on visual inputs. Its ability to process multiple images
and understand their context collectively is particularly notable.

# Conclusion and Further Exploration

OpenAI's GPT-4 Vision model represents a significant stride in AI,
bridging the gap between visual and textual understanding. Whether
you're analyzing images from the web or local storage, GPT-4V offers a
versatile tool for a wide range of applications.

For more detailed information and advanced features, delve into OpenAI's
[official documentation](https://platform.openai.com/docs/guides/vision).

Embrace this new era of AI with OpenAI's GPT-4 Vision, where images and
text converge to create a richer, more comprehensive understanding of
the world around us.
