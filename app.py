import chainlit as cl
import os
import groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq API client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = groq.Groq(api_key=GROQ_API_KEY)

@cl.on_chat_start
async def start_chat():
    cl.user_session.set(
        "interaction",
        [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            }
        ],
    )

    msg = cl.Message(content="")

    start_message = "Hello, I'm your AI assistant powered by Groq's LLaMA3. How can I assist you today?"

    for token in start_message:
        await msg.stream_token(token)

    await msg.send()

@cl.step(type="tool")
async def tool(input_message, image=None):

    interaction = cl.user_session.get("interaction")

    if image:
        interaction.append({"role": "user",
                            "content": input_message,
                            "images": image})
    else:
        interaction.append({"role": "user",
                            "content": input_message})
    
    # Send request to Groq's API
    response = client.chat.completions.create(
        model="llama3-8b-8192",  # Ensure this model is available in Groq
        messages=interaction
    ) 

    assistant_reply = response.choices[0].message.content

    interaction.append({"role": "assistant",
                        "content": assistant_reply})
    
    return assistant_reply


@cl.on_message 
async def main(message: cl.Message):

    images = [file for file in message.elements if "image" in file.mime]

    if images:
        tool_res = await tool(message.content, [i.path for i in images])

    else:
        tool_res = await tool(message.content)

    msg = cl.Message(content="")

    for token in tool_res:
        await msg.stream_token(token)

    await msg.send()
