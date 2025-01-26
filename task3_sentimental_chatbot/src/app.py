import os
from PIL import Image
import streamlit as st
import pollinations as ai
from random import randint
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
import google.api_core.exceptions as api_exc

# Custom CSS for the Streamlit app
st.markdown(
    """
<style>
    .st-emotion-cache-janbn0 {
        flex-direction: row-reverse;
        text-align: right;
    }

    .st-emotion-cache-1dnm2d2 .es2srfl5 {
        display: none;
    }

    [data-testid='stFileUploader'] {
        display: none 
        width: min-content;
    }
    [data-testid='stFileUploader'] section {
        padding: 0;
        float: left;
    }
    [data-testid='stFileUploader'] section > input + div {
        display: none;
    }
    [data-testid='stFileUploader'] section + div {
        float: right;
        padding-top: 0;
    }

</style>
""",
    unsafe_allow_html=True,
)

# Config Model
gemini_config = genai.GenerationConfig(
    # max_output_tokens=1000,
    # temperature=0.1,
)

# Load environment variables
load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# Initialize the models
model = genai.GenerativeModel(
    "gemini-1.5-flash",
    generation_config=gemini_config,
)

prompt_image_model = genai.GenerativeModel(
    "gemini-1.5-flash",
)

image_model = ai.Image(
    model="flux",
    width=1024,
    height=1024,
    seed=randint(1, 1000000000),
)

try: 
    sentimental_model = genai.GenerativeModel(
        model_name=os.environ.get("SEMTIMENTIAL_MODEL")
    )
except Exception as e:
    sentimental_model = None
    print(f"log: Error while loading sentimental model: {e}")

sentimental_context_model = genai.GenerativeModel(
    "gemini-1.5-flash",
)
print("log: Models initialized")


# Initialize Streamlit App
st.title("Multi Modal Chatbot /w _sentiments_")

# Initialize chat history in session state
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])
    st.session_state.sentiment_context = sentimental_context_model.start_chat(history=[])
    st.session_state.messages = []
    st.session_state.generated_image = (
        []
    )  # [{"image_prompt": "prompt to generate an image", "image_path": "generated image path"}]
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 1
print("log: Streamlit app initialized")


# Function to get the color for a given sentiment
def sentiment_color(sentiment: str) -> str:
    """
    Returns the color for a given sentiment.

    Args:
        sentiment (str): The sentiment to get the color for.

    Returns:
        str: The HTML color for the given sentiment.
    """

    # Define a dictionary mapping sentiments to colors
    sentiment_colors = {
        "positive": "green",
        "negative": "red",
        "neutral": "blue",
    }
    return sentiment_colors.get(sentiment.lower())


# Function to save uploaded image
def save_uploaded_image(uploaded_file: any) -> str:
    """
    Saves the uploaded image file with a timestamped filename.

    Args:
        uploaded_file: An uploaded file object, typically from a file upload interface.

    Returns:
        str: The name of the saved image file, formatted with a timestamp.
    """

    if uploaded_file is not None:

        # Get the current timestamp
        current_time = datetime.now()

        # Format the timestamp to match the desired format
        timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S-%f")

        uploaded_image = Image.open(uploaded_file)
        image_name = f"image-input-{timestamp}.jpg"
        uploaded_image.save(image_name)

        print(f"log: Image saved successfully to {image_name}")

        return image_name


# Display previous messages
for message in st.session_state.messages:
    # Displays a chat messages based on its roles stored in streamlit session state.

    with st.chat_message(message["role"]):
        if message["parts"]:
            if message["role"] == "user":
                st.markdown(
                    f"""
                    {message["parts"]}  
                    :{sentiment_color(str(message["sentiment"]))}[{str(message["sentiment"]).upper()}]
                    """
                )
            else:
                st.markdown(message["parts"])
        if message["image"]:
            image_source = (
                "Uploaded Image" if message["role"] == "user" else "Generated Image"
            )
            st.image(message["image"], caption=image_source, use_container_width=True)


# User Text Input
if prompt := st.chat_input("Talk to the Bot. Type your message here..."):
    # Processes the user's text input and generates a response.

    is_image_gen = False
    is_similar_image_gen = False
    try:
        user_sentiment = sentimental_model.generate_content(prompt).text
    except Exception as e:
        print("\nWARNING: Sentimental model not loaded. Using 'neutral' sentiment as default which will be corrected by sentiment context model verification.\n")
        user_sentiment = 'neutral'

    # Check if the prompt is requesting an image
    if "/generate " in prompt:
        # Check if the prompt is requesting an image

        print("log: Prompt is requesting an image")
        is_image_gen = True
        prompt = prompt.replace("/generate", "generate")
        image_gen_prompt = prompt
        print(f"log: Prompt for image generation is '{image_gen_prompt}'")

    # Add user message to session state
    user_message = {
        "role": "user",
        "parts": prompt,
        "image": None,
        "sentiment": user_sentiment,
    }
    print("log: user_message generated")

    # Check if the prompt is requesting a similar image
    similar_image_gen_response = prompt_image_model.generate_content(
        f"You are provided with a prompt by the user. Determine whether the prompt is specifically requesting a similar image, a variation of a previously generated image, or an image with specific twists or modifications to a previously generated image.\
        If the prompt is asking for a new image generation, describing an image, or purely textual in nature, respond with 'False'.\
        Only respond with 'True' or 'False' and avoid using any other words.\n\
        Prompt you need to analyze: {prompt}\n\
        Response must be in the following format: True/False"
    )

    if "True" in similar_image_gen_response.text:
        # Check if the prompt is requesting a similar image

        print("log: Prompt is requesting a similar image")
        is_similar_image_gen = True
        try:
            image_gen_prompt = prompt_image_model.generate_content(
                [
                    f"Create a new prompt that generates a similar image to the one described in: '{st.session_state.generated_image[-1]['image_prompt']}'.\
                    The new prompt should take inspiration from the provided image to introduce variations or a different version of it.\
                    The new prompt should not be the same as the original prompt.\
                    The new prompt must be relatively simple.\
                    There must be only one new prompt no need for options.\
                    The new prompt must be in a format which can be directly feeded to the image model to generate an image.\
                    The new prompt must not have any textual formatting like markdown or html just plain text.",
                    Image.open(st.session_state.generated_image[-1]["image_path"]),
                ]
            ).text
            print(
                f"log: Prompt for similar image generated successfully as '{image_gen_prompt}'"
            )
        except IndexError as e:
            print("log: No previous image found to generate similar image")
        except Exception as e:
            print("log: Error occurred while generating prompt for similar image:", e)

    user_sentiment_re = None

    if st.session_state.uploaded_image:
        # Store the uploaded image in user_message if it exists

        user_message["image"] = Image.open(st.session_state.uploaded_image)
        print("log: 'image' added in user_message")

        user_sentiment_re = st.session_state.sentiment_context.send_message(
            [
                f"You are chatbot only designed to classify user sentiments and confirm the sentiments generated by a different model. \
            The user has given the following prompt/ message: {user_message['parts']} \n\
            The sentiment given by the other model is '{user_message['sentiment']}' \n\
            If the sentiment given by previous model is 'neutral' give extra attentation as its more likely to be incorrect. \
            Consider the previous conversations ie the history carefully and reclassify the sentiment if necessary. \
            An image may be provided also take that into consideration. \n\
            Only respond with one of the following: 'positive', 'neutral', or 'negative'. \
            DO NOT give any explanation or reasoning just a sentiment as the output. \
            Output: positive/neutral/negative",
                user_message["image"],
            ]
        )
    else:
        user_sentiment_re = st.session_state.sentiment_context.send_message(
            [
                f"You are chatbot only designed to classify user sentiments and confirm the sentiments generated by a different model. \
            The user has given the following prompt/ message: {user_message['parts']} \n\
            The sentiment given by the other model is '{user_message['sentiment']}' \n\
            If the sentiment given by previous model is 'neutral' give extra attentation as its more likely to be incorrect. \
            Consider the previous conversations carefully and reclassify the sentiment if necessary. \
            An image may be provided also take that into consideration. \n\
            Only respond with one of the following: 'positive', 'neutral', or 'negative'. \
            DO NOT give any explanation or reasoning just a sentiment as the output. \
            Output: positive/neutral/negative",
            ]
        )

    if user_sentiment_re is not None:
        user_message["sentiment"] = user_sentiment_re.text.strip().splitlines()[0]

    print(f"log: user sentiment generated as {user_message['sentiment']}")

    # Add user message to session state
    st.session_state.messages.append(user_message)
    print("log: user_message stored in session state")

    # Display user message
    with st.chat_message("user"):
        st.markdown(
            f"""
        {prompt}  
        :{sentiment_color(str(user_message['sentiment']))}[{str(user_message["sentiment"]).upper()}]
        """
        )

        if st.session_state.uploaded_image:
            st.image(
                user_message["image"],
                caption="Uploaded Image",
                use_container_width=True,
            )
        print("log: user_message displayed")

    # Display AI response
    with st.chat_message("ai"):
        message_placeholder = st.empty()
        full_response = ""
        full_image = None

        with st.spinner("Thinking..."):
            try:
                if not is_image_gen and not is_similar_image_gen:
                    # If the prompt is not requesting an image, generate a text response

                    # Construct sentimental prompt
                    sentimental_prompt = f"""
                    You are a sentimental chatbot that generates responses to user prompts based on the user's sentiment.\n\
                    The user's sentiment is: {user_sentiment.upper()}.\n\
                    The user's prompt is: {prompt}.\n\
                    Generate a response to the user's prompt based on the user's sentiment.\
                    But also consider the previous conversations between the user and the chatbot.\
                    Make sure respond to the overall user sentiment from the entire chat history while also give importance to the currnet user sentiment if they are [POSITIVE or NEGATIVE] user sentiment [NEUTRAL] can be given less importance and the overall sentiment can be given more importance in that case.
                    """

                    # Generate AI response
                    if prompt and st.session_state.uploaded_image:
                        # If image is uploaded, add it to the prompt and generate a response

                        response = st.session_state.chat.send_message(
                            [prompt, user_message["image"]], stream=True
                        )
                        print("log: response generated from text and image prompt")

                    else:
                        # If no image is uploaded, generate a response

                        response = st.session_state.chat.send_message(
                            sentimental_prompt, stream=True
                        )
                        print("log: response generated from text prompt")

                    # Stream Text Response
                    for chunk in response:
                        full_response += chunk.text or ""
                        message_placeholder.markdown(full_response + "▌")

                    message_placeholder.markdown(full_response)
                    print("log: text response displayed")

                else:
                    # If the prompt is requesting an image, generate an image

                    # Generate AI Image
                    image_model.__call__(
                        prompt=image_gen_prompt,
                    )

                    timestamp = image_model.timestamp.strftime("%Y-%m-%d-%H-%M-%S-%f")

                    # Save the generated image
                    image_name = f"image-output-{timestamp}.jpg"
                    image_model.save(image_name)
                    print(f"log: image generated and saved as '{image_name}'")
                    full_image = Image.open(image_name)

                    # Inform the text model about the generated image
                    st.session_state.chat.send_message(
                        [
                            f"You have generated an image based on the following prompt: '{image_gen_prompt}'.\
                            The generated image is now available for reference. If asked, you should describe or provide details about this image.",
                            full_image,
                        ]
                    )
                    print("log: Informing the text model about the generated image")

                    # Store generated image and prompt
                    st.session_state.generated_image.append(
                        {
                            "image_prompt": image_gen_prompt,
                            "image_path": image_name,
                        }
                    )
                    print("log: Generated image and prompt stored in session state")

                    # Display Image Response
                    message_placeholder.image(full_image, caption="Generated Image")
                    print("log: image response displayed")

                # Add AI response to session state
                ai_message = {"role": "ai", "parts": full_response, "image": full_image}
                st.session_state.messages.append(ai_message)
                print("log: ai_message generated and added in session state")

            except api_exc.GoogleAPIError as e:
                st.error(f"API Error: {e.message}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

        print("log: ai_message displayed")

    # Update uploader key
    st.session_state.uploader_key += 1
    st.session_state.uploaded_image = None
    print("log: Uploader keys updated")

# Suggest the user to generate a new image
st.markdown(
    "**Tip:** Type `/generate {prompt}` to request a new image or a variation of the previous one."
)

# File uploader at the bottom
uploaded_file = st.file_uploader(
    "📂Upload an image/ ⬇️Drop a File",
    type=["jpg", "jpeg", "png"],
    key=st.session_state["uploader_key"],
)

if uploaded_file:
    # If an image is uploaded:
    # 1. Store the image in the streamlit session state to be used to display it or generate a response
    # 2. Generate a prompt to generate the same image which will be used to generate the image if requested

    st.session_state.uploaded_image = uploaded_file
    print("log: Image uploaded and stored in session state")

    # Display the uploaded image
    with st.chat_message("user"):

        # Generate a prompt to generate the same image
        image_gen_prompt = prompt_image_model.generate_content(
            [
                "Generate a new prompt to create the exact same image as the one described in the provided image.\
            Use the provided image to replicate it exactly.\
            You must genererate only one detailed prompt.\
            Avoid generating multiple prompts.\
            The prompt must be in a format which can be directly feeded to the image model to generate an image.\
            The prompt must not have any textual formatting like markdown or html just plain text.",
                Image.open(uploaded_file),
            ]
        ).text
        print(f"log: Image generation prompt generated as '{image_gen_prompt}'")

        # Store generated image and prompt
        image_name = save_uploaded_image(st.session_state.uploaded_image)
        print(f"log: Image saved as '{image_name}'")

        st.session_state.generated_image.append(
            {
                "image_prompt": image_gen_prompt,
                "image_path": image_name,
            }
        )
        print(f"log: Image saved in session state")

        # Display the uploaded image
        st.image(
            Image.open(uploaded_file),
            caption="Uploaded Image",
            use_container_width=True,
        )
        print("log: Image displayed")
