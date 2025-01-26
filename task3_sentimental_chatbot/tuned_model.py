import google.generativeai as genai
import pandas as pd
import seaborn as sns
import time, os
from dotenv import load_dotenv

load_dotenv()

sentiments = pd.read_csv("../task3_sentimental_chatbot/dataset/chat_dataset.csv")
BASE_MODEL = "models/gemini-1.5-flash-001-tuning"
MODEL_DISPLAY_NAME = "sentimental-bot"

training_data = []

for index, row in sentiments.iterrows():
    training_data.append({"text_input": row.message, "output": row.sentiment})

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

operation = genai.create_tuned_model(
    display_name=MODEL_DISPLAY_NAME,
    source_model=BASE_MODEL,
    epoch_count=20,
    batch_size=5,
    learning_rate=0.001,
    training_data=training_data,
)

for status in operation.wait_bar():
    time.sleep(10)

result = operation.result()
print(result)

# Plot loss function curve
snapshots = pd.DataFrame(result.tuning_task.snapshots)
sns.lineplot(data=snapshots, x="epoch", y="mean_loss")

# Update the environment variable for model path
os.environ.update({"SEMTIMENTIAL_MODEL": result.name})

model = genai.GenerativeModel(model_name=result.name)

result = model.generate_content("I dont like this product")
print(result.text)

# Success Message
print("\n\nModel Successfully fine tuned")

genai.update_tuned_model(
    os.environ.get("SEMTIMENTIAL_MODEL"),
    {
        "description": "Classifies Sentiments based on the the input into [positive, neutral, negative] categories"
    },
)

# List all the models fine-tuned
for model_info in genai.list_tuned_models():
    print(
        f"\n\
        Model: {model_info.name}\n\
        Description: {model_info.description}"
    )
