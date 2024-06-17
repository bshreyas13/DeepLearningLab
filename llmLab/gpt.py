import os
from dotenv import load_dotenv
import json
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage

# Function to print response and write to file
def process_response(response_dict, cnt):
    response_id = response_dict['id']
    model_used = response_dict['model']
    message_content = response_dict['choices'][0]['message']['content']
    total_tokens_used = response_dict['usage']['total_tokens']

    print(f"\n\tMessage Content:\n{message_content}")
    print(f"\tTotal Tokens Used: {total_tokens_used}")

    # Write to JSON file
    json_filename = 'prompt_responses.json'
    if os.path.exists(json_filename):
        with open(json_filename, 'r+') as json_file:
            existing_data = json.load(json_file)
            existing_data.append(response_dict)
            json_file.seek(0)
            json.dump(existing_data, json_file, indent=2)
    else:
        with open(json_filename, 'w') as json_file:
            json.dump([response_dict], json_file, indent=2)

    # Write to Markdown file
    with open('prompt_responses.md', 'a') as md_file:
        md_file.write(f"# Completion {cnt}\n")
        md_file.write(f"**Response ID:** {response_id}\n\n")
        md_file.write(f"**Model Used:** {model_used}\n\n")
        md_file.write(f"**Message Content:**\n\n{message_content}\n\n")
        md_file.write(f"**Total Tokens Used:** {total_tokens_used}\n\n")

if __name__ == "__main__":
    cnt = 0

    # Load environment variables from the .env file
    load_dotenv()

    # Get the OpenAI API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")

    # Check if the API key is loaded correctly
    if not api_key:
        raise ValueError("API key not found. Please check your .env file.")

    # Initialize LangChain OpenAI LLM with the API key
    llm = ChatOpenAI(api_key=api_key, model_name="gpt-3.5-turbo")

    # Create a prompt template
    prompt_template = PromptTemplate(
        input_variables=["prompt"],
        template="You are a software engineer who is skilled in programming in different languages and algorithms. Your task is to generate programs and text for the given prompt. {prompt}"
    )

    # Create a RunnableSequence by piping the prompt template to the LLM
    chain = prompt_template | llm

    # Load the JSON file
    with open('prompts.json', 'r') as file:
        data = json.load(file)

    # Iterate through each prompt in the JSON data
    for prompt in data['prompts']:
        print(f"\n\n**** Next question on the topic: {prompt['prompt']}")
        response = chain.invoke({"prompt": prompt['prompt']})

        # Convert the response to the expected dictionary format
        response_dict = {
            'id': 'N/A',  # AIMessage doesn't have an ID, so we set it as 'N/A'
            'model': "gpt-3.5-turbo",
            'choices': [{'message': {'content': response.content}}],
            'usage': {'total_tokens': 'N/A'}  # Adjust as necessary based on actual token usage
        }
        response_dict['prompt'] = prompt['prompt']
        process_response(response_dict, cnt)
        cnt += 1
