import os
import base64
from mimetypes import guess_type
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.identity import ChainedTokenCredential, AzureCliCredential, ManagedIdentityCredential, get_bearer_token_provider
# from dotenv import load_dotenv
from openai import AzureOpenAI



# import tiktoken

# def count_tokens(text: str, model: str = "gpt-4o") -> int:
#     enc = tiktoken.encoding_for_model(model)
#     return len(enc.encode(text))

if __name__ == "__main__":
    
    # 组里的API
    scope = "https://cognitiveservices.azure.com/.default"
    endpoint = "https://readinwestus.openai.azure.com/"
    model = 'gpt-4o-global'
    api_version="2024-04-01-preview"
    api_version="2025-04-01-preview"
    
    
    # scope = "https://cognitiveservices.azure.com/.default"
    # endpoint = "https://conversationhubeastus2.openai.azure.com/"
    # model = 'gpt-5-DZS'
    # model = 'gpt-5-global'
    # api_version="2025-04-01-preview"
    
    
    scope = "https://cognitiveservices.azure.com/.default"
    endpoint = "https://conversationhubswedencentral.openai.azure.com/"
    # endpoint = "https://readineastus2.openai.azure.com/"
    model = 'gpt-5-DZS'
    # model = 'gpt-5-global'
    api_version="2025-04-01-preview"
    

    scope = "https://cognitiveservices.azure.com/.default"
    endpoint = "https://conversationhubeastus2.openai.azure.com/"
    model = 'gpt-5-chat'
    api_version="2025-04-01-preview"
    
    # gcr-shared
    scope = "api://trapi/.default"
    endpoint = "https://trapi.research.microsoft.com/gcr/shared"
    # endpoint = "https://trapi.research.microsoft.com/msra/shared"
    # model = 'gpt-4o_2024-11-20'
    model = 'gpt-5_2025-08-07'
    model = 'gpt-4.1_2025-04-14'
    model = 'gpt-5.2_2025-12-11'
    model = 'gpt-5-chat_2025-10-03'
    api_version="2025-04-01-preview"
    
    # model = 'gpt-5-pro_2025-10-06' # unavailablecd 
    # model = 'gpt-5.1_2025-11-13' # unavailable
    # api_version="2025-10-01-preview"

    token_provider = get_bearer_token_provider(ChainedTokenCredential(
        AzureCliCredential(),
        ManagedIdentityCredential(),
    ), scope)
    # token_provider = get_bearer_token_provider(
    #                 DefaultAzureCredential(managed_identity_client_id="18731b9d-0488-49be-bb05-6ccb08f78cf3"),
    #                 scope)
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_version,
        max_retries=0,
    )
    # response = client.chat.completions.create(
    #     model=model,
    #     messages=[
    #         {"role": "system", "content": "You are a helpful assistant."},
    #         {"role": "user", "content": "Hello, how are you?"}
    #     ]
    # )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ],
        # reasoning_effort="medium" # for gpt-5 series        
    )
    print(response.choices[0].message.content)
    print("Prompt tokens:", response.usage.prompt_tokens)
    print("Completion tokens:", response.usage.completion_tokens)
    print("Total tokens:", response.usage.total_tokens)

    
    
    # from pydantic import BaseModel
    
    # class CalendarEvent(BaseModel):
    #     name: str
    #     date: str
    #     participants: list[str]

    # completion = client.beta.chat.completions.parse(
    #     model=model, # replace with the model deployment name of your gpt-4o 2024-08-06 deployment
    #     messages=[
    #         {"role": "system", "content": "Extract the event information."},
    #         {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
    #     ],
    #     response_format=CalendarEvent,
    # )

    # event = completion.choices[0].message.parsed

    # print(event)
    # print(completion.model_dump_json(indent=2))
    # print(completion.choices[0].message.content)
    # print("Prompt tokens:", completion.usage.prompt_tokens)
    # print("Completion tokens:", completion.usage.completion_tokens)
    # print("Total tokens:", completion.usage.total_tokens)
    # print("Counted tokens:", count_tokens(completion.choices[0].message.content, model='gpt-4o'))
    