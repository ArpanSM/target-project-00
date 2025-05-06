import os
import json
import litellm
from litellm import Router, completion, completion_cost
from pydantic import BaseModel, Field
from typing import Optional

# load environment variables
from dotenv import load_dotenv
load_dotenv()

router = Router(
    routing_strategy="usage-based-routing-v2",
    enable_pre_call_checks=True,
    model_list=[
        {
            "model_name": "gemini/gemini-2.5-flash-preview-04-17",
            "litellm_params":
            {
                "model": "gemini/gemini-2.5-flash-preview-04-17",
                "api_key": os.getenv("GEMINI_API_KEY_1")
            }
        },
        {
            "model_name": "gemini/gemini-2.5-flash-preview-04-17",
            "litellm_params":
            {
                "model": "gemini/gemini-2.5-flash-preview-04-17",
                "api_key": os.getenv("GEMINI_API_KEY_2")
            }
        },
        {
            "model_name": "gemini/gemini-2.0-flash",
            "litellm_params":
            {
                "model": "gemini/gemini-2.0-flash",
                "api_key": os.getenv("GEMINI_API_KEY_1")
            }
        },
        {
            "model_name": "gemini/gemini-2.0-flash",
            "litellm_params":
            {
                "model": "gemini/gemini-2.0-flash",
                "api_key": os.getenv("GEMINI_API_KEY_2")
            }
        },
        {
            "model_name": "gemini/ gemini-2.5-pro-exp-03-25",
            "litellm_params":
            {
                "model": "gemini/ gemini-2.5-pro-exp-03-25",
                "api_key": os.getenv("GEMINI_API_KEY_1")
            }
        },
        {
            "model_name": "gemini/gemini-2.5-pro-exp-03-25",
            "litellm_params":
            {
                "model": "gemini/gemini-2.5-pro-exp-03-25",
                "api_key": os.getenv("GEMINI_API_KEY_2")
            }
        },
        {
            "model_name": "gemini/gemini-2.0-flash-lite",
            "litellm_params":
            {
                "model": "gemini/gemini-2.0-flash-lite",
                "api_key": os.getenv("GEMINI_API_KEY_1")
            }
        },
        {
            "model_name": "gemini/gemini-2.0-flash-lite",
            "litellm_params":
            {
                "model": "gemini/gemini-2.0-flash-lite",
                "api_key": os.getenv("GEMINI_API_KEY_2")
            }
        },
        {
            "model_name": "groq/llama-3.3-70b-versatile",
            "litellm_params":
            {
                "model": "groq/llama-3.3-70b-versatile",
                "api_key": os.getenv("GROQ_API_KEY_1")
            }
        },
        {
            "model_name": "groq/qwen-qwq-32b",
            "litellm_params":
            {
                "model": "groq/qwen-qwq-32b",
                "api_key": os.getenv("GROQ_API_KEY_1")
            }
        },
        {
            "model_name": "openrouter/deepseek/deepseek-chat-v3-0324:free",
            "litellm_params":
            {
                "model": "openrouter/deepseek/deepseek-chat-v3-0324",
                "api_key": os.getenv("OPEN_ROUTER_API_KEY_1")
            }
        },
        {
            "model_name": "openrouter/deepseek/deepseek-r1:free",
            "litellm_params":
            {
                "model": "openrouter/deepseek/deepseek-r1",
                "api_key": os.getenv("OPEN_ROUTER_API_KEY_1")
            }
        }
    ]
)

def call_litellm(**kwargs):
    response = router.completion(**kwargs)
    cost = completion_cost(response)
    print(f"{kwargs.get('model')} Cost: {cost}")
    output = response.choices[0].message.content
    if kwargs.get('response_format'):
        output = json.loads(output)
    return response, output