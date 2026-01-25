"""
API Configuration for Azure OpenAI endpoints.
Contains all API endpoint configurations for different model families (GPT-4o, GPT-4.1, GPT-5).
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class ApiConfig:
    """Configuration for a single Azure OpenAI API endpoint."""
    scope: str
    endpoint: str
    model: str
    api_version: str


# ========================== GPT-4.1 Configurations ==========================
API_CONFIGS_GPT41: List[ApiConfig] = [
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://readinnorthcentralus.openai.azure.com/",
        model="gpt-4.1-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://readinnorthcentralus.openai.azure.com/",
        model="gpt-4.1-DZS",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubnorthcentralus.openai.azure.com/",
        model="gpt-4.1-DZS",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubsouthcentralus.openai.azure.com/",
        model="gpt-4.1-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubsouthcentralus.openai.azure.com/",
        model="gpt-4.1-DZS",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubswedencentral.openai.azure.com/",
        model="gpt-4.1-DZS",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="api://trapi/.default",
        endpoint="https://trapi.research.microsoft.com/gcr/shared",
        model="gpt-4.1_2025-04-14",
        api_version="2025-04-01-preview",
    ),
]


# ========================== GPT-5 Configurations ==========================
API_CONFIGS_GPT5: List[ApiConfig] = [
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubeastus2.openai.azure.com/",
        model="gpt-5-DZS",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubeastus2.openai.azure.com/",
        model="gpt-5-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubswedencentral.openai.azure.com/",
        model="gpt-5-DZS",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubswedencentral.openai.azure.com/",
        model="gpt-5-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://readineastus2.openai.azure.com/",
        model="gpt-5-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://readineastus2.openai.azure.com/",
        model="gpt-5-DZS",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://readinswedencentral.openai.azure.com/",
        model="gpt-5-2025-08-07-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://readinswedencentral.openai.azure.com/",
        model="gpt-5-DZS",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="api://trapi/.default",
        endpoint="https://trapi.research.microsoft.com/gcr/shared",
        model="gpt-5_2025-08-07",
        api_version="2025-04-01-preview",
    ),
    # ApiConfig(
    #     scope="api://trapi/.default",
    #     endpoint="https://trapi.research.microsoft.com/gcr/shared",
    #     model="gpt-5-chat_2025-08-07",
    #     api_version="2025-04-01-preview",
    # ),
    # ApiConfig(
    #     scope="api://trapi/.default",
    #     endpoint="https://trapi.research.microsoft.com/gcr/shared",
    #     model="gpt-5-chat_2025-10-03",
    #     api_version="2025-04-01-preview",
    # ),
    # ApiConfig(
    #     scope="api://trapi/.default",
    #     endpoint="https://trapi.research.microsoft.com/gcr/shared",
    #     model="gpt-5.1_2025-11-13",
    #     api_version="2025-04-01-preview",
    # ),
    # ApiConfig(
    #     scope="api://trapi/.default",
    #     endpoint="https://trapi.research.microsoft.com/gcr/shared",
    #     model="gpt-5.1-chat_2025-11-13",
    #     api_version="2025-04-01-preview",
    # ),
    # ApiConfig(
    #     scope="api://trapi/.default",
    #     endpoint="https://trapi.research.microsoft.com/gcr/shared",
    #     model="gpt-5.2_2025-12-11",
    #     api_version="2025-04-01-preview",
    # ),
    # ApiConfig(
    #     scope="api://trapi/.default",
    #     endpoint="https://trapi.research.microsoft.com/gcr/shared",
    #     model="gpt-5.2-chat_2025-12-11",
    #     api_version="2025-04-01-preview",
    # ),
]


# ========================== GPT-4o Configurations ==========================
API_CONFIGS_GPT4O: List[ApiConfig] = [
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubeastus.openai.azure.com/",
        model="gpt-4o",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubeastus2.openai.azure.com/",
        model="gpt-4o",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubnorthcentralus.openai.azure.com/",
        model="gpt-4o",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubsouthcentralus.openai.azure.com/",
        model="gpt-4o",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://readineastus.openai.azure.com/",
        model="gpt-4o",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://readineastus2.openai.azure.com/",
        model="gpt-4o",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://readinwestus.openai.azure.com/",
        model="gpt-4o",
        api_version="2025-04-01-preview",
    ),
    # gpt-4o-global group
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubeastus.openai.azure.com/",
        model="gpt-4o-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubeastus2.openai.azure.com/",
        model="gpt-4o-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubnorthcentralus.openai.azure.com/",
        model="gpt-4o-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubsouthcentralus.openai.azure.com/",
        model="gpt-4o-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://conversationhubwestus.openai.azure.com/",
        model="gpt-4o-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://readineastus.openai.azure.com/",
        model="gpt-4o-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://readineastus2.openai.azure.com/",
        model="gpt-4o-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://readinnorthcentralus.openai.azure.com/",
        model="gpt-4o-global",
        api_version="2025-04-01-preview",
    ),
    ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://readinwestus.openai.azure.com/",
        model="gpt-4o-global",
        api_version="2025-04-01-preview",
    ),
]


# ========================== API Config Groups ==========================
API_CONFIG_GROUPS: Dict[str, List[ApiConfig]] = {
    "gpt5": API_CONFIGS_GPT5,
    "gpt4o": API_CONFIGS_GPT4O,
    "gpt41": API_CONFIGS_GPT41,
    "gpt5+gpt4o": API_CONFIGS_GPT5 + API_CONFIGS_GPT4O,
    "gpt5+gpt41": API_CONFIGS_GPT5 + API_CONFIGS_GPT41,
    "gpt4o+gpt41": API_CONFIGS_GPT4O + API_CONFIGS_GPT41,
    "gpt5+gpt4o+gpt41": API_CONFIGS_GPT5 + API_CONFIGS_GPT4O + API_CONFIGS_GPT41,
}


def get_api_configs(family: str) -> List[ApiConfig]:
    """
    Get API configurations by family name.
    
    Args:
        family: API family name (e.g., 'gpt5', 'gpt4o', 'gpt41', or combinations)
        
    Returns:
        List of ApiConfig objects
        
    Raises:
        ValueError: If family name is not recognized
    """
    if family not in API_CONFIG_GROUPS:
        raise ValueError(
            f"Unknown API family '{family}'. "
            f"Available families: {list(API_CONFIG_GROUPS.keys())}"
        )
    return API_CONFIG_GROUPS[family]


def get_default_config() -> ApiConfig:
    """Get a default API configuration for single test."""
    return ApiConfig(
        scope="https://cognitiveservices.azure.com/.default",
        endpoint="https://readinwestus.openai.azure.com/",
        model="gpt-4o-global",
        api_version="2025-04-01-preview",
    )
