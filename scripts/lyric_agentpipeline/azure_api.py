"""
Azure OpenAI API Client with Ray-based concurrency.
Provides both single-request and parallel batch processing capabilities.
Supports custom task processing with input preparation and output validation.
"""

import json
import random
import time
import datetime
import builtins
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import yaml
from pydantic import BaseModel
from azure.identity import (
    AzureCliCredential,
    ChainedTokenCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from openai import AzureOpenAI

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from API_config import ApiConfig, get_api_configs, get_default_config


# ========================== Global Settings ==========================
API_EVENT_LOOKBACK_SECONDS = 600
API_MINIMAL_RATIO = 0.01
CLIENT_REFRESH_SECONDS = 3600
DEBUG_MODE = False


def get_beijing_time() -> str:
    """Get current Beijing time as formatted string."""
    bj_time = (
        datetime.datetime.utcnow()
        .replace(tzinfo=datetime.timezone.utc)
        .astimezone(datetime.timezone(datetime.timedelta(hours=8)))
        .strftime("%Y-%m-%d %H:%M:%S")
    )
    return bj_time


# Override print function to include timestamp
original_print = builtins.print


def custom_print(*args, **kwargs):
    original_print(get_beijing_time(), *args, **kwargs)


builtins.print = custom_print


def debug_print(*args, **kwargs):
    """Print only when DEBUG_MODE is enabled."""
    if DEBUG_MODE:
        print("[DEBUG]", *args, **kwargs)


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in text using tiktoken."""
    if not TIKTOKEN_AVAILABLE:
        # Approximate token count if tiktoken not available
        return len(text) // 4
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


# ========================== Client Building ==========================
def build_openai_client(api_config: ApiConfig) -> AzureOpenAI:
    """
    Build an Azure OpenAI client from the given configuration.
    
    Args:
        api_config: API configuration with endpoint, model, etc.
        
    Returns:
        Configured AzureOpenAI client
    """
    print(
        f"Initializing client for endpoint {api_config.endpoint} "
        f"model {api_config.model} (api_version={api_config.api_version})"
    )
    credential = ChainedTokenCredential(
        AzureCliCredential(),
        ManagedIdentityCredential(),
    )
    token_provider = get_bearer_token_provider(credential, api_config.scope)
    client = AzureOpenAI(
        azure_endpoint=api_config.endpoint,
        azure_ad_token_provider=token_provider,
        api_version=api_config.api_version,
        max_retries=0,
    )
    return client


# ========================== API Event Tracking ==========================
class APIEventTracker:
    """Track API call success/failure events for load balancing."""
    
    def __init__(self):
        self.events: Dict[Tuple[str, str, str], List[Tuple[float, bool]]] = {}
    
    def _api_key(self, api_config: ApiConfig) -> Tuple[str, str, str]:
        return (api_config.scope, api_config.endpoint, api_config.model)
    
    def _prune_events(self, events: List[Tuple[float, bool]]) -> List[Tuple[float, bool]]:
        cutoff = time.time() - API_EVENT_LOOKBACK_SECONDS
        return [event for event in events if event[0] >= cutoff]
    
    def record_event(self, api_config: ApiConfig, success: bool) -> None:
        """Record an API call event (success or failure)."""
        key = self._api_key(api_config)
        events = self.events.get(key, [])
        events.append((time.time(), success))
        events = self._prune_events(events)
        self.events[key] = events
    
    def compute_weight(self, api_config: ApiConfig) -> float:
        """Compute selection weight based on recent success rate."""
        key = self._api_key(api_config)
        events = self._prune_events(self.events.get(key, []))
        self.events[key] = events
        
        if not events:
            return 1.0
        
        successes = sum(1 for _, ok in events if ok)
        total = len(events)
        ratio = successes / total if total > 0 else 0.0
        return ratio + API_MINIMAL_RATIO


# Global event tracker
_event_tracker = APIEventTracker()


# ========================== Client Management ==========================
class ClientManager:
    """Manage multiple Azure OpenAI clients with load balancing."""
    
    def __init__(self, api_configs: List[ApiConfig]):
        self.api_configs = api_configs
        self.clients: List[Tuple[ApiConfig, AzureOpenAI]] = []
        self.last_refresh: float = 0.0
        self._initialize_clients()
    
    def _initialize_clients(self) -> None:
        """Initialize all clients from configs at startup."""
        self.clients = []
        print(f"Initializing {len(self.api_configs)} API clients...")
        
        for cfg in self.api_configs:
            try:
                client = build_openai_client(cfg)
                self.clients.append((cfg, client))
            except Exception as exc:
                print(f"Failed to initialize client for {cfg.endpoint} ({cfg.model}): {exc}")
        
        if not self.clients:
            raise RuntimeError("Unable to initialize any Azure OpenAI clients.")
        
        self.last_refresh = time.time()
        print(f"✅ Successfully initialized {len(self.clients)} API clients.")
    
    def refresh_if_needed(self) -> None:
        """Refresh clients if they're stale."""
        if time.time() - self.last_refresh > CLIENT_REFRESH_SECONDS:
            print("Refreshing stale clients...")
            self._initialize_clients()
    
    def get_all_clients(self) -> List[Tuple[ApiConfig, AzureOpenAI]]:
        """Get all available clients."""
        self.refresh_if_needed()
        return self.clients.copy()
    
    def select_client(self, exclude_configs: Optional[List[ApiConfig]] = None) -> Optional[Tuple[ApiConfig, AzureOpenAI]]:
        """
        Select a client using weighted random selection based on success rates.
        
        Args:
            exclude_configs: List of ApiConfigs to exclude from selection (e.g., failed ones)
            
        Returns:
            Tuple of (ApiConfig, AzureOpenAI) or None if no clients available
        """
        self.refresh_if_needed()
        
        if not self.clients:
            return None
        
        # Filter out excluded configs
        available_clients = self.clients
        if exclude_configs:
            exclude_set = {(cfg.endpoint, cfg.model) for cfg in exclude_configs}
            available_clients = [
                (cfg, client) for cfg, client in self.clients 
                if (cfg.endpoint, cfg.model) not in exclude_set
            ]
        
        if not available_clients:
            return None
        
        # Compute weights based on success rates
        weights = []
        for api_config, _ in available_clients:
            weight = max(_event_tracker.compute_weight(api_config), API_MINIMAL_RATIO)
            weights.append(weight)
        
        total = sum(weights)
        if total <= 0:
            return random.choice(available_clients)
        
        # Weighted random selection
        r = random.random() * total
        cumulative = 0.0
        for (api_config, client), weight in zip(available_clients, weights):
            cumulative += weight
            if r <= cumulative:
                return api_config, client
        
        return available_clients[-1]


# ========================== Single Request Functions ==========================
def get_chat_response(
    client: AzureOpenAI,
    model: str,
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
) -> Tuple[str, Dict[str, int]]:
    """
    Get a simple chat completion response.
    
    Args:
        client: Azure OpenAI client
        model: Model name to use
        prompt: User prompt
        system_prompt: System prompt
        
    Returns:
        Tuple of (response_text, usage_dict)
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    
    return response.choices[0].message.content or "", usage


def get_structured_response(
    client: AzureOpenAI,
    model: str,
    prompt: str,
    response_format: Type[BaseModel],
    system_prompt: str = "You are a helpful assistant.",
) -> Tuple[str, Dict[str, int]]:
    """
    Get a structured response using Pydantic model.
    
    Args:
        client: Azure OpenAI client
        model: Model name to use
        prompt: User prompt
        response_format: Pydantic model class for response parsing
        system_prompt: System prompt
        
    Returns:
        Tuple of (response_text, usage_dict)
    """
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        response_format=response_format,
    )
    
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    
    return response.choices[0].message.content or "", usage


def call_with_retry(
    client_manager: ClientManager,
    prompt: str,
    max_retries: int = 5,
    response_format: Optional[Type[BaseModel]] = None,
    system_prompt: str = "You are a helpful assistant.",
    task_id: str = "",
) -> Optional[Tuple[str, Dict[str, int]]]:
    """
    Call API with automatic retry and load balancing.
    
    Strategy:
    1. Randomly select a client from available pool
    2. If request fails, exclude that client and select another
    3. Continue until success or all clients exhausted or max_retries reached
    
    Args:
        client_manager: ClientManager instance
        prompt: User prompt
        max_retries: Maximum retry attempts
        response_format: Optional Pydantic model for structured output
        system_prompt: System prompt
        task_id: Optional task identifier for logging
        
    Returns:
        Tuple of (response_text, usage_dict) or None if all retries fail
    """
    failed_configs: List[ApiConfig] = []  # Track failed clients to exclude
    attempt = 0
    
    while attempt <= max_retries:
        # Select client excluding previously failed ones
        selection = client_manager.select_client(exclude_configs=failed_configs)
        
        if selection is None:
            # All clients have been tried and failed, reset and retry with all
            if failed_configs:
                debug_print(f"All {len(failed_configs)} clients failed, resetting exclusion list...")
                failed_configs = []
                selection = client_manager.select_client()
            
            if selection is None:
                print(f"❌ No clients available. {task_id}")
                return None
        
        api_config, client = selection
        
        try:
            if response_format:
                result = get_structured_response(
                    client, api_config.model, prompt, response_format, system_prompt
                )
            else:
                result = get_chat_response(
                    client, api_config.model, prompt, system_prompt
                )
            
            debug_print(
                f"✅ {api_config.endpoint} {api_config.model} Success on attempt #{attempt+1}. {task_id}"
            )
            _event_tracker.record_event(api_config, True)
            return result
            
        except Exception as exc:
            attempt += 1
            _event_tracker.record_event(api_config, False)
            
            # Add to failed list so we try a different client next time
            failed_configs.append(api_config)
            
            if attempt > max_retries:
                print(f"❌ Max retries ({max_retries}) reached ({exc}), giving up. {task_id}")
                return None
            
            # Shorter sleep since we're switching to a different client
            remaining_clients = len(client_manager.clients) - len(failed_configs)
            if remaining_clients > 0:
                # Quick switch to another client
                sleep_seconds = random.randint(1, 5)
                print(
                    f"⚠️ {api_config.endpoint} {api_config.model} "
                    f"Error ({type(exc).__name__}): {exc}. "
                    f"Switching to another client ({remaining_clients} remaining)... {task_id}"
                )
            else:
                # All clients tried, longer sleep before reset
                sleep_seconds = random.randint(30, 60)
                print(
                    f"⚠️ All clients failed. "
                    f"Sleeping {sleep_seconds}s before retry #{attempt}... {task_id}"
                )
            
            time.sleep(sleep_seconds)
    
    return None


# ========================== Single Test Function ==========================
def single_test(
    prompt: str = "Hello, how are you?",
    api_config: Optional[ApiConfig] = None,
    system_prompt: str = "You are a helpful assistant.",
) -> None:
    """
    Single test function for API connectivity (no concurrency).
    
    Args:
        prompt: Test prompt to send
        api_config: Optional specific API config to use (uses default if None)
        system_prompt: System prompt
    """
    if api_config is None:
        api_config = get_default_config()
    
    print(f"Testing API: {api_config.endpoint} / {api_config.model}")
    print(f"Prompt: {prompt}")
    print("-" * 50)
    
    try:
        client = build_openai_client(api_config)
        response_text, usage = get_chat_response(
            client, api_config.model, prompt, system_prompt
        )
        
        print(f"Response: {response_text}")
        print(f"Prompt tokens: {usage['prompt_tokens']}")
        print(f"Completion tokens: {usage['completion_tokens']}")
        print(f"Total tokens: {usage['total_tokens']}")
        
    except Exception as exc:
        print(f"Error: {type(exc).__name__}: {exc}")


# ========================== Ray Concurrent Processing ==========================
if RAY_AVAILABLE:
    @ray.remote
    class RayAPIWorker:
        """Ray actor for parallel API calls."""
        
        def __init__(self, api_configs: List[ApiConfig]):
            self.client_manager = ClientManager(api_configs)
            self.event_tracker = APIEventTracker()
        
        def process_request(
            self,
            request_id: int,
            prompt: str,
            max_retries: int = 5,
            response_format_cls: Optional[str] = None,
            system_prompt: str = "You are a helpful assistant.",
        ) -> Dict[str, Any]:
            """
            Process a single request.
            
            Args:
                request_id: Unique request identifier
                prompt: User prompt
                max_retries: Maximum retry attempts
                response_format_cls: Optional fully qualified class name for structured output
                system_prompt: System prompt
                
            Returns:
                Dict with request_id, success, response, usage, error
            """
            try:
                # Parse response format if provided
                response_format = None
                if response_format_cls:
                    # Dynamic import of response format class
                    module_name, class_name = response_format_cls.rsplit(".", 1)
                    import importlib
                    module = importlib.import_module(module_name)
                    response_format = getattr(module, class_name)
                
                result = call_with_retry(
                    self.client_manager,
                    prompt,
                    max_retries=max_retries,
                    response_format=response_format,
                    system_prompt=system_prompt,
                    task_id=f"request_{request_id}",
                )
                
                if result:
                    response_text, usage = result
                    return {
                        "request_id": request_id,
                        "success": True,
                        "response": response_text,
                        "usage": usage,
                        "error": None,
                    }
                else:
                    return {
                        "request_id": request_id,
                        "success": False,
                        "response": None,
                        "usage": None,
                        "error": "All retries failed",
                    }
                    
            except Exception as exc:
                return {
                    "request_id": request_id,
                    "success": False,
                    "response": None,
                    "usage": None,
                    "error": str(exc),
                }


def process_batch_with_ray(
    prompts: List[str],
    api_family: str = "gpt4o",
    num_workers: int = 10,
    max_retries: int = 5,
    response_format_cls: Optional[str] = None,
    system_prompt: str = "You are a helpful assistant.",
) -> List[Dict[str, Any]]:
    """
    Process multiple prompts in parallel using Ray.
    
    Args:
        prompts: List of prompts to process
        api_family: API family to use (e.g., 'gpt4o', 'gpt5', 'gpt41')
        num_workers: Number of Ray workers
        max_retries: Maximum retry attempts per request
        response_format_cls: Optional fully qualified class name for structured output
        system_prompt: System prompt
        
    Returns:
        List of result dicts with request_id, success, response, usage, error
    """
    if not RAY_AVAILABLE:
        raise ImportError("Ray is not installed. Install with: pip install ray")
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    api_configs = get_api_configs(api_family)
    
    # Create workers
    workers = [RayAPIWorker.remote(api_configs) for _ in range(num_workers)]
    
    # Submit all tasks
    futures = []
    for i, prompt in enumerate(prompts):
        worker = workers[i % num_workers]
        future = worker.process_request.remote(
            i, prompt, max_retries, response_format_cls, system_prompt
        )
        futures.append(future)
    
    # Collect results
    results = ray.get(futures)
    
    # Sort by request_id to maintain order
    results.sort(key=lambda x: x["request_id"])
    
    return results


def process_batch_sequential(
    prompts: List[str],
    api_family: str = "gpt4o",
    max_retries: int = 5,
    response_format: Optional[Type[BaseModel]] = None,
    system_prompt: str = "You are a helpful assistant.",
) -> List[Dict[str, Any]]:
    """
    Process multiple prompts sequentially (no concurrency).
    
    Args:
        prompts: List of prompts to process
        api_family: API family to use
        max_retries: Maximum retry attempts per request
        response_format: Optional Pydantic model for structured output
        system_prompt: System prompt
        
    Returns:
        List of result dicts
    """
    api_configs = get_api_configs(api_family)
    client_manager = ClientManager(api_configs)
    
    results = []
    for i, prompt in enumerate(prompts):
        print(f"Processing request {i+1}/{len(prompts)}...")
        
        result = call_with_retry(
            client_manager,
            prompt,
            max_retries=max_retries,
            response_format=response_format,
            system_prompt=system_prompt,
            task_id=f"request_{i}",
        )
        
        if result:
            response_text, usage = result
            results.append({
                "request_id": i,
                "success": True,
                "response": response_text,
                "usage": usage,
                "error": None,
            })
        else:
            results.append({
                "request_id": i,
                "success": False,
                "response": None,
                "usage": None,
                "error": "All retries failed",
            })
    
    return results


# ========================== Task Processor Framework ==========================
class TaskProcessor(ABC):
    """
    Abstract base class for custom task processing.
    
    Subclass this to implement your own task processing logic with:
    - prepare_input(): Prepare input data before API call
    - build_prompt(): Build the prompt from input data
    - check_output(): Validate the API response
    - process_output(): Post-process the validated output
    
    Example usage:
        class MyTaskProcessor(TaskProcessor):
            def prepare_input(self, raw_data):
                return {"text": raw_data["content"]}
            
            def build_prompt(self, input_data):
                return f"Summarize: {input_data['text']}"
            
            def check_output(self, output, input_data):
                return len(output) > 10  # Must have some content
            
            def process_output(self, output, input_data):
                return {"summary": output}
    """
    
    @abstractmethod
    def prepare_input(self, raw_data: Any) -> Any:
        """
        Prepare input data before API call.
        
        Args:
            raw_data: Raw input data for this task
            
        Returns:
            Processed input data to be used for prompt building
        """
        pass
    
    @abstractmethod
    def build_prompt(self, input_data: Any) -> str:
        """
        Build the prompt from prepared input data.
        
        Args:
            input_data: Prepared input data from prepare_input()
            
        Returns:
            Prompt string to send to the API
        """
        pass
    
    @abstractmethod
    def check_output(self, output: str, input_data: Any) -> bool:
        """
        Validate the API response.
        
        Args:
            output: API response text
            input_data: The input data used for this request
            
        Returns:
            True if output is valid, False to retry
        """
        pass
    
    def process_output(self, output: str, input_data: Any) -> Any:
        """
        Post-process the validated output (optional override).
        
        Args:
            output: Validated API response text
            input_data: The input data used for this request
            
        Returns:
            Final processed result
        """
        return output
    
    def get_system_prompt(self) -> str:
        """Get system prompt for API calls (optional override)."""
        return "You are a helpful assistant."
    
    def get_response_format(self) -> Optional[Type[BaseModel]]:
        """Get Pydantic model for structured output (optional override)."""
        return None


class SimpleTaskProcessor(TaskProcessor):
    """
    Simple task processor that passes through input as prompt.
    Useful for basic prompt-based tasks without complex processing.
    """
    
    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
        check_fn: Optional[Callable[[str, Any], bool]] = None,
        process_fn: Optional[Callable[[str, Any], Any]] = None,
    ):
        self._system_prompt = system_prompt
        self._check_fn = check_fn or (lambda output, input_data: True)
        self._process_fn = process_fn or (lambda output, input_data: output)
    
    def prepare_input(self, raw_data: Any) -> Any:
        return raw_data
    
    def build_prompt(self, input_data: Any) -> str:
        if isinstance(input_data, str):
            return input_data
        elif isinstance(input_data, dict) and "prompt" in input_data:
            return input_data["prompt"]
        else:
            return str(input_data)
    
    def check_output(self, output: str, input_data: Any) -> bool:
        return self._check_fn(output, input_data)
    
    def process_output(self, output: str, input_data: Any) -> Any:
        return self._process_fn(output, input_data)
    
    def get_system_prompt(self) -> str:
        return self._system_prompt


def process_single_task(
    task_processor: TaskProcessor,
    raw_data: Any,
    client_manager: ClientManager,
    max_api_retries: int = 5,
    max_check_retries: int = 3,
    task_id: str = "",
) -> Dict[str, Any]:
    """
    Process a single task with input preparation, API call, and output validation loop.
    
    Args:
        task_processor: TaskProcessor instance defining the task logic
        raw_data: Raw input data for this task
        client_manager: ClientManager for API calls
        max_api_retries: Max retries for API errors
        max_check_retries: Max retries when output validation fails
        task_id: Task identifier for logging
        
    Returns:
        Dict with success, input, output, processed_output, attempts, error
    """
    result = {
        "task_id": task_id,
        "success": False,
        "input": None,
        "output": None,
        "processed_output": None,
        "attempts": 0,
        "error": None,
    }
    
    try:
        # Step 1: Prepare input
        input_data = task_processor.prepare_input(raw_data)
        result["input"] = input_data
        
        # Step 2: Build prompt
        prompt = task_processor.build_prompt(input_data)
        system_prompt = task_processor.get_system_prompt()
        response_format = task_processor.get_response_format()
        
        # Step 3: API call with output validation loop
        for check_attempt in range(max_check_retries):
            result["attempts"] = check_attempt + 1
            
            # Call API
            api_result = call_with_retry(
                client_manager,
                prompt,
                max_retries=max_api_retries,
                response_format=response_format,
                system_prompt=system_prompt,
                task_id=f"{task_id}_attempt_{check_attempt}",
            )
            
            if api_result is None:
                result["error"] = f"API call failed after {max_api_retries} retries"
                continue
            
            output_text, usage = api_result
            result["output"] = output_text
            result["usage"] = usage
            
            # Step 4: Check output validity
            try:
                if task_processor.check_output(output_text, input_data):
                    # Step 5: Process output
                    processed = task_processor.process_output(output_text, input_data)
                    result["processed_output"] = processed
                    result["success"] = True
                    result["error"] = None
                    debug_print(f"✅ Task {task_id} succeeded on attempt {check_attempt + 1}")
                    return result
                else:
                    result["error"] = f"Output validation failed on attempt {check_attempt + 1}"
                    debug_print(f"⚠️ Task {task_id} output validation failed, retrying...")
            except Exception as check_exc:
                result["error"] = f"Output check error: {str(check_exc)}"
                debug_print(f"⚠️ Task {task_id} check error: {check_exc}")
        
        # All check retries exhausted
        if not result["success"]:
            result["error"] = f"All {max_check_retries} validation attempts failed"
            print(f"❌ Task {task_id} failed after {max_check_retries} attempts")
            
    except Exception as exc:
        result["error"] = f"Task processing error: {str(exc)}\n{traceback.format_exc()}"
        print(f"❌ Task {task_id} error: {exc}")
    
    return result


def process_tasks_sequential(
    task_processor: TaskProcessor,
    data_list: List[Any],
    api_family: str = "gpt4o",
    max_api_retries: int = 5,
    max_check_retries: int = 3,
) -> List[Dict[str, Any]]:
    """
    Process multiple tasks sequentially.
    
    Args:
        task_processor: TaskProcessor instance
        data_list: List of raw input data items
        api_family: API family to use
        max_api_retries: Max retries for API errors per task
        max_check_retries: Max retries for output validation per task
        
    Returns:
        List of result dicts
    """
    api_configs = get_api_configs(api_family)
    client_manager = ClientManager(api_configs)
    
    results = []
    for i, raw_data in enumerate(data_list):
        print(f"Processing task {i+1}/{len(data_list)}...")
        result = process_single_task(
            task_processor,
            raw_data,
            client_manager,
            max_api_retries=max_api_retries,
            max_check_retries=max_check_retries,
            task_id=f"task_{i}",
        )
        results.append(result)
    
    return results


# Ray-based parallel task processing
if RAY_AVAILABLE:
    @ray.remote
    class RayTaskWorker:
        """Ray actor for parallel task processing with validation loop."""
        
        def __init__(self, api_configs: List[ApiConfig]):
            self.client_manager = ClientManager(api_configs)
        
        def process_task(
            self,
            task_id: int,
            raw_data: Any,
            processor_cls: str,
            processor_kwargs: Dict[str, Any],
            max_api_retries: int = 5,
            max_check_retries: int = 3,
        ) -> Dict[str, Any]:
            """
            Process a single task within Ray worker.
            
            Args:
                task_id: Task identifier
                raw_data: Raw input data
                processor_cls: Fully qualified class name of TaskProcessor
                processor_kwargs: Kwargs to instantiate the processor
                max_api_retries: Max API retries
                max_check_retries: Max validation retries
                
            Returns:
                Result dict
            """
            try:
                # Dynamically instantiate the task processor
                module_name, class_name = processor_cls.rsplit(".", 1)
                import importlib
                module = importlib.import_module(module_name)
                ProcessorClass = getattr(module, class_name)
                task_processor = ProcessorClass(**processor_kwargs)
                
                return process_single_task(
                    task_processor,
                    raw_data,
                    self.client_manager,
                    max_api_retries=max_api_retries,
                    max_check_retries=max_check_retries,
                    task_id=f"task_{task_id}",
                )
            except Exception as exc:
                return {
                    "task_id": f"task_{task_id}",
                    "success": False,
                    "input": raw_data,
                    "output": None,
                    "processed_output": None,
                    "attempts": 0,
                    "error": f"Worker error: {str(exc)}\n{traceback.format_exc()}",
                }


def process_tasks_with_ray(
    task_processor: TaskProcessor,
    data_list: List[Any],
    api_family: str = "gpt4o",
    num_workers: int = 10,
    max_api_retries: int = 5,
    max_check_retries: int = 3,
    processor_cls: Optional[str] = None,
    processor_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Process multiple tasks in parallel using Ray.
    
    Args:
        task_processor: TaskProcessor instance (used if processor_cls is None)
        data_list: List of raw input data items
        api_family: API family to use
        num_workers: Number of Ray workers
        max_api_retries: Max retries for API errors per task
        max_check_retries: Max retries for output validation per task
        processor_cls: Fully qualified class name for Ray workers (required for Ray)
        processor_kwargs: Kwargs to instantiate the processor in workers
        
    Returns:
        List of result dicts
    """
    if not RAY_AVAILABLE:
        raise ImportError("Ray is not installed. Install with: pip install ray")
    
    if processor_cls is None:
        # Fall back to sequential processing if no processor class specified
        print("Warning: processor_cls not specified, falling back to sequential processing")
        return process_tasks_sequential(
            task_processor, data_list, api_family, max_api_retries, max_check_retries
        )
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    api_configs = get_api_configs(api_family)
    processor_kwargs = processor_kwargs or {}
    
    # Create workers
    workers = [RayTaskWorker.remote(api_configs) for _ in range(num_workers)]
    
    # Submit all tasks
    futures = []
    for i, raw_data in enumerate(data_list):
        worker = workers[i % num_workers]
        future = worker.process_task.remote(
            i, raw_data, processor_cls, processor_kwargs,
            max_api_retries, max_check_retries
        )
        futures.append(future)
    
    # Collect results with progress
    from tqdm import tqdm
    results = []
    for future in tqdm(futures, desc="Processing tasks"):
        result = ray.get(future)
        results.append(result)
    
    # Sort by task_id to maintain order
    results.sort(key=lambda x: int(x["task_id"].split("_")[-1]))
    
    return results


# ========================== Config Loading ==========================
def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_from_config(config_path: Union[str, Path]) -> None:
    """
    Run API calls based on YAML configuration.
    
    Args:
        config_path: Path to YAML config file
    """
    config = load_config(config_path)
    
    global DEBUG_MODE
    DEBUG_MODE = config.get("debug", False)
    
    mode = config.get("mode", "single_test")
    
    if mode == "single_test":
        api_config = None
        if "api_config" in config:
            api_config = ApiConfig(**config["api_config"])
        
        single_test(
            prompt=config.get("prompt", "Hello, how are you?"),
            api_config=api_config,
            system_prompt=config.get("system_prompt", "You are a helpful assistant."),
        )
        
    elif mode == "batch_sequential":
        results = process_batch_sequential(
            prompts=config.get("prompts", []),
            api_family=config.get("api_family", "gpt4o"),
            max_retries=config.get("max_retries", 5),
            system_prompt=config.get("system_prompt", "You are a helpful assistant."),
        )
        
        # Save results if output path specified
        if "output_path" in config:
            output_path = Path(config["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_path}")
        else:
            print("Results:", json.dumps(results, indent=2, ensure_ascii=False))
            
    elif mode == "batch_ray":
        results = process_batch_with_ray(
            prompts=config.get("prompts", []),
            api_family=config.get("api_family", "gpt4o"),
            num_workers=config.get("num_workers", 10),
            max_retries=config.get("max_retries", 5),
            response_format_cls=config.get("response_format_cls"),
            system_prompt=config.get("system_prompt", "You are a helpful assistant."),
        )
        
        # Save results if output path specified
        if "output_path" in config:
            output_path = Path(config["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_path}")
        else:
            print("Results:", json.dumps(results, indent=2, ensure_ascii=False))
    
    elif mode == "task_sequential":
        # Load data from file or use inline data
        data_list = config.get("data_list", [])
        if "data_file" in config:
            data_file = Path(config["data_file"])
            with data_file.open("r", encoding="utf-8") as f:
                data_list = json.load(f)
        
        # Create task processor
        processor_cls = config.get("processor_cls")
        processor_kwargs = config.get("processor_kwargs", {})
        
        if processor_cls:
            module_name, class_name = processor_cls.rsplit(".", 1)
            import importlib
            module = importlib.import_module(module_name)
            ProcessorClass = getattr(module, class_name)
            task_processor = ProcessorClass(**processor_kwargs)
        else:
            # Use SimpleTaskProcessor with config
            task_processor = SimpleTaskProcessor(
                system_prompt=config.get("system_prompt", "You are a helpful assistant."),
            )
        
        results = process_tasks_sequential(
            task_processor=task_processor,
            data_list=data_list,
            api_family=config.get("api_family", "gpt4o"),
            max_api_retries=config.get("max_api_retries", 5),
            max_check_retries=config.get("max_check_retries", 3),
        )
        
        # Save results
        if "output_path" in config:
            output_path = Path(config["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_path}")
        else:
            print("Results:", json.dumps(results, indent=2, ensure_ascii=False))
    
    elif mode == "task_ray":
        # Load data from file or use inline data
        data_list = config.get("data_list", [])
        if "data_file" in config:
            data_file = Path(config["data_file"])
            with data_file.open("r", encoding="utf-8") as f:
                data_list = json.load(f)
        
        processor_cls = config.get("processor_cls")
        processor_kwargs = config.get("processor_kwargs", {})
        
        # For Ray mode, we need the class name to instantiate in workers
        if processor_cls:
            module_name, class_name = processor_cls.rsplit(".", 1)
            import importlib
            module = importlib.import_module(module_name)
            ProcessorClass = getattr(module, class_name)
            task_processor = ProcessorClass(**processor_kwargs)
        else:
            task_processor = SimpleTaskProcessor(
                system_prompt=config.get("system_prompt", "You are a helpful assistant."),
            )
            processor_cls = "azure_api.SimpleTaskProcessor"
            processor_kwargs = {"system_prompt": config.get("system_prompt", "You are a helpful assistant.")}
        
        results = process_tasks_with_ray(
            task_processor=task_processor,
            data_list=data_list,
            api_family=config.get("api_family", "gpt4o"),
            num_workers=config.get("num_workers", 10),
            max_api_retries=config.get("max_api_retries", 5),
            max_check_retries=config.get("max_check_retries", 3),
            processor_cls=processor_cls,
            processor_kwargs=processor_kwargs,
        )
        
        # Save results
        if "output_path" in config:
            output_path = Path(config["output_path"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_path}")
        else:
            print("Results:", json.dumps(results, indent=2, ensure_ascii=False))
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Available: single_test, batch_sequential, batch_ray, task_sequential, task_ray")


# ========================== Main Entry Point ==========================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Azure OpenAI API Client")
    parser.add_argument(
        "--config",
        type=str,
        default="basic_gpt_api.yaml",
        help="Path to YAML configuration file (all parameters are read from this file)",
    )
    
    args = parser.parse_args()
    
    # All parameters are read from YAML config file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Creating default config file...")
        default_config = {
            "mode": "single_test",
            "debug": False,
            "api_family": "gpt4o",
            "prompt": "Hello, how are you?",
            "system_prompt": "You are a helpful assistant.",
            "max_retries": 5,
            "num_workers": 10,
        }
        with config_path.open("w", encoding="utf-8") as f:
            yaml.dump(default_config, f, default_flow_style=False)
        print(f"Default config created at: {config_path}")
    
    run_from_config(config_path)
