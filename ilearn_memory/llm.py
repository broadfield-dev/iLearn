import os
import requests
import json
import logging
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

# Load model data from the JSON file within the package
try:
    _models_file_path = os.path.join(os.path.dirname(__file__), 'models.json')
    with open(_models_file_path, 'r') as f:
        MODELS_BY_PROVIDER = json.load(f)
except Exception:
    log.error("models.json not found or failed to load. The package might be improperly installed.")
    MODELS_BY_PROVIDER = {}

API_KEYS_ENV_VARS = {
  "GROQ": 'GROQ_API_KEY', "OPENROUTER": 'OPENROUTER_API_KEY',
  "OPENAI": 'OPENAI_API_KEY', "GOOGLE": 'GOOGLE_API_KEY', "COHERE": 'COHERE_API_KEY',
}

API_URLS = {
  "GROQ": 'https://api.groq.com/openai/v1/chat/completions',
  "OPENROUTER": 'https://openrouter.ai/api/v1/chat/completions',
  "OPENAI": 'https://api.openai.com/v1/chat/completions',
  "GOOGLE": 'https://generativelanguage.googleapis.com/v1beta/models/',
}

def _get_api_key(provider: str, api_key_override: str = None) -> str | None:
    provider_upper = provider.upper()
    if api_key_override: return api_key_override
    env_var_name = API_KEYS_ENV_VARS.get(provider_upper)
    return os.getenv(env_var_name) if env_var_name else None

async def call_model_stream(provider: str, model_display_name: str, messages: list[dict], api_key_override: str = None, temperature: float = 0.7, max_tokens: int = 2048):
    """
    Calls the specified model via its provider and streams the response.
    Yields chunks of the response text.
    """
    provider_lower = provider.lower()
    api_key = _get_api_key(provider_lower, api_key_override)
    base_url = API_URLS.get(provider.upper())
    models_dict = MODELS_BY_PROVIDER.get(provider_lower, {}).get("models", {})
    model_id = models_dict.get(model_display_name)

    if not api_key:
        yield f"[Error: API Key for {provider} not found.]"
        return
    if not base_url or not model_id:
        yield f"[Error: Model or provider '{provider}/{model_display_name}' not configured.]"
        return

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model_id, "messages": messages, "stream": True, "temperature": temperature}
    if max_tokens: payload["max_tokens"] = max_tokens
    
    if provider_lower in ["groq", "openrouter", "openai"]:
        request_url = base_url
    elif provider_lower == "google":
        system_instruction = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
        filtered_messages = [{"role": "model" if m["role"] == "assistant" else m["role"], "parts": [{"text": m["content"]}]} for m in messages if m["role"] != "system"]
        payload = {"contents": filtered_messages, "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}}
        if system_instruction: payload["system_instruction"] = {"parts": [{"text": system_instruction}]}
        request_url = f"{base_url}{model_id}:streamGenerateContent?key={api_key}"
    else:
        yield f"[Error: Provider '{provider}' not yet supported in this handler.]"
        return

    try:
        response = requests.post(request_url, headers=headers, json=payload, stream=True, timeout=180)
        response.raise_for_status()

        for chunk in response.iter_content(chunk_size=None):
            decoded_chunk = chunk.decode('utf-8', errors='replace')
            if provider_lower in ["groq", "openrouter", "openai"]:
                for line in decoded_chunk.splitlines():
                    if line.startswith('data: '):
                        data_json = line[len('data: '):].strip()
                        if data_json == '[DONE]': continue
                        try:
                            data = json.loads(data_json)
                            if data.get("choices") and data["choices"][0].get("delta", {}).get("content"):
                                yield data["choices"][0]["delta"]["content"]
                        except json.JSONDecodeError: continue
            elif provider_lower == "google":
                try:
                    data = json.loads(decoded_chunk)
                    if data.get("candidates") and data["candidates"][0].get("content", {}).get("parts"):
                        yield data["candidates"][0]["content"]["parts"][0]["text"]
                except json.JSONDecodeError: continue

    except requests.exceptions.HTTPError as e:
        err_text = e.response.text[:200]
        log.error(f"API HTTP Error ({e.response.status_code}) for {provider}: {err_text}")
        yield f"[Error: API returned {e.response.status_code}.]"
    except Exception as e:
        log.error(f"Unexpected error during {provider} stream: {e}", exc_info=True)
        yield f"[Error: An unexpected error occurred: {e}]"
