import os
from typing import Any

class LLMFactory:
    """
    A factory class to create LLM instances based on the environment configuration.
    Supports:
    - MOCK: Returns fixed test data for development.
    - CLOUD: Uses DeepSeek API (for Phase 2 online inference judge/analysis).
    - LOCAL: Uses local GGUF model via llama-cpp-python (for privacy-preserving inference).
    """
    
    @staticmethod
    def create_llm(llm_type: str = None) -> Any:
        """
        Creates and returns an LLM instance.
        
        Args:
            llm_type (str, optional): The type of LLM to create. 
                                      Defaults to env var LLM_TYPE or 'MOCK'.
        """
        if llm_type is None:
            llm_type = os.getenv("LLM_TYPE", "MOCK").upper()
            
        if llm_type == "LOCAL":
            # Lazy import to avoid dependency issues if not installed
            try:
                from llama_cpp import Llama
            except ImportError:
                raise ImportError("llama-cpp-python is not installed. Please install it to use LOCAL mode.")
                
            model_path = os.getenv("LOCAL_MODEL_PATH", "models/psychologist-8b-q4_k_m.gguf")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"GGUF model not found at {model_path}. Please download it from Colab.")
                
            return Llama(
                model_path=model_path,
                n_gpu_layers=-1, # Offload all layers to GPU if possible
                n_ctx=4096,
                verbose=False
            )
            
        elif llm_type == "CLOUD":
            from src.api.deepseek_client import DeepseekClient
            from src.api.models import APIConfig

            config = APIConfig.from_env()
            if not config.api_key:
                raise ValueError("DEEPSEEK_API_KEY environment variable not set.")

            return DeepseekClient(config=config, mock_mode=False)
            
        else:
            return MockLLM()

class MockLLM:
    """A mock LLM for testing purposes."""
    def __call__(self, prompt: str, **kwargs):
        return {
            "choices": [
                {
                    "text": "This is a mock response from the Psychologist Agent [MOCK MODE].",
                    "finish_reason": "stop"
                }
            ]
        }
