import requests
from typing import List, Dict, Optional
import os


class OllamaClient:
    """Ollama HTTP client dùng requests"""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        
        print(f"🔵 [Ollama Client] Initialized: {self.base_url}")
    
    def embeddings(self, model: str, prompt: str) -> Dict:
        """
        Get text embeddings
        
        Args:
            model: Model name (e.g. "nomic-embed-text")
            prompt: Text to embed
        
        Returns:
            {"embedding": [0.1, 0.2, ...]}
        """
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": model,
            "prompt": prompt
        }
        
        try:
            response = self.session.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.Timeout:
            raise Exception(f"Ollama embedding timeout for model {model}")
        except requests.exceptions.ConnectionError:
            raise Exception(f"Cannot connect to Ollama at {self.base_url}")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"Ollama HTTP error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise Exception(f"Ollama error: {str(e)}")
    
    def chat(
        self,
        model: str,
        messages: List[Dict],
        options: Optional[Dict] = None
    ) -> Dict:
        """
        Chat completion
        
        Args:
            model: Model name (e.g. "qwen2.5:7b")
            messages: [{"role": "user", "content": "..."}]
            options: {"temperature": 0.7, "num_predict": 500}
        
        Returns:
            {
                "model": "qwen2.5:7b",
                "message": {"role": "assistant", "content": "..."},
                "done": true
            }
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        
        if options:
            payload["options"] = options
        
        try:
            response = self.session.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.Timeout:
            raise Exception(f"Ollama chat timeout for model {model}")
        except requests.exceptions.ConnectionError:
            raise Exception(f"Cannot connect to Ollama at {self.base_url}")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"Ollama HTTP error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise Exception(f"Ollama error: {str(e)}")
    
    def health_check(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'session'):
            try:
                self.session.close()
            except:
                pass


# Singleton instance
ollama_client = OllamaClient()