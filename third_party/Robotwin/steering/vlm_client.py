import base64
import json
import re
import requests
from io import BytesIO
from typing import List, Optional
import torch
from PIL import Image
import random


class VLMClient:
    """
    A simple VLM client to interact with OpenAI-compatible VLM servers.
    Sends images with trajectory visualizations to the VLM and gets trajectory selection.
    """

    def __init__(self, server_url: str, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"):
        """
        Initialize the VLM Client.
        
        Args:
            server_url: The base URL of the OpenAI-compatible VLM server.
            model_name: The model name identifier.
        """
        self.server_url = server_url
        self.endpoint = f"{server_url}/v1/chat/completions"
        self.model_name = model_name
        
        # Store intermediate text responses
        self.last_text_responses = []
        
        # Health check
        health_url = server_url.rstrip('/') + "/health"
        try:
            health_response = requests.get(health_url, timeout=5)
            health_response.raise_for_status()
            print(f"Successfully connected to VLM server at {server_url}")
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not connect to VLM server at {health_url}: {e}")

    def _pil_to_base64(self, image: Image.Image) -> str:
        """Converts a PIL Image to a base64 encoded string."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def _extract_chosen_trajectory(self, text_output: str, num_trajectories: int) -> int:
        """
        Parses the text output to extract the chosen trajectory index.
        Supports both color-based ("red", "orange", "blue", "cyan", "magenta", "none") 
        and numeric (0, 1, 2, ...) trajectory choices.
        
        Returns:
            int: The chosen trajectory index (0-based), or -1 if "none" is chosen, or 0 as default.
        """
        # Define color to index mapping
        color_to_idx = {
            "red": 0,
            "orange": 1,
            "blue": 2,
            "cyan": 3,
            "magenta": 4,
            "none": -1
        }
        
        try:
            # Try JSON format first
            json_start_index = text_output.rfind('{')
            json_end_index = text_output.rfind('}') + 1
            if json_start_index != -1 and json_end_index > json_start_index:
                json_str = text_output[json_start_index:json_end_index]
                data = json.loads(json_str)
                
                if "chosen_trajectory" in data:
                    choice = data["chosen_trajectory"]
                    
                    # Handle color-based choice
                    if isinstance(choice, str):
                        choice_lower = choice.lower().strip()
                        if choice_lower in color_to_idx:
                            idx = color_to_idx[choice_lower]
                            if idx == -1:  # "none" was chosen
                                print(f"VLM rejected all trajectories (chose 'none'). Defaulting to trajectory {random.randint(0, num_trajectories - 1)}.")
                                return random.randint(0, num_trajectories - 1)
                            if 0 <= idx < num_trajectories:
                                return idx
                        
                        # Try to extract number from string (backward compatibility)
                        match = re.search(r'\d+', choice)
                        if match:
                            idx = int(match.group())
                            if 0 <= idx < num_trajectories:
                                return idx
                    
                    # Handle numeric choice
                    elif isinstance(choice, int) and 0 <= choice < num_trajectories:
                        return choice
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        
        # Try simple pattern matching for backward compatibility
        patterns = [
            r"trajectory[:\s]+(\d+)",
            r"choose[:\s]+(\d+)",
            r"select[:\s]+(\d+)",
            r"option[:\s]+(\d+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_output, re.IGNORECASE)
            if match:
                idx = int(match.group(1))
                if 0 <= idx < num_trajectories:
                    return idx
        
        print(f"Warning: Could not parse trajectory choice from VLM output. Defaulting to trajectory 0.")
        print(f"Output: {text_output[:200]}...")
        return 0

    def select_trajectory(
        self, 
        annotated_image: Image.Image, 
        prompt_text: str,
        num_trajectories: int,
        max_new_tokens: int = 1024,
        timeout: int = 60,
        primitive: bool = False
    ) -> tuple[int, str]:
        """
        Send an annotated image to the VLM and get trajectory selection.
        
        Args:
            annotated_image: PIL Image with trajectories drawn on it.
            prompt_text: The prompt to send to the VLM.
            num_trajectories: Number of trajectories drawn on the image.
            max_new_tokens: Maximum tokens in VLM response.
            timeout: Request timeout in seconds.
            
        Returns:
            tuple: (chosen_trajectory_index, text_response)
        """
        user_content = [
            {"type": "text", "text": prompt_text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{self._pil_to_base64(annotated_image)}"
                }
            }
        ]
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert robot policy advisor."
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        }

        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"]
            
            if primitive:
                return None, generated_text
            
            chosen_idx = self._extract_chosen_trajectory(generated_text, num_trajectories)
            self.last_text_responses = [generated_text]
            
            return chosen_idx, generated_text

        except requests.exceptions.Timeout:
            print("Error: Request to VLM server timed out. Defaulting to trajectory 0.")
            return random.randint(0, num_trajectories - 1), ""
        except requests.exceptions.RequestException as e:
            print(f"Error during request to VLM server: {e}. Defaulting to trajectory 0.")
            return random.randint(0, num_trajectories - 1), ""
        except (KeyError, IndexError) as e:
            print(f"Error parsing VLM server response: {e}. Defaulting to trajectory 0.")
            return random.randint(0, num_trajectories - 1), ""

    def get_last_text_responses(self) -> List[str]:
        """Get the text responses from the last VLM call."""
        return self.last_text_responses

