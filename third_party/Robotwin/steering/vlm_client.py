import base64
import json
import re
import requests
from io import BytesIO
from typing import List, Optional, Tuple
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

    def _extract_chosen_trajectories(self, text_output: str, num_trajectories: int) -> Tuple[int, int]:
        """
        Parses the text output to extract chosen trajectory indices for both arms.
        Looks for 'chosen_trajectory_left' and 'chosen_trajectory_right'.
        
        Returns:
            tuple: (idx_left, idx_right) - 0-based indices.
        """
        # Define color to index mapping
        color_to_idx = {
            "red": 0, "orange": 1, "blue": 2, "cyan": 3, "magenta": 4, "none": -1
        }
        
        # Default fallback values
        results = {"left": 0, "right": 0}
        
        try:
            # Try JSON format first
            json_start_index = text_output.rfind('{')
            json_end_index = text_output.rfind('}') + 1
            if json_start_index != -1 and json_end_index > json_start_index:
                json_str = text_output[json_start_index:json_end_index]
                data = json.loads(json_str)
                
                for side in ["left", "right"]:
                    key = f"chosen_trajectory_{side}"
                    if key in data:
                        choice = data[key]
                        
                        # Handle color-based choice
                        if isinstance(choice, str):
                            choice_lower = choice.lower().strip()
                            if choice_lower in color_to_idx:
                                idx = color_to_idx[choice_lower]
                                results[side] = idx if idx != -1 else random.randint(0, num_trajectories - 1)
                            else:
                                # Regex fallback for string numbers
                                match = re.search(r'\d+', choice)
                                if match:
                                    results[side] = int(match.group())
                        
                        # Handle numeric choice
                        elif isinstance(choice, int):
                            results[side] = choice
                
                # Validation
                return (
                    max(0, min(results["left"], num_trajectories - 1)),
                    max(0, min(results["right"], num_trajectories - 1))
                )

        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        
        # Fallback: Simple pattern matching for "left: X" or "right: Y"
        for side in ["left", "right"]:
            pattern = rf"{side}[:\s]+(?:trajectory|option|color)?[:\s]*(\w+)"
            match = re.search(pattern, text_output, re.IGNORECASE)
            if match:
                val = match.group(1).lower()
                if val in color_to_idx:
                    results[side] = color_to_idx[val] if color_to_idx[val] != -1 else 0
                elif val.isdigit():
                    results[side] = int(val)

        print(f"Warning: Partial or failed parse. Using: L={results['left']}, R={results['right']}")
        return results["left"], results["right"]

    def select_trajectories(
        self, 
        annotated_image: Image.Image, 
        prompt_text: str,
        num_trajectories: int,
        max_new_tokens: int = 1024,
        timeout: int = 60
    ) -> tuple[Tuple[int, int], str]:
        """
        Send an annotated image to the VLM and get trajectory selections for both arms.
        
        Returns:
            tuple: ((idx_left, idx_right), text_response)
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
                {"role": "system", "content": "You are an expert robot policy advisor specializing in dual-arm coordination."},
                {"role": "user", "content": user_content}
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
            
            chosen_idxs = self._extract_chosen_trajectories(generated_text, num_trajectories)
            self.last_text_responses = [generated_text]
            
            return chosen_idxs, generated_text

        except Exception as e:
            print(f"VLM Request failed: {e}. Defaulting to (0, 0).")
            return (0, 0), str(e)

    def get_last_text_responses(self) -> List[str]:
        """Get the text responses from the last VLM call."""
        return self.last_text_responses