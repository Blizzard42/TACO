import base64
import json
import re
import requests
from io import BytesIO
from typing import List, Optional, Tuple
import torch
from PIL import Image
import random

import logging
log = logging.getLogger(__name__)


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
        self.sampling_params = self._get_sampling_params(model_name)
        print('[VLMClient] sampling params for %s: %s' % (model_name, self.sampling_params))
        
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

    @staticmethod
    def _get_sampling_params(model_name: str) -> dict:
        """Return server-side sampling params tuned per model family.

        Mirrors the recipe in open-pi-zero src/steering/vlm_client.py.
        """
        n = model_name.lower()
        if "qwen3-vl" in n:
            return {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.0,
            }
        if any(k in n for k in ("qwen3.6", "qwen3.8", "a3b")):
            # Thinking models stream chain-of-thought into a separate `reasoning`
            # field and leave `content` null until it terminates, which breaks
            # steering. Disabling thinking puts the answer back in `content`.
            # Rides in the payload as a top-level key, not a sampling parameter.
            return {
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        # Default: Qwen2.5-VL and others -> deterministic.
        return {"temperature": 0.0}

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
            **self.sampling_params,
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
        
    def _extract_chosen_primitives(self, response_text: str) -> dict:
        """
        Extracts 'chosen_primitive_left', 'chosen_primitive_right', 
        'gripper_state_left', and 'gripper_state_right' from VLM response.
        Handles standard JSON and Markdown formatting.
        
        Args:
            response_text (str): The raw string returned by the VLM.
            
        Returns:
            dict: {'chosen_primitive_left': str, 'chosen_primitive_right': str,
                   'gripper_state_left': str, 'gripper_state_right': str}
                Values are None if extraction fails.
        """
        # Default return structure
        extracted_data: dict[str, str] = {
            "chosen_primitive_left": "None",
            "chosen_primitive_right": "None",
            "gripper_state_left": "None",
            "gripper_state_right": "None"
        }

        # 1. Clean up Markdown code blocks (e.g., ```json ... ```)
        clean_text = response_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:]
        
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        
        clean_text = clean_text.strip()

        # 2. Attempt strict JSON parsing
        try:
            data = json.loads(clean_text)
            extracted_data["chosen_primitive_left"] = data.get("chosen_primitive_left")
            extracted_data["chosen_primitive_right"] = data.get("chosen_primitive_right")
            extracted_data["gripper_state_left"] = data.get("gripper_state_left")
            extracted_data["gripper_state_right"] = data.get("gripper_state_right")
            return extracted_data
        except json.JSONDecodeError:
            log.warning(f"JSON parse failed for VLM response. Attempting Regex fallback. Response: {response_text[:50]}...")

        # 3. Fallback: Regex extraction
        # This handles cases where the VLM adds extra text outside the JSON block
        
        # Extract chosen_primitive_left
        traj_pattern_left = r'"chosen_primitive_left"\s*:\s*"([^"]+)"'
        traj_match_left = re.search(traj_pattern_left, response_text)
        if traj_match_left:
            extracted_data["chosen_primitive_left"] = traj_match_left.group(1)

        # Extract chosen_primitive_right
        traj_pattern_right = r'"chosen_primitive_right"\s*:\s*"([^"]+)"'
        traj_match_right = re.search(traj_pattern_right, response_text)
        if traj_match_right:
            extracted_data["chosen_primitive_right"] = traj_match_right.group(1)

        # Extract gripper_state_left
        state_pattern_left = r'"gripper_state_left"\s*:\s*"([^"]+)"'
        state_match_left = re.search(state_pattern_left, response_text)
        if state_match_left:
            extracted_data["gripper_state_left"] = state_match_left.group(1)

        # Extract gripper_state_right
        state_pattern_right = r'"gripper_state_right"\s*:\s*"([^"]+)"'
        state_match_right = re.search(state_pattern_right, response_text)
        if state_match_right:
            extracted_data["gripper_state_right"] = state_match_right.group(1)

        return extracted_data
    
    def select_primitives(
        self, 
        annotated_image: Image.Image, 
        prompt_text: str,
        primitive_names: List[str],
        max_new_tokens: int = 1024,
        timeout: int = 60
    ) -> tuple[Tuple[int, int], str]:
        """
        Send an annotated image to the VLM and get trajectory selections for both arms.
        
        Returns:
            tuple: (dict[str, int], text_response)
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
            **self.sampling_params,
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
            
            extracted_data = self._extract_chosen_primitives(generated_text)
            for key, value in extracted_data.items():
                if value == "None" or value is None:
                    extracted_data[key] = 0
                else:
                    if 'chosen_primitive' in key:
                        try:
                            extracted_data[key] = primitive_names.index(value)
                        except ValueError:
                            extracted_data[key] = 0
                    else: # Gripper_state
                        if value.lower() == "open":
                            extracted_data[key] = 1.0
                        else:
                            extracted_data[key] = 0.0
            
            return extracted_data, generated_text

        except Exception as e:
            print(f"VLM Request failed: {e}. Defaulting to (0, 0).")
            return {"chosen_primitive_left": 0, "chosen_primitive_right": 0, "gripper_state_left": 0, "gripper_state_right": 0}, str(e)
        
    def get_last_text_responses(self) -> List[str]:
        """Get the text responses from the last VLM call."""
        return self.last_text_responses