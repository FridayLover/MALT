import json
import requests
import time
from IPython.display import clear_output

class GPTChatbot:
    def __init__(self):
        self.api_url = "http://my.lab.ip.for.calling.model/" #"my local model"
        self.headers = {"Content-Type": "application/json"}
        self.history = []

    def generate_response(self, user_input, temp=0.0, top_p=1.0, top_k=100, presence_penalty=0.0, frequency_penalty=0.0, max_tokens=-1, seed=-1, clear=True):
        system_prompt = """You are a pioneering researcher and expert in clinical ophthalmology. Your mission is to develop a novel, heuristic-driven methodology to estimate a visual acuity score from non-traditional, and potentially noisy, digital test data.

The ultimate goal is to produce a score that is highly correlated with the gold-standard Snellen score, but your approach does not need to rigidly mimic the traditional Snellen letter-by-letter scoring process. You are free to invent a new method.

You must derive your logic from the patterns within the data itself. Think from first principles: what features in the data (e.g., accuracy at different sizes, patterns of errors, consistency of responses) are the most powerful predictors of a person's true visual threshold?

Your final output must be a robust, transparent, and explainable step-by-step algorithm. The most innovative and effective heuristic will win the competition."""
        
        messages = [{"role": "system", "content": system_prompt}] + self.history + [{"role": "user", "content": user_input}]

        payload_dict = {
            "model": "openai/gpt-oss-120b",
            "messages": messages,
            "temperature": temp,
            "top_p": top_p,
            "top_k": top_k,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "seed": seed,
            "max_tokens": max_tokens,
            "stream": True
        }

        start_time = time.time()
        timeout_seconds = 7199  # 120 minutes

        full_response_content = ""

        try:
            with requests.post(self.api_url, headers=self.headers, json=payload_dict, stream=True, timeout=(5, None)) as response:
                response.raise_for_status()
                print("Streaming response:")

                reasoning_started = False
                content_started = False

                for line in response.iter_lines():
                    if time.time() - start_time > timeout_seconds:
                        print(f"\n--- Stream timed out after {timeout_seconds} seconds ---")
                        full_response_content = ""
                        break

                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith('data: '):
                            json_data_str = decoded_line[len('data: '):].strip()
                            if json_data_str == "[DONE]":
                                print("\n\n--- Stream finished ---")
                                break
                            try:
                                chunk = json.loads(json_data_str)
                                delta = chunk.get('choices', [{}])[0].get('delta', {})

                                reasoning = delta.get('reasoning')
                                if reasoning:
                                    if not reasoning_started:
                                        print("\n\n--- Reasoning ---\n")
                                        reasoning_started = True
                                    print(reasoning, end='', flush=True)
                                    full_response_content += reasoning

                                content_part = delta.get('content')
                                if content_part:
                                    if not content_started:
                                        if reasoning_started:
                                            print("\n")
                                        print("\n--- Content ---\n")
                                        content_started = True
                                    print(content_part, end='', flush=True)
                                    full_response_content += content_part

                            except json.JSONDecodeError:
                                print(f"\n[Error decoding JSON: {json_data_str}]")
                            except IndexError:
                                pass
                            except Exception as e:
                                print(f"\n[Error processing chunk: {chunk} - {e}]")

        except requests.exceptions.Timeout:
            print(f"\n--- Request timed out after {time.time() - start_time:.2f} seconds ---")
            full_response_content = ""
        except requests.exceptions.RequestException as e:
            print(f"\n--- An error occurred: {e} ---")
            full_response_content = ""
        except KeyboardInterrupt:
            print("\n--- Stream interrupted by user ---")
            full_response_content = ""
        finally:
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": full_response_content})
            end_time = time.time()
            if clear:
                clear_output(wait=True)
            print(f"\n\nTotal time taken: {end_time - start_time:.2f} seconds")

        return full_response_content

    def edit_last_response(self, new_user_input, temp=0.0, top_p=1.0, top_k=100, presence_penalty=0.0, frequency_penalty=0.0, max_tokens=-1, seed=-1, clear=True):
        if len(self.history) >= 2:
            print("--- Editing last response ---")
            last_assistant_response = self.history.pop()
            last_user_prompt = self.history.pop()

            print(f"- Removed last user prompt: '{last_user_prompt['content'][:50]}...'")
            print(f"- Removed last assistant response (first 50 chars): '{last_assistant_response['content'][:50]}...'")
            print(f"- Current history length after removal: {len(self.history)}")
        else:
            print("--- History too short to edit the last response. Proceeding as a new turn. ---")

        return self.generate_response(new_user_input, temp, top_p, top_k, presence_penalty, frequency_penalty, max_tokens, seed, clear)

    def clear_history(self):
        print("--- Clearing conversation history ---")
        self.history = []

    def print_history(self):
        print("\n--- Conversation History ---")
        if not self.history:
            print("History is empty.")
            return
        for i, entry in enumerate(self.history):
            print(f"{i+1}. Role: {entry['role']}, Content: {entry['content'][:100]}...")
        print("--------------------------\n")

    def get_lastest(self):
        if not self.history:
            print("History is empty.")
            return ''
        return self.history[-1]['content']

    def add_history(self, user_input, prompt_output):
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": prompt_output})
