import asyncio
import json
import sys
import os
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
import mcp.types
from mcp.client.stdio import stdio_client

from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool as GeminiTool
try:
    from google.generativeai.types import GenerationConfig
except ImportError:
    GenerationConfig = None 

load_dotenv()

def _mcp_schema_to_gemini_schema(mcp_schema: Dict[str, Any], is_root_schema: bool = True) -> Dict[str, Any]:
    if not isinstance(mcp_schema, dict):
        return mcp_schema
    
    gemini_schema = {}
    original_properties_for_root = mcp_schema.get("properties", {}) if is_root_schema else {}

    for key, value in mcp_schema.items():
        if key == "type" and isinstance(value, str):
            gemini_schema[key] = value.lower()
        elif key == "properties" and isinstance(value, dict):
            gemini_schema["properties"] = {
                prop_name: _mcp_schema_to_gemini_schema(prop_value, is_root_schema=False)
                for prop_name, prop_value in value.items()
            }
        elif key == "items" and isinstance(value, dict):
             gemini_schema["items"] = _mcp_schema_to_gemini_schema(value, is_root_schema=False)
        elif key == "required":
            if not is_root_schema:
                 gemini_schema[key] = value 
        elif key == "default":
            if not is_root_schema and "type" in mcp_schema: 
                continue 
            elif is_root_schema: 
                continue
            else: 
                gemini_schema[key] = value

        elif isinstance(value, dict): 
            gemini_schema[key] = _mcp_schema_to_gemini_schema(value, is_root_schema=False)
        elif isinstance(value, list):
            gemini_schema[key] = [
                _mcp_schema_to_gemini_schema(item, is_root_schema=False) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            gemini_schema[key] = value

    if is_root_schema and "properties" in gemini_schema: 
        gemini_required = []
        for prop_name, prop_details_mcp in original_properties_for_root.items():
            if isinstance(prop_details_mcp, dict) and "default" not in prop_details_mcp:
                gemini_required.append(prop_name)
        
        if "n_total_examples" in gemini_schema["properties"] and "n_total_examples" not in gemini_required:
            gemini_required.append("n_total_examples")
        if "m_examples_per_string" in gemini_schema["properties"] and "m_examples_per_string" not in gemini_required:
            gemini_required.append("m_examples_per_string")

        if gemini_required:
            gemini_schema["required"] = list(set(gemini_required))
        elif "required" in gemini_schema: 
            del gemini_schema["required"] 

    return gemini_schema


class CompetitionMCPClient:
    def __init__(self, gemini_model_name: str = "gemini-2.5-flash"):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.stdio_reader = None
        self.stdio_writer = None
        
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        genai.configure(api_key=self.gemini_api_key)
        self.gemini_model = genai.GenerativeModel(gemini_model_name)
        self.conversation_history: List[Any] = [] 
        self.mcp_tools_cache: Optional[List[mcp.types.Tool]] = None

    async def connect_to_server(self, server_script_path: str):
        if not server_script_path.endswith('.py'):
            raise ValueError("Server script must be a .py file for this client.")
        server_params = StdioServerParameters(
            command="python", args=["-u", server_script_path], env=None)
        stdio_transport_cm = stdio_client(server_params)
        self.stdio_reader, self.stdio_writer = await self.exit_stack.enter_async_context(stdio_transport_cm)
        session_cm = ClientSession(self.stdio_reader, self.stdio_writer)
        self.session = await self.exit_stack.enter_async_context(session_cm)
        await self.session.initialize()
        list_tools_response = await self.session.list_tools()
        self.mcp_tools_cache = list_tools_response.tools
        print("\nSuccessfully connected to MCP server.")
        print("Available tools:")
        if self.mcp_tools_cache:
            for tool in self.mcp_tools_cache:
                print(f"- {tool.name}: {tool.description}")
        else:
            print("No tools reported by the server!")
        print("\nAgentic AI is ready. You can now type your queries.")

    def _get_gemini_tools(self) -> Optional[List[GeminiTool]]:
        if not self.mcp_tools_cache: return None
        gemini_function_declarations = []
        for mcp_tool in self.mcp_tools_cache:
            gemini_params_schema = _mcp_schema_to_gemini_schema(mcp_tool.inputSchema)
            
            try:
                effective_params = None
                if gemini_params_schema and gemini_params_schema.get("properties"):
                    effective_params = gemini_params_schema
                elif gemini_params_schema and not gemini_params_schema.get("properties") and gemini_params_schema.get("type") == "OBJECT":
                     # If it's an object type but no properties, treat as no parameters for Gemini
                    effective_params = None


                fd = FunctionDeclaration(
                    name=mcp_tool.name,
                    description=mcp_tool.description,
                    parameters=effective_params)
                gemini_function_declarations.append(fd)
            except Exception as e:
                print(f"Warning: Could not create FunctionDeclaration for tool '{mcp_tool.name}': {e}")
                print(f"Schema that caused error: {json.dumps(gemini_params_schema, indent=2)}")
        if not gemini_function_declarations: return None
        return [GeminiTool(function_declarations=gemini_function_declarations)]

    async def _print_tool_response(self, response: Any, tool_name: str):
        print(f"\n--- MCP Response from '{tool_name}' ---")
        if response is None:
            print("ERROR: Received None response object from server.")
            return
        if hasattr(response, 'error') and response.error:
            error_obj = response.error
            print(f"TOOL CALL FAILED (MCP Error): Code: {getattr(error_obj, 'code', 'N/A')}, Message: {getattr(error_obj, 'message', str(error_obj))}")
            if hasattr(error_obj, 'data') and error_obj.data: print(f"  Error Data: {error_obj.data}")
            return
        if hasattr(response, 'content') and response.content:
            for item in response.content:
                item_type = getattr(item, 'type', None)
                if item_type == "text" and hasattr(item, 'text'): print(item.text)
                elif item_type == "error" and hasattr(item, 'text'): print(f"SERVER-REPORTED ERROR: {item.text}")
                elif item_type: print(f"Content type: {item_type}, Data: {str(item)[:200]}...")
                else: print(f"Content item without type: {str(item)[:200]}...")
        elif hasattr(response, 'isError') and response.isError:
            print("Tool call indicated error (isError=True), but no standard error object or content.")
            print(f"Raw response: {str(response)[:500]}...")
        else:
            print("(Tool successful, but no content items returned or content not in expected format.)")
        print(f"--- End of MCP response from '{tool_name}' ---")

    def _format_mcp_response_for_gemini(self, mcp_response: Any, tool_name: str) -> str:
        if mcp_response is None: return f"Tool '{tool_name}' execution resulted in None response."
        if hasattr(mcp_response, 'error') and mcp_response.error:
            err = mcp_response.error
            return f"Tool '{tool_name}' failed with MCP Error Code {getattr(err, 'code', 'N/A')}: {getattr(err, 'message', str(err))}"
        texts = []
        if hasattr(mcp_response, 'content') and mcp_response.content:
            for item in mcp_response.content:
                if getattr(item, 'type', None) == "text" and hasattr(item, 'text'): texts.append(item.text)
                elif getattr(item, 'type', None) == "error" and hasattr(item, 'text'): texts.append(f"Server-reported error: {item.text}")
        if not texts:
            if hasattr(mcp_response, 'isError') and mcp_response.isError: return f"Tool '{tool_name}' indicated error (isError=True) but no specific content."
            return f"Tool '{tool_name}' executed, but returned no textual content."
        return "\n".join(texts)

    def _convert_gemini_args_to_dict(self, data: Any) -> Any:
        if hasattr(data, 'keys'):  
            return {key: self._convert_gemini_args_to_dict(data[key]) for key in data.keys()}
        if hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
            return [self._convert_gemini_args_to_dict(item) for item in data]
        return data

    async def process_query_with_gemini(self, user_query: str):
        if not self.session:
            print("Error: Not connected.")
            return
        print(f"\nUser Query: {user_query}")
        self.conversation_history.append({'role': 'user', 'parts': [{'text': user_query}]})
        
        gemini_tools = self._get_gemini_tools()
        if not gemini_tools:
            print("Warning: No tools available for Gemini. Proceeding with text generation only.")

        generation_args = {}
        if GenerationConfig: 
            generation_args['generation_config'] = GenerationConfig(temperature=0.0)
        
        while True:
            try:
                print("Asking AI...")
                gemini_response_obj = self.gemini_model.generate_content(
                    self.conversation_history,
                    tools=gemini_tools,
                    **generation_args 
                )
                
                if not hasattr(gemini_response_obj, 'candidates') or not gemini_response_obj.candidates:
                    print("Gemini response has no candidates.")
                    self.conversation_history.append({'role': 'model', 'parts': [{'text': "Error: No candidates in response."}]})
                    break
                
                model_content = gemini_response_obj.candidates[0].content
                if not model_content:
                    print("Gemini response candidate has no content.")
                    self.conversation_history.append({'role': 'model', 'parts': [{'text': "Error: No content in candidate."}]})
                    break
                self.conversation_history.append(model_content) 

                if not hasattr(model_content, 'parts') or not model_content.parts:
                    print("Gemini response content has no parts.")
                    self.conversation_history.append({'role': 'model', 'parts': [{'text': "Error: No parts in content."}]})
                    break
                first_part_obj = model_content.parts[0]

            except Exception as e:
                print(f"Error calling Gemini API or processing its response structure: {e}")
                import traceback; traceback.print_exc()
                self.conversation_history.append({'role': 'model', 'parts': [{'text': f"Error communicating with LLM: {e}"}]})
                break 

            function_call_details = None
            text_content = None
            try:
                if hasattr(first_part_obj, 'function_call'):
                    function_call_details = first_part_obj.function_call
            except Exception as e_fc:
                print(f"Note: Could not access first_part_obj.function_call directly: {e_fc}")
            if not function_call_details:
                try:
                    if hasattr(first_part_obj, 'text'):
                        text_content = first_part_obj.text
                except Exception as e_txt:
                     print(f"Note: Could not access first_part_obj.text directly: {e_txt}")

            if function_call_details:
                tool_name = getattr(function_call_details, 'name', None)
                tool_args_struct = getattr(function_call_details, 'args', None)
                tool_args = self._convert_gemini_args_to_dict(tool_args_struct) if tool_args_struct else {}
                
                if not tool_name:
                    print("Error: Gemini function call has no name.")
                    self.conversation_history.append({'role': 'model', 'parts': [{'text': "Error: Malformed function call (no name)."}]})
                    break
                print(f"\nAgentic AI wants to call MCP tool: '{tool_name}'")
                print(f"Arguments: {json.dumps(tool_args, indent=2)}")
                try:
                    mcp_tool_response = await self.session.call_tool(tool_name, tool_args)
                    await self._print_tool_response(mcp_tool_response, tool_name)
                    tool_output_for_gemini = self._format_mcp_response_for_gemini(mcp_tool_response, tool_name)
                    self.conversation_history.append({
                        'role': 'tool',
                        'parts': [{'function_response': {'name': tool_name, 'response': {"content": tool_output_for_gemini}}}]
                    })
                except Exception as e:
                    print(f"Error calling MCP tool '{tool_name}': {e}")
                    self.conversation_history.append({
                        'role': 'tool',
                        'parts': [{'function_response': {'name': tool_name, 'response': {"error": f"Failed to execute tool {tool_name}: {str(e)}" }}}]
                    })
            elif text_content:
                print(f"\nGemini: {text_content}")
                break 
            else: 
                print("Gemini response was neither a recognizable tool call nor text.")
                self.conversation_history.append({'role': 'model', 'parts': [{'text': "Error: Unrecognized response format."}]})
                break

    async def chat_loop(self):
        if not self.session:
            print("Error: Not connected. Call connect_to_server first.")
            return
        print("\n--- Interactive Chat with Competition Manager ---")
        print("Type requests (e.g., 'start competition with 10 examples').")
        print("Type 'reset' to clear Agentic AI history, 'quit' or 'exit' to end.")
        while True:
            try:
                user_input = await asyncio.to_thread(input, "\nChat: ")
                user_input = user_input.strip()
                if user_input.lower() in ["quit", "exit"]: break
                if user_input.lower() == "reset":
                    self.conversation_history = []
                    print("Agentic AI conversation history reset.")
                    continue
                if not user_input: continue
                await self.process_query_with_gemini(user_input)
            except KeyboardInterrupt: print("\nExiting chat loop (KeyboardInterrupt)."); break
            except Exception as e:
                print(f"\nError in chat loop: {e}")
                import traceback; traceback.print_exc()

    async def cleanup(self):
        print("\nCleaning up client resources...")
        if self.exit_stack:
            print("Closing resources via AsyncExitStack...")
            await self.exit_stack.aclose()
            print("Client resources closed.")
        print("Client cleanup complete.")

async def main_workflow():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path_to_server_script.py>")
        sys.exit(1)
    server_script = sys.argv[1]
    client = None
    try:
        client = CompetitionMCPClient()
        await client.connect_to_server(server_script)
        await client.chat_loop()
    except ConnectionError as ce: print(f"Connection Error: {ce}.")
    except ValueError as ve: print(f"Configuration Error: {ve}")
    except Exception as e:
        print(f"Unexpected error in main_workflow: {e}")
        import traceback; traceback.print_exc()
    finally:
        if client: await client.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main_workflow())
    except KeyboardInterrupt:
        print("\nClient shutdown (Ctrl+C).")