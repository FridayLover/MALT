import asyncio
import json
import os
import pandas as pd
import random
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
from mcp.server import Server
from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource
import mcp.types as types
import functools
import shutil
import re

from gpt_oss_np import GPTChatbot
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

class AlgorithmCompetitionMCP:
    def __init__(self):
        self.server = Server("algorithm-competition")
        self.current_round = 0
        self.survivors: List[str] = []
        self.losers_history: List[List[str]] = []
        self.competition_data: Dict[str, Any] = {}
        self.default_max_rounds = 20
        self.default_evolution_strategy = "tournament"
        self.default_ranking_rule_params: Dict[str, Any] = {}
        self.default_rand_mode = "fully_random"

        # State for Tournament Strategy
        self.elo_ratings: Dict[str, float] = {}
        self.default_elo = 1200
        self.base_elo_k_factor = 32
        self.initial_contestant_data: Dict[str, List[Dict]] = {}
        self.contestant_ancestry: Dict[str, Set[str]] = {}
        
        self.setup_tools()
        random.seed(42)

        try:
            gemini_api_key = os.environ.get("GEMINI_API_KEY")
            if not gemini_api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set.")
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        except Exception as e:
            print(f"Error configuring Gemini: {e}")
            self.gemini_model = None


    def setup_tools(self):
        """Setup MCP tools for the competition workflow"""
        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            return [
                Tool(
                    name="start_competition",
                    description="Start a new algorithm competition with specified parameters and run round 1.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "n_total_examples": {"type": "number", "description": "Target total number of examples for the initial prompt pool (float allowed, will be rounded)."},
                            "m_contestants": {"type": "integer", "description": "Target number of contestants (algorithms) to generate."},
                            "metadata_path": {"type": "string", "description": "Path to metadata Excel file", "default": "../dataset/rawdata.xlsx"},
                            "data_dir": {"type": "string", "description": "Path to raw data directory", "default": "../dataset/rawdata/"},
                            "output_dir": {"type": "string", "description": "Output directory for results", "default": "./competition_results"},
                            "evolution_strategy": {
                                "type": "string",
                                "description": "Strategy for algorithm revision. Only 'tournament' is supported.",
                                "enum": ["tournament"],
                                "default": self.default_evolution_strategy
                            },
                            "rand_mode": {
                                "type": "string",
                                "description": "Randomization mode for selecting examples. 'fully_random' or 'high_variability'.",
                                "enum": ["fully_random", "high_variability"],
                                "default": self.default_rand_mode
                            },
                            "ranking_rule_params": {
                                "type": "object",
                                "description": "Parameters for the ranking rule (e.g., {'percentage_to_eliminate': 0.5}).",
                                "default": self.default_ranking_rule_params
                            },
                            "prompt_mode": {
                                "type": "string",
                                "description": "Controls info in revision prompts. 'default': loser's data only. 'add_best_step': also includes the best performer's logic from the last round. 'add_all_contestants_step': includes logic from all other rivals.",
                                "enum": ["default", "add_best_step", "add_all_contestants_step"],
                                "default": "default"
                            },
                            "train_data": {"type": "string", "description": "Optional path to a CSV file with training data. If provided, n_total_examples is ignored."},
                            "evaluation_data": {"type": "string", "description": "Optional path to a CSV file with evaluation data. Must be provided if train_data is."},
                            "test_data": {"type": "string", "description": "Optional path to a CSV file with test data for final evaluation."},
                            "metric_mode": {
                                "type": "string",
                                "description": "Metric used for ranking to eliminate competitors. Both MAE and RMSE are always reported in CSVs.",
                                "enum": ["MAE", "RMSE"],
                                "default": "MAE"
                            },
                            "evaluate_mode": {
                                "type": "string",
                                "description": "Evaluation mode for the 'tournament' strategy. 'normal' uses the global evaluation set. 'pair' uses the combined training data of the two contestants in a match.",
                                "enum": ["normal", "pair"],
                                "default": "normal"
                            }
                        },
                        "required": ["n_total_examples", "m_contestants"]
                    }
                ),
                Tool(
                    name="get_competition_status",
                    description="Get current competition status",
                    inputSchema={"type": "object", "properties": {}}
                ),
                Tool(
                    name="advance_round",
                    description="Advance to the next round of competition using the configured evolution strategy.",
                    inputSchema={"type": "object", "properties": {}}
                ),
                Tool(
                    name="get_results",
                    description="Get final competition results",
                    inputSchema={"type": "object", "properties": {}}
                ),
                Tool(
                    name="full_competition",
                    description="Run a full algorithm competition: start, advance rounds until one survivor or max_rounds, then get results.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "n_total_examples": {"type": "number", "description": "Target total number of examples for the initial prompt pool."},
                            "m_contestants": {"type": "integer", "description": "Target number of contestants (algorithms) to generate."},
                            "metadata_path": {"type": "string", "description": "Path to metadata Excel file", "default": "../dataset/rawdata.xlsx"},
                            "data_dir": {"type": "string", "description": "Path to raw data directory", "default": "../dataset/rawdata/"},
                            "output_dir": {"type": "string", "description": "Output directory for results", "default": "./competition_results"},
                            "max_rounds": {"type": "integer", "description": "Maximum number of total rounds to run", "default": self.default_max_rounds},
                            "evolution_strategy": {
                                "type": "string",
                                "description": "Strategy for algorithm revision. Only 'tournament' is supported.",
                                "enum": ["tournament"],
                                "default": self.default_evolution_strategy
                            },
                            "rand_mode": {
                                "type": "string",
                                "description": "Randomization mode for selecting examples.",
                                "enum": ["fully_random", "high_variability"],
                                "default": self.default_rand_mode
                            },
                            "ranking_rule_params": {
                                "type": "object",
                                "description": "Parameters for the ranking rule.",
                                "default": self.default_ranking_rule_params
                            },
                            "prompt_mode": {
                                "type": "string",
                                "description": "Controls info in revision prompts. 'default': loser's data only. 'add_best_step': also includes the best performer's logic from the last round. 'add_all_contestants_step': includes logic from all other rivals.",
                                "enum": ["default", "add_best_step", "add_all_contestants_step"],
                                "default": "default"
                            },
                            "train_data": {"type": "string", "description": "Optional path to a CSV file with training data. If provided, n_total_examples is ignored."},
                            "evaluation_data": {"type": "string", "description": "Optional path to a CSV file with evaluation data. Must be provided if train_data is."},
                            "test_data": {"type": "string", "description": "Optional path to a CSV file with test data for final evaluation."},
                            "metric_mode": {
                                "type": "string",
                                "description": "Metric used for ranking to eliminate competitors. Both MAE and RMSE are always reported in CSVs.",
                                "enum": ["MAE", "RMSE"],
                                "default": "MAE"
                            },
                            "evaluate_mode": {
                                "type": "string",
                                "description": "Evaluation mode for the 'tournament' strategy. 'normal' uses the global evaluation set. 'pair' uses the combined training data of the two contestants in a match.",
                                "enum": ["normal", "pair"],
                                "default": "normal"
                            }
                        },
                        "required": ["n_total_examples", "m_contestants"]
                    }
                ),
                Tool(
                    name="run_multiple_competitions",
                    description="Run multiple full competitions with specified configurations.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "competitions": {
                                "type": "array",
                                "description": "A list of configurations, where each item is a dictionary of parameters for a single full_competition.",
                                "items": {
                                    "type": "object"
                                }
                            }
                        },
                        "required": ["competitions"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
            if name == "start_competition":
                return await self.start_competition(**arguments)
            elif name == "get_competition_status":
                return await self.get_competition_status()
            elif name == "advance_round":
                return await self._advance_round_logic()
            elif name == "get_results":
                return await self.get_results()
            elif name == "full_competition":
                return await self.run_full_competition_logic(**arguments)
            elif name == "run_multiple_competitions":
                return await self.run_multiple_competitions(**arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

    def _calculate_dynamic_k_factor(self, winner_score: float, loser_score: float) -> float:
        if math.isinf(loser_score) and not math.isinf(winner_score):
            return self.base_elo_k_factor * 2.0
        
        if math.isinf(winner_score) or winner_score >= loser_score:
            return self.base_elo_k_factor
        
        relative_diff = (loser_score - winner_score) / (loser_score + 1e-9)
        
        if relative_diff > 0.5:
            return self.base_elo_k_factor * 1.75
        elif relative_diff > 0.2:
            return self.base_elo_k_factor * 1.25
        elif relative_diff > 0.05:
            return self.base_elo_k_factor
        else:
            return self.base_elo_k_factor * 0.75

    def _get_elo(self, contestant_id: str) -> float:
        return self.elo_ratings.get(str(contestant_id), self.default_elo)

    def _update_elo(self, winner_id: str, loser_id: str, winner_score: float, loser_score: float) -> Tuple[float, float]:
        r_winner = self._get_elo(winner_id)
        r_loser = self._get_elo(loser_id)

        k_factor = self._calculate_dynamic_k_factor(winner_score, loser_score)
        
        e_winner = 1 / (1 + 10**((r_loser - r_winner) / 400))
        e_loser = 1 / (1 + 10**((r_winner - r_loser) / 400))

        new_r_winner = r_winner + k_factor * (1 - e_winner)
        new_r_loser = r_loser + k_factor * (0 - e_loser)

        self.elo_ratings[str(winner_id)] = new_r_winner
        self.elo_ratings[str(loser_id)] = new_r_loser
        
        return new_r_winner, new_r_loser

    async def start_competition(self, n_total_examples: float, m_contestants: float,
                                metadata_path: str = "../dataset/rawdata.xlsx",
                                data_dir: str = "../dataset/rawdata/",
                                output_dir: str = "./competition_results",
                                evolution_strategy: str = "tournament",
                                rand_mode: str = "fully_random",
                                ranking_rule_params: Optional[Dict[str, Any]] = None,
                                prompt_mode: str = "default",
                                train_data: Optional[str] = None,
                                evaluation_data: Optional[str] = None,
                                test_data: Optional[str] = None,
                                metric_mode: str = "MAE",
                                evaluate_mode: str = "normal"
                               ) -> list[TextContent]:
        try:
            if not self.gemini_model:
                 return [TextContent(type="text", text="Error: Gemini model not initialized. Check API key.")]

            self.current_round = 0
            self.survivors = []
            self.losers_history = []
            self.competition_data = {}
            self.elo_ratings = {}
            self.initial_contestant_data = {}
            self.contestant_ancestry = {}

            m_contestants = int(round(m_contestants))

            if evolution_strategy != "tournament":
                msg = f"Error: Invalid evolution_strategy '{evolution_strategy}'. This version only supports 'tournament'."
                print(msg)
                return [TextContent(type="text", text=msg)]
            
            parsed_revision_mode = "tournament"
            parsed_ranking_rule = "tournament_cutoff"

            train_samples_df = None
            eval_samples_list = []
            if train_data and evaluation_data:
                print(f"Info: Using provided training data '{train_data}' and evaluation data '{evaluation_data}'.")
                try:
                    train_samples_df = pd.read_csv(train_data)
                    eval_df = pd.read_csv(evaluation_data)
                    eval_samples_list = eval_df.to_dict('records')
                    n_total_examples = len(train_samples_df)
                    print(f"Info: Loaded {n_total_examples} training samples and {len(eval_samples_list)} evaluation samples.")
                except FileNotFoundError as e:
                    msg = f"Error: Could not find provided data files: {e}"
                    print(msg)
                    return [TextContent(type="text", text=msg)]
            
            if m_contestants < 1:
                msg = f"Error: m_contestants ({m_contestants}) must be at least 1."
                print(msg)
                return [TextContent(type="text", text=msg)]
            
            if m_contestants & (m_contestants - 1) != 0:
                msg = f"Error: For 'tournament' strategy, m_contestants must be a power of 2 (e.g., 2, 4, 8, 16). Got {m_contestants}."
                print(msg)
                return [TextContent(type="text", text=msg)]

            if n_total_examples < 0.5:
                msg = f"Error: n_total_examples ({n_total_examples}) is too small, must be at least 0.5 to round to 1 or more."
                print(msg)
                return [TextContent(type="text", text=msg)]
            n_rounded_total_examples = int(round(n_total_examples))
            print(f"Info: n_total_examples (requested: {n_total_examples}) processed to initial rounded target: {n_rounded_total_examples}")

            if n_rounded_total_examples < m_contestants:
                message = (f"Error: Rounded n_total_examples ({n_rounded_total_examples}) "
                           f"is less than m_contestants ({m_contestants}). "
                           f"Not enough examples to give at least one to each contestant.")
                print(message)
                return [TextContent(type="text", text=message)]
            
            self.competition_data = {
                "n_total_examples_requested": n_total_examples,
                "m_contestants_requested": m_contestants,
                "metadata_path": metadata_path,
                "data_dir": data_dir,
                "output_dir": output_dir,
                "evolution_strategy": evolution_strategy,
                "revision_mode": parsed_revision_mode,
                "ranking_rule": parsed_ranking_rule,
                "rand_mode": rand_mode,
                "ranking_rule_params": ranking_rule_params if ranking_rule_params is not None else {},
                "prompt_mode": prompt_mode,
                "evaluation_samples": eval_samples_list,
                "metric_mode": metric_mode,
                "train_data_path": train_data,
                "evaluation_data_path": evaluation_data,
                "test_data_path": test_data,
                "evaluate_mode": evaluate_mode,
            }

            os.makedirs(output_dir, exist_ok=True)

            round1_prompt_data_dir = f"{output_dir}/round1"

            generated_prompts, initial_data_assignments = self.create_prompt_strings(
                metadata_path=metadata_path,
                data_folder_path=data_dir,
                rand_csv_path=round1_prompt_data_dir,
                n_total_examples_target=n_total_examples,
                num_contestants=m_contestants,
                rand_mode=rand_mode,
                train_samples_df=train_samples_df
            )
            self.initial_contestant_data = initial_data_assignments

            if not generated_prompts:
                return [TextContent(type="text", text="Error: No prompts could be generated. Check data and parameters.")]

            actual_num_contestants = len(generated_prompts)
            self.competition_data["num_contestants_initial"] = actual_num_contestants
            
            self.competition_data["n_total_examples_used_for_prompts"] = n_rounded_total_examples

            self.current_round = 1
            await self.initialize_round1(generated_prompts)
            
            initial_contestant_ids = [str(i + 1) for i in range(self.competition_data["num_contestants_initial"])]
            
            self.survivors = initial_contestant_ids
            self.elo_ratings = {cid: self.default_elo for cid in initial_contestant_ids}
            self.contestant_ancestry = {cid: {cid} for cid in initial_contestant_ids}
            print(f"Initialized Elo ratings for {len(self.elo_ratings)} contestants to {self.default_elo}.")

            print("\n--- Running Tournament Round 1 ---")
            round1_winners, round1_losers, round1_results_summary = await self._run_tournament_round()
            
            self.survivors = round1_winners
            self.losers_history.append(round1_losers)
            
            message = (
                f"Competition started in Tournament mode! Round 1 completed.\n"
                f"- Initial contestants: {self.competition_data['num_contestants_initial']}.\n"
                f"- Winners (advancing to Round 2): {len(self.survivors)} ({self.survivors})\n"
                f"- Losers of Round 1: {len(round1_losers)} ({round1_losers})\n"
                f"- Current round: {self.current_round}\n"
                f"Results saved to: {output_dir}/round{self.current_round}"
            )
            
            if len(self.survivors) > 1:
                print(f"\n--- Preparing Winners of Round 1 for Round 2 ---")
                await self._prepare_next_tournament_round(round1_results_summary)
                message += f"\n- Winners have been revised for Round 2."

            return [TextContent(type="text", text=message)]

        except Exception as e:
            import traceback
            print(f"Error in start_competition: {e}\n{traceback.format_exc()}")
            return [TextContent(type="text", text=f"Error starting competition: {str(e)}")]

    def create_prompt_strings(self, metadata_path: str, data_folder_path: str,
                            rand_csv_path: str, n_total_examples_target: float,
                            num_contestants: int, id_col: str = 'ID',
                            score_col: str = 'VA ETDRS Score',
                            rand_mode: str = "fully_random",
                            train_samples_df: Optional[pd.DataFrame] = None) -> Tuple[List[str], Dict[str, List[Dict]]]:
        HEADER = "stimulus (up = 0, left = 1, down = 2, right = 3)  side (right eye=0, left eye = 1)  size rgb response (up = 0, left = 1, down = 2, right = 3)  reaction_time cumulative_time is_correct"

        if num_contestants <= 0:
             print(f"Error in create_prompt_strings: num_contestants must be positive, got {num_contestants}.")
             return [], {}
        
        if n_total_examples_target < 0:
            print(f"Error in create_prompt_strings: n_total_examples_target ({n_total_examples_target}) cannot be negative.")
            return [], {}
            
        rounded_n_target = int(round(n_total_examples_target))
        print(f"Info (create_prompt_strings): n_total_examples_target (float: {n_total_examples_target}) rounded to int: {rounded_n_target}.")

        if rounded_n_target < num_contestants:
            print(f"Error in create_prompt_strings: n_total_examples ({rounded_n_target}) is less than num_contestants ({num_contestants}). Cannot create prompts.")
            return [], {}

        all_samples_from_source = []
        if train_samples_df is not None:
            print("Info: Using pre-loaded DataFrame for prompt creation.")
            all_samples_from_source = train_samples_df.to_dict('records')
        else:
            try:
                metadata_df = pd.read_excel(metadata_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

            for _, row in metadata_df.iterrows():
                sample_id = row[id_col]
                score = row[score_col]
                csv_path = os.path.join(data_folder_path, f"{sample_id}.csv")
                if os.path.exists(csv_path):
                     all_samples_from_source.append({'id': sample_id, 'score': score, 'path': csv_path, 'original_index': len(all_samples_from_source)})
                else:
                    print(f"Warning: Data file not found for ID {sample_id} at {csv_path}")

        if not all_samples_from_source:
            raise ValueError("No valid samples found from source. Check data_folder_path and metadata content.")

        if rounded_n_target > len(all_samples_from_source):
            error_msg = (f"Error: Requested n_total_examples ({n_total_examples_target}, rounded to {rounded_n_target}) "
                         f"exceeds the total number of available unique samples ({len(all_samples_from_source)}). "
                         f"Please request {len(all_samples_from_source)} or fewer examples.")
            print(error_msg)
            return [], {}

        actual_n_for_pool = rounded_n_target

        overall_selected_samples: List[Dict] = []
        if train_samples_df is None:
            if rand_mode == "high_variability":
                print(f"Using 'high_variability' rand_mode for selecting the overall pool of {actual_n_for_pool} examples.")
                sorted_all_samples = sorted(all_samples_from_source, key=lambda s: s['score'])
                
                if actual_n_for_pool == len(sorted_all_samples):
                    overall_selected_samples = list(sorted_all_samples)
                else:
                    indices = np.linspace(0, len(sorted_all_samples) - 1, actual_n_for_pool, dtype=int)
                    overall_selected_samples = [sorted_all_samples[i] for i in np.unique(indices)]
                    if len(overall_selected_samples) < actual_n_for_pool:
                        print(f"Warning (high_variability): Linspace selection yielded {len(overall_selected_samples)} unique items, expected {actual_n_for_pool}. Filling with random samples.")
                        remaining_needed = actual_n_for_pool - len(overall_selected_samples)
                        existing_ids = {s['id'] for s in overall_selected_samples}
                        potential_fillers = [s for s in all_samples_from_source if s['id'] not in existing_ids]
                        if len(potential_fillers) >= remaining_needed:
                            overall_selected_samples.extend(random.sample(potential_fillers, remaining_needed))
                
                # Note: For high_variability, we usually want to keep them sorted or structured to ensure spread,
                # but the distribution logic below handles the spread.
            else:
                print(f"Using 'fully_random' rand_mode for selecting the overall pool of {actual_n_for_pool} examples.")
                overall_selected_samples = random.sample(all_samples_from_source, actual_n_for_pool)

            prompt_sample_ids = {s['id'] for s in overall_selected_samples}
            evaluation_samples = [s for s in all_samples_from_source if s['id'] not in prompt_sample_ids]
            self.competition_data['evaluation_samples'] = evaluation_samples
            
            output_path = Path(self.competition_data['output_dir'])
            train_df_path = output_path / "training_data_split.csv"
            eval_df_path = output_path / "evaluation_data_split.csv"
            pd.DataFrame.from_dict(overall_selected_samples).to_csv(train_df_path, index=False)
            pd.DataFrame.from_dict(evaluation_samples).to_csv(eval_df_path, index=False)
            print(f"Info: Split data into {len(overall_selected_samples)} prompt samples and {len(evaluation_samples)} evaluation samples.")
        else:
            overall_selected_samples = all_samples_from_source
            if rand_mode != "high_variability":
                random.shuffle(overall_selected_samples)

        os.makedirs(rand_csv_path, exist_ok=True)
        self.competition_data['prompt_samples_for_round1'] = overall_selected_samples
        pd.DataFrame.from_dict(overall_selected_samples).to_csv(f"{rand_csv_path}/rand_data_for_prompts_round1_pool.csv", index=False)

        grouped_samples_for_prompts: List[List[Dict]] = [[] for _ in range(num_contestants)]

        if rand_mode == "high_variability":
            print(f"Distributing overall pool to prompts using 'high_variability' logic (score-spread per prompt).")
            samples_to_distribute_to_prompts = sorted(list(overall_selected_samples), key=lambda s: s['score'])
            for idx, sample in enumerate(samples_to_distribute_to_prompts):
                grouped_samples_for_prompts[idx % num_contestants].append(sample)
        else: 
            print(f"Distributing overall pool to prompts using 'fully_random' logic with random remainder distribution.")
            base_count = actual_n_for_pool // num_contestants
            remainder = actual_n_for_pool % num_contestants
            
            chunk_sizes = [base_count + 1] * remainder + [base_count] * (num_contestants - remainder)
            
            random.shuffle(chunk_sizes)
            
            temp_distribute_list = list(overall_selected_samples) 
            random.shuffle(temp_distribute_list) 
            
            current_idx = 0
            for i, size in enumerate(chunk_sizes):
                grouped_samples_for_prompts[i] = temp_distribute_list[current_idx : current_idx + size]
                current_idx += size

        final_output_list = []
        final_assignments: Dict[str, List[Dict]] = {}
        prompt_idx_counter = 1

        for i_prompt, chunk_of_samples in enumerate(grouped_samples_for_prompts):
            if not chunk_of_samples:
                print(f"Warning: Prompt {i_prompt+1} has 0 samples. Skipping this prompt.")
                continue 

            parts_for_current_string = []
            local_example_index = 0
            for sample in chunk_of_samples:
                try:
                    raw_df = pd.read_csv(sample['path'], skiprows=1)
                    if raw_df.shape[1] < 8:
                        print(f"Warning: File for ID {sample['id']} ({sample['path']}) has {raw_df.shape[1]} columns, less than the 8 required. Skipping.")
                        continue
                    sliced_df = raw_df.iloc[:, :8]

                    data_as_string = sliced_df.to_string(index=False, header=False)
                    score = sample['score']
                    snellen_score = self.etdrs_to_snellen(score)
                    example_block = f"""<example_data_{local_example_index}>
{HEADER}
{data_as_string}
</example_data_{local_example_index}>
The ground truth Snellen score for example {local_example_index} is {snellen_score}.
"""
                    parts_for_current_string.append(example_block)
                    local_example_index += 1
                except Exception as e:
                    print(f"Warning: Could not process file for ID {sample['id']} ({sample.get('path', 'N/A')}) during prompt creation: {e}")

            if len(parts_for_current_string) > 0: 
                full_string = "\n".join(parts_for_current_string)
                final_output_list.append(full_string)
                contestant_id = str(prompt_idx_counter)
                final_assignments[contestant_id] = chunk_of_samples
                prompt_idx_counter += 1
            else:
                print(f"Warning: Prompt {i_prompt+1} ended up with 0 valid examples. Discarding.")

        if not final_output_list:
            if actual_n_for_pool > 0 :
                print("Warning: Could not generate any valid prompt strings after processing samples. Check data integrity and parameters.")
            return [], {}
        
        return final_output_list, final_assignments

    async def initialize_round1(self, generated_prompts: List[str]):
        output_dir = self.competition_data["output_dir"]
        round_dir = f"{output_dir}/round{self.current_round}"

        for subdir in ["prompts", "outputs", "data", "algorithms"]:
            os.makedirs(f"{round_dir}/{subdir}", exist_ok=True)

        print(f"Processing {len(generated_prompts)} contestants for Round {self.current_round}...")

        for i, prompt_content in enumerate(generated_prompts):
            contestant_num = i + 1
            print(f"  Initializing Contestant {contestant_num} for Round 1...")

            full_prompt_for_gpt = self.create_full_prompt(prompt_content)

            prompt_file_r1 = f"{round_dir}/prompts/{contestant_num}_gpt_prompt_round1.txt"
            with open(prompt_file_r1, 'w', encoding='utf-8') as f:
                f.write(full_prompt_for_gpt)

            data_file_r1 = f"{round_dir}/data/{contestant_num}_data.txt"
            with open(data_file_r1, 'w', encoding='utf-8') as f:
                f.write(prompt_content) 

            gpt_chatbot = GPTChatbot()
            gpt_response_text = None
            timeout = 7200.0
            loop = asyncio.get_running_loop()

            print(f"  Running initial GPT generation for contestant {contestant_num}...")
            try:
                gpt_response_text = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        functools.partial(gpt_chatbot.generate_response, full_prompt_for_gpt, temp=0.0, top_p=1.0, top_k=100, clear=True)
                    ),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                print(f"  Initial GPT generation for {contestant_num} timed out.")
                gpt_response_text = None
            except Exception as e:
                print(f"  Initial GPT generation for {contestant_num} failed with error: {e}")
                gpt_response_text = None

            retries = 0
            max_retries = 10
            while not gpt_response_text and retries < max_retries:
                retries += 1
                print(f"  Retrying GPT for contestant {contestant_num} (R1)... (Attempt {retries}/{max_retries})")

                presence_penalty = 0.0
                while presence_penalty <= 1.0:
                    try:
                        print(f"    Trying edit_last_response with presence_penalty={presence_penalty:.1f}...")
                        response = await asyncio.wait_for(
                            loop.run_in_executor(
                                None,
                                functools.partial(gpt_chatbot.edit_last_response, full_prompt_for_gpt, temp=0.0, top_p=1.0, top_k=100, presence_penalty=presence_penalty, clear=True)
                            ),
                            timeout=timeout
                        )
                        if response and response.strip():
                            gpt_response_text = response
                            print(f"    Success on attempt {retries} with presence_penalty={presence_penalty:.1f}.")
                            break
                        else:
                            print(f"    Got empty response with presence_penalty={presence_penalty:.1f}. Increasing presence_penalty.")
                            presence_penalty += 0.1
                    except asyncio.TimeoutError:
                        print(f"    Timed out with presence_penalty={presence_penalty:.1f}. Increasing presence_penalty.")
                        presence_penalty += 0.1
                    except Exception as e:
                        print(f"    Error during edit_last_response with presence_penalty={presence_penalty:.1f}: {e}. Breaking inner loop.")
                        break

                if gpt_response_text:
                    break

            if not gpt_response_text or gpt_response_text.strip() == "":
                gpt_response_text = f"FAILED_TO_GENERATE_GPT_RESPONSE_FOR_CONTESTANT_{contestant_num}_ROUND_1"
                print(f"  GPT failed for contestant {contestant_num} (R1) after all retries and presence_penalty increases (result is inf).")

            output_file_r1 = f"{round_dir}/outputs/{contestant_num}_gpt_output_round1.txt"
            with open(output_file_r1, 'w', encoding='utf-8') as f:
                f.write(gpt_response_text)

            print(f"  Generating Python code for Contestant {contestant_num} (R1) using Gemini...")
            algorithm_code = self.generate_algorithm_code_with_gemini(gpt_response_text, str(contestant_num))
            algorithm_file = f"{round_dir}/algorithms/{contestant_num}_algorithm.py"
            with open(algorithm_file, 'w', encoding='utf-8') as f:
                f.write(algorithm_code)
        print(f"Finished initializing contestants for Round {self.current_round}.")


    def create_full_prompt(self, prompt_content: str) -> str:
        return f"""Your task is to devise a heuristic-based, step-by-step method to predict an Snellen-like score. This is your first attempt in a competitive environment; a well-reasoned, robust initial approach is critical.

**Objective:** Create a robust and explainable heuristic that intelligently interprets the provided data to produce a score. Your method must be generalizable and not overfit to the initial examples.

**Data Characteristics:**
- `size`: Corresponds to letter size. Smaller `size` values are harder to see and correspond to lower Snellen scores. The available sizes are: 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100.
- `is_correct`: Indicates a correct (1) or incorrect (0) response.
- `reaction_time`: The time taken for the user to provide an answer for a single trial.
- `cumulative_time`: The total time elapsed from the start of the test to the end of the current trial.
- The sequence of `size` values presented to the user is non-random and appears to change based on their previous correct or incorrect responses.
- The value `-1` may occasionally appear in the data columns.
- Your heuristic must develop a coherent and justifiable strategy for interpreting and utilizing all of these characteristics. You must decide which features are important signals of visual acuity (e.g., is `reaction_time` a meaningful indicator of difficulty?) and which, if any, should be ignored.

**Calibration Data:**
{prompt_content}

**Output Structure:**
Your final output must strictly adhere to the following structure. Do not include any text outside of the specified tags.

<self_reflection>
[1. **Internal Rubric for a World-Class Heuristic:** First, define the 4-5 critical categories that a top-tier heuristic for this problem must excel at (e.g., 'Signal Extraction from Sequential Data', 'Intelligent Feature Use', 'Handling of Anomalies', 'Scoring Logic Plausibility').
 2. **Design Goals:** Based on your rubric, state the primary goals your heuristic will aim to achieve.]
</self_reflection>

<rationale>
[Explain the core logic behind your heuristic. How does your chosen approach meet the design goals you set in your <self_reflection>? Justify your strategies for interpreting the sequential nature of the data, the timing information, and the presence of any anomalous values. Explain why you chose to use or ignore certain features.]
</rationale>

<step-by-step>
[Clearly outline the general, numbered steps of your proposed heuristic method.
**Crucially, any constants, value mappings (e.g., size-to-score tables), or mathematical formulas used in the calculation must be explicitly defined and justified within these steps.** The `<calculation>` block must only apply the rules defined here, with no unexplained "magic numbers."]
</step-by-step>

<calculation>
[Demonstrate your step-by-step method on at least one of the provided examples. You may show calculations for multiple examples if it helps to prove the generalizability and clarify the behavior of your method.]
</calculation>
"""

    def _extract_python_code_from_gemini(self, gemini_text_response: str) -> str:
        match = re.search(r"```python\n(.*?)\n```", gemini_text_response, re.DOTALL)
        if match:
            return match.group(1).strip()
        stripped_response = gemini_text_response.strip()
        if stripped_response.startswith("def ") or stripped_response.startswith("import "):
            lines = stripped_response.splitlines()
            if len(lines) > 1 and (lines[0].startswith("def ") or lines[0].startswith("import ")):
                 print("Warning: Python code extracted without ```python ... ```, using raw response starting with def/import.")
                 return stripped_response
        print("Warning: Could not find Python code block ```python ... ``` in Gemini response. Returning raw response as potential code.")
        return gemini_text_response 

    def generate_algorithm_code_with_gemini(self, gpt_response: str, contestant_id: str) -> str:
        if not self.gemini_model:
            return f"# Gemini model not initialized. Cannot generate code for contestant {contestant_id}."

        step_by_step_content = self.extract_section(gpt_response, "step-by-step")
        calculation_content = self.extract_section(gpt_response, "calculation")

        if "No content found" in step_by_step_content or "No content found" in calculation_content or \
           step_by_step_content.strip() == "" or calculation_content.strip() == "" or \
           "FAILED_TO_GENERATE_GPT_RESPONSE" in gpt_response :
            error_message = f"# Could not extract valid step-by-step or calculation from GPT response for contestant {contestant_id}.\n"
            if "FAILED_TO_GENERATE_GPT_RESPONSE" in gpt_response:
                error_message += f"# GPT response indicated failure: {gpt_response.splitlines()[0]}\n"
            else:
                error_message += f"# One or both sections were missing or empty.\n"
            error_message += f"# GPT response was:\n# {gpt_response}"
            return error_message

        raw_data_example_for_gemini_prompt = """stimulus (up = 0, left = 1, down = 2, right = 3)  side (right eye=0, left eye = 1)  size rgb response (up = 0, left = 1, down = 2, right = 3)  reaction_time cumulative_time is_correct
3 1 20 32 3 0.80857  1.81437 1
0 1 20 32 0 0.81165  3.64382 1
3 1 10 32 3 0.72199  5.38185 1
0 1 10 32 0 0.74794  7.15231 1
3 1 10 32 3 0.72431  8.88838 1
1 1 10 32 1 0.62406 10.53109 1
0 1 10 32 0 0.99715 12.54914 1
3 1 10 32 3 0.94271 14.50296 1
2 1 10 32 2 0.74709 16.27277 1
3 1  5 32 0 1.13137 18.41451 0
3 1  5 32 3 1.24969 20.68090 1
0 1  5 32 0 0.72345 22.41602 1
1 1  5 32 0 1.18248 24.62048 0
2 1  5 32 2 1.31128 26.94677 1"""

        gemini_prompt = f"""Given the following example of raw data format:
{raw_data_example_for_gemini_prompt}

And based on the following heuristic approach described in <step-by-step> and an example <calculation>:

<step-by-step>
{step_by_step_content}
</step-by-step>

<calculation>
{calculation_content}
</calculation>

Please write a single Python function named `calculate_snellen_score_{contestant_id}`.

This function must:
1.  Accept one argument: `data_string`, a multi-line string representing trial data.
2.  Parse the `data_string`. Each line corresponds to a trial with space-separated values. The columns are: `stimulus, side, size, rgb, response, reaction_time, cumulative_time, is_correct`.
3.  Implement the exact logic described in the <step-by-step> section. The <calculation> provides a working example.
4.  Return a single int value: the predicted Snellen score.
5.  Be self-contained and rely only on standard Python libraries (e.g., `math`). Do not use any machine learning libraries like PyTorch or TensorFlow..
6.  **Handle potential errors gracefully.** If the `data_string` is empty or malformed, return None.
7.  Do NOT include any `print()` statements or example usage. The output must be only the function code.
"""
        round_dir = f"{self.competition_data['output_dir']}/round{self.current_round}"
        os.makedirs(f"{round_dir}/prompts", exist_ok=True)
        gemini_prompt_file = f"{round_dir}/prompts/{contestant_id}_gemini_prompt_round{self.current_round}.txt"
        with open(gemini_prompt_file, 'w', encoding='utf-8') as f:
            f.write(gemini_prompt)

        try:
            print(f"    Sending prompt to Gemini for contestant {contestant_id} (Round {self.current_round})...")
            gemini_response = self.gemini_model.generate_content(
                gemini_prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.0)
            )
            python_code = self._extract_python_code_from_gemini(gemini_response.text)

            if not python_code.strip():
                 python_code = f"# Gemini returned empty code for contestant {contestant_id} (Round {self.current_round}).\n# Raw response was: {gemini_response.text}"

            if not python_code.strip().startswith(f"def calculate_snellen_score_{contestant_id}"):
                print(f"    Warning: Gemini output for {contestant_id} (Round {self.current_round}) might not be a correctly named function. Actual start: '{python_code.strip()[:50]}...'")
                if not python_code.strip().startswith("def ") and not python_code.strip().startswith("import ") and not python_code.strip().startswith("from "): 
                    python_code = f"# Gemini output (Round {self.current_round}) for contestant {contestant_id} did not start with 'def', 'import', or 'from'. Raw output:\n# {python_code}"
        except Exception as e:
            print(f"    Error calling Gemini API for contestant {contestant_id} (Round {self.current_round}): {e}")
            python_code = f"# Error generating code with Gemini for contestant {contestant_id} (Round {self.current_round}): {e}"

        return python_code

    def extract_section(self, text: str, section_name: str) -> str:
        start_tag_pattern = re.compile(rf"(?:\*\*)?<{re.escape(section_name)}>(?:\*\*)?")
        start_matches = list(start_tag_pattern.finditer(text))
        if not start_matches:
            return f"No content found for tag <{section_name}>"

        last_start_match = start_matches[-1]
        start_idx = last_start_match.end()

        end_tag_pattern = re.compile(rf"(?:\*\*)?</{re.escape(section_name)}>(?:\*\*)?")
        end_match = end_tag_pattern.search(text, pos=start_idx)

        if not end_match:
            end_tag = f"</{section_name}>"
            print(f"Warning: Missing end tag {end_tag} for section {section_name}. Taking content until end of text.")
            return text[start_idx:].strip()

        end_idx = end_match.start()
        return text[start_idx:end_idx].strip()

    def etdrs_to_snellen(self, etdrs_score: float) -> int:
        if not isinstance(etdrs_score, (int, float)) or math.isnan(etdrs_score) or math.isinf(etdrs_score):
            return 2000 
        logmar = 1.7 - (0.02 * etdrs_score)
        snellen_denominator = 20 * math.pow(10, logmar)
        snellen_denominator = max(5, min(2000, snellen_denominator))
        return int(round(snellen_denominator))

    async def _evaluate_historical_contestant(self, contestant_id: str, round_num: int, eval_df: pd.DataFrame) -> Dict[str, float]:
        round_dir = Path(self.competition_data['output_dir']) / f"round{round_num}"
        data_dir = self.competition_data['data_dir']
        
        algo_path = round_dir / "algorithms" / f"{contestant_id}_algorithm.py"
        if not algo_path.exists():
            print(f"      - Algorithm file not found for {contestant_id} in round {round_num}. Assigning high scores.")
            return {'MAE': float('inf'), 'RMSE': float('inf')}

        return await self._evaluate_algorithm_from_path(algo_path, contestant_id, eval_df, data_dir)

    async def _evaluate_single_contestant(self, contestant_id: str, eval_df: pd.DataFrame) -> Dict[str, float]:
        round_dir = f"{self.competition_data['output_dir']}/round{self.current_round}"
        data_dir = self.competition_data['data_dir']
        
        algo_path = Path(round_dir) / "algorithms" / f"{contestant_id}_algorithm.py"
        if not algo_path.exists():
            print(f"      - Algorithm file not found for {contestant_id}. Assigning high scores.")
            return {'MAE': float('inf'), 'RMSE': float('inf')}

        return await self._evaluate_algorithm_from_path(algo_path, contestant_id, eval_df, data_dir)

    async def _evaluate_algorithm_from_path(self, algo_path: Path, contestant_id: str, eval_df: pd.DataFrame, data_dir: str) -> Dict[str, float]:
        module_globals = {}
        try:
            code_to_exec = algo_path.read_text(encoding='utf-8')
            if not code_to_exec.strip() or all(line.strip().startswith("#") or not line.strip() for line in code_to_exec.splitlines()):
                print(f"      - Algorithm file {algo_path.name} is empty/commented. Assigning high scores.")
                return {'MAE': float('inf'), 'RMSE': float('inf')}
            exec(code_to_exec, module_globals)
        except Exception as e:
            print(f"      - Error executing algorithm from {algo_path.name}: {e}")
            return {'MAE': float('inf'), 'RMSE': float('inf')}

        func_name = f"calculate_snellen_score_{contestant_id}"
        if func_name in module_globals and callable(module_globals[func_name]):
            algorithm_func = module_globals[func_name]
            scores = {
                'MAE': self.calculate_mae(algorithm_func, eval_df, data_dir),
                'RMSE': self.calculate_rmse(algorithm_func, eval_df, data_dir)
            }
            return scores
        else:
            print(f"      - Function {func_name} not found in {algo_path.name}. Assigning high scores.")
            return {'MAE': float('inf'), 'RMSE': float('inf')}

    async def _iteratively_improve_and_evaluate(
        self,
        contestant_id: str,
        eval_df: pd.DataFrame,
        last_round_score: float,
        pair_dir: Path
    ) -> Tuple[Dict[str, float], int, int]:
        ranking_metric = 'RMSE' if self.competition_data['metric_mode'] == 'RMSE' else 'MAE'
        data_dir = self.competition_data['data_dir']
        round_dir = Path(self.competition_data['output_dir']) / f"round{self.current_round}"
        
        initial_algo_path = round_dir / "algorithms" / f"{contestant_id}_algorithm.py"
        initial_gpt_path = round_dir / "outputs" / f"{contestant_id}_gpt_output_round{self.current_round}.txt"
        current_scores = await self._evaluate_algorithm_from_path(initial_algo_path, contestant_id, eval_df, data_dir)
        current_score = current_scores.get(ranking_metric, float('inf'))
        
        attempts_dir = pair_dir / f"{contestant_id}_attempts"
        attempts_dir.mkdir(exist_ok=True)
        attempt_0_dir = attempts_dir / "attempt_0_temp_0.0"
        attempt_0_dir.mkdir(exist_ok=True)
        shutil.copy(initial_algo_path, attempt_0_dir / f"{contestant_id}_algorithm.py")
        if initial_gpt_path.exists():
            shutil.copy(initial_gpt_path, attempt_0_dir / f"{contestant_id}_gpt_output.txt")

        if current_score <= last_round_score:
            print(f"      - Contestant {contestant_id}: Performance is good (Current: {current_score:.4f} <= Last: {last_round_score:.4f}). No improvement loop needed.")
            return current_scores, 0, 0

        print(f"      - Contestant {contestant_id}: Performance degraded (Current: {current_score:.4f} > Last: {last_round_score:.4f}). Starting improvement loop.")
        
        best_score_in_loop = current_score
        best_scores_dict_in_loop = current_scores
        best_code_in_loop = initial_algo_path.read_text(encoding='utf-8')
        best_gpt_output_in_loop = initial_gpt_path.read_text(encoding='utf-8') if initial_gpt_path.exists() else ""
        best_iteration_in_loop = 0
        
        gpt_chatbot = GPTChatbot()
        for r in range(1, self.current_round):
            history_prompt_path = f"{self.competition_data['output_dir']}/round{r}/prompts/{contestant_id}_gpt_prompt_round{r}.txt"
            history_output_path = f"{self.competition_data['output_dir']}/round{r}/outputs/{contestant_id}_gpt_output_round{r}.txt"
            if os.path.exists(history_prompt_path) and os.path.exists(history_output_path):
                with open(history_prompt_path, 'r', encoding='utf-8') as hp_file: hist_prompt = hp_file.read()
                with open(history_output_path, 'r', encoding='utf-8') as ho_file: hist_output = ho_file.read()
                if hasattr(gpt_chatbot, 'add_history'):
                    gpt_chatbot.add_history(hist_prompt, hist_output)

        re_prompt_text = (round_dir / "prompts" / f"{contestant_id}_gpt_prompt_round{self.current_round}.txt").read_text(encoding='utf-8')
        
        iteration = 0
        loop = asyncio.get_running_loop()
        
        for i in range(20): # Loop 20 times for temps 0.1 to 2.0
            iteration += 1
            temp = (i + 1) * 0.1
            print(f"        - Attempt {iteration}: Trying temp={temp:.1f}...")
            
            try:
                gpt_response_text = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        functools.partial(gpt_chatbot.edit_last_response, re_prompt_text, temp=temp, top_p=0.95)
                    ),
                    timeout=7200.0
                )
                if not gpt_response_text or not gpt_response_text.strip():
                    print(f"        - GPT returned empty response for temp={temp:.1f}. Skipping.")
                    continue

                new_code = self.generate_algorithm_code_with_gemini(gpt_response_text, contestant_id)
                
                attempt_dir = attempts_dir / f"attempt_{iteration}_temp_{temp:.1f}"
                attempt_dir.mkdir(exist_ok=True)
                new_algo_path = attempt_dir / f"{contestant_id}_algorithm.py"
                new_algo_path.write_text(new_code, encoding='utf-8')
                (attempt_dir / f"{contestant_id}_gpt_output.txt").write_text(gpt_response_text, encoding='utf-8')

                new_scores = await self._evaluate_algorithm_from_path(new_algo_path, contestant_id, eval_df, data_dir)
                new_score = new_scores.get(ranking_metric, float('inf'))
                print(f"        - Attempt {iteration} result: {ranking_metric}={new_score:.4f}")

                if new_score < best_score_in_loop:
                    print(f"        - New best in loop found at iteration {iteration}!")
                    best_score_in_loop = new_score
                    best_scores_dict_in_loop = new_scores
                    best_code_in_loop = new_code
                    best_gpt_output_in_loop = gpt_response_text
                    best_iteration_in_loop = iteration
                
                if new_score <= last_round_score:
                    print(f"        - Success! Performance goal reached. Stopping loop.")
                    break
                
            except Exception as e:
                print(f"        - Error during attempt {iteration} with temp={temp:.1f}: {e}")

        print(f"      - Loop finished for {contestant_id}. Best score in loop: {best_score_in_loop:.4f} (from iteration {best_iteration_in_loop}). Total iterations: {iteration}.")
        if best_code_in_loop != initial_algo_path.read_text(encoding='utf-8'):
            print(f"      - Updating main algorithm and output files for {contestant_id} with the best version from the loop.")
            initial_algo_path.write_text(best_code_in_loop, encoding='utf-8')
            initial_gpt_path.write_text(best_gpt_output_in_loop, encoding='utf-8')
            
        return best_scores_dict_in_loop, iteration, best_iteration_in_loop

    async def _evaluate_pair(self, c1_id: str, c2_id: str, pair_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        evaluate_mode = self.competition_data.get("evaluate_mode", "normal")
        ranking_metric = 'RMSE' if self.competition_data['metric_mode'] == 'RMSE' else 'MAE'
        
        c1_results = {'scores': {'MAE': float('inf'), 'RMSE': float('inf')}, 'iteration': 0, 'best_iteration': 0}
        c2_results = {'scores': {'MAE': float('inf'), 'RMSE': float('inf')}, 'iteration': 0, 'best_iteration': 0}

        if evaluate_mode == "pair":
            print(f"    - Evaluating pair ({c1_id}, {c2_id}) using 'pair' mode with cumulative data.")
            
            c1_ancestors = self.contestant_ancestry.get(c1_id, {c1_id})
            c2_ancestors = self.contestant_ancestry.get(c2_id, {c2_id})
            combined_ancestors = c1_ancestors.union(c2_ancestors)
            
            print(f"      - Contestant {c1_id} has lineage: {c1_ancestors}")
            print(f"      - Contestant {c2_id} has lineage: {c2_ancestors}")
            print(f"      - Evaluating on combined lineage: {combined_ancestors}")

            cumulative_data = [item for ancestor_id in combined_ancestors for item in self.initial_contestant_data.get(ancestor_id, [])]
            unique_cumulative_data = list({item['id']: item for item in cumulative_data}.values())

            if not unique_cumulative_data:
                print(f"    - Warning: No initial data found for pair ({c1_id}, {c2_id}). Both will fail.")
                return c1_results, c2_results
                
            pair_eval_df = pd.DataFrame(unique_cumulative_data)
            pair_eval_df.to_csv(pair_dir / "paired_evaluation_data.csv", index=False)
            
            last_round_scores = {}
            if self.current_round > 1:
                try:
                    prev_round_results_path = Path(self.competition_data['output_dir']) / f"round{self.current_round - 1}" / f"results_round{self.current_round - 1}.csv"
                    df = pd.read_csv(prev_round_results_path, dtype={'Contestant': str})
                    last_round_scores = df.set_index('Contestant')[ranking_metric].to_dict()
                except FileNotFoundError:
                    print("      - Warning: Previous round results not found. Cannot compare for improvement. Will not loop.")
            
            c1_last_score = last_round_scores.get(c1_id, float('inf'))
            c2_last_score = last_round_scores.get(c2_id, float('inf'))

            c1_final_scores, c1_iterations, c1_best_iteration = await self._iteratively_improve_and_evaluate(c1_id, pair_eval_df, c1_last_score, pair_dir)
            c2_final_scores, c2_iterations, c2_best_iteration = await self._iteratively_improve_and_evaluate(c2_id, pair_eval_df, c2_last_score, pair_dir)
            
            c1_results = {'scores': c1_final_scores, 'iteration': c1_iterations, 'best_iteration': c1_best_iteration}
            c2_results = {'scores': c2_final_scores, 'iteration': c2_iterations, 'best_iteration': c2_best_iteration}

        else: # Normal mode
            print(f"    - Evaluating pair ({c1_id}, {c2_id}) using 'normal' mode.")
            
            if 'evaluation_samples' in self.competition_data and self.competition_data['evaluation_samples']:
                eval_df = pd.DataFrame(self.competition_data['evaluation_samples'])
            else:
                metadata_path = self.competition_data.get('metadata_path', "../dataset/rawdata.xlsx")
                eval_df = pd.read_excel(metadata_path)

            c1_scores = await self._evaluate_single_contestant(c1_id, eval_df)
            c2_scores = await self._evaluate_single_contestant(c2_id, eval_df)
            c1_results['scores'] = c1_scores
            c2_results['scores'] = c2_scores

        return c1_results, c2_results

    async def evaluate_algorithms(self) -> Dict[str, Dict[str, float]]:
        round_dir = f"{self.competition_data['output_dir']}/round{self.current_round}"
        
        algo_dir_for_current_round_logic = f"{round_dir}/algorithms"
        if not os.path.exists(algo_dir_for_current_round_logic):
            print(f"Algorithm directory not found for round {self.current_round}: {algo_dir_for_current_round_logic}")
            return {}

        algorithm_files = [f for f in os.listdir(algo_dir_for_current_round_logic) if f.endswith('_algorithm.py')]
        results: Dict[str, Dict[str, float]] = {}
        
        if 'evaluation_samples' in self.competition_data and self.competition_data['evaluation_samples']:
            print(f"Using explicit evaluation dataset with {len(self.competition_data['evaluation_samples'])} samples.")
            eval_df = pd.DataFrame(self.competition_data['evaluation_samples'])
        else:
            print("Warning: Explicit evaluation dataset not found. Falling back to using full metadata file for evaluation.")
            try:
                metadata_path = self.competition_data.get('metadata_path', "../dataset/rawdata.xlsx")
                eval_df = pd.read_excel(metadata_path)
            except FileNotFoundError:
                print(f"Error: Metadata file not found at {metadata_path} during evaluation for round {self.current_round}.")
                return {}

        eligible_contestant_ids_this_round = {f.removesuffix('_algorithm.py') for f in algorithm_files}
        print(f"Evaluating {len(eligible_contestant_ids_this_round)} algorithms for Round {self.current_round} on evaluation data: {sorted(list(eligible_contestant_ids_this_round))}")

        for contestant_id_str in eligible_contestant_ids_this_round:
            scores = await self._evaluate_single_contestant(contestant_id_str, eval_df)
            results[contestant_id_str] = scores
            report_str = ", ".join([f"{k} = {v:.4f}" for k, v in scores.items()])
            print(f"  Contestant {contestant_id_str} (Round {self.current_round}): {report_str}")
        
        final_results_for_round = {}
        for cid in eligible_contestant_ids_this_round:
            if cid in results:
                final_results_for_round[cid] = results[cid]
            else:
                print(f"  Contestant {cid} was eligible for Round {self.current_round} but had no evaluation result. Assigning high scores.")
                final_results_for_round[cid] = {'MAE': float('inf'), 'RMSE': float('inf')}

        if final_results_for_round:
            results_list = []
            for k, v in final_results_for_round.items():
                row = {"Contestant": k, **v, "iteration": 0, "best_iteration": 0}
                results_list.append(row)
            
            ranking_metric = 'RMSE' if self.competition_data['metric_mode'] == 'RMSE' else 'MAE'
            results_df = pd.DataFrame(results_list).sort_values(ranking_metric)
            results_df.to_csv(f"{round_dir}/results_round{self.current_round}.csv", index=False)
        else:
            print(f"No results to save for round {self.current_round}.")
        
        return final_results_for_round


    def calculate_mae(self, algorithm_func, metadata_df: pd.DataFrame, data_dir: str) -> float:
        errors = []
        for _, row in metadata_df.iterrows():
            file_id = str(row.get('ID', row.get('id')))
            ground_truth_etdrs = float(row.get('VA ETDRS Score', row.get('score')))
            input_data_path = row.get('path', os.path.join(data_dir, f"{file_id}.csv"))
            input_data_string = self.load_and_prepare_input_string(input_data_path)
            
            if not input_data_string:
                errors.append(abs(self.etdrs_to_snellen(ground_truth_etdrs) - 2000))
                continue
            try:
                prediction_snellen = algorithm_func(input_data_string)
                if not isinstance(prediction_snellen, (int, float)) or math.isnan(prediction_snellen) or math.isinf(prediction_snellen):
                    errors.append(abs(self.etdrs_to_snellen(ground_truth_etdrs) - 2000)) 
                    continue
                snellen_ground_truth = self.etdrs_to_snellen(ground_truth_etdrs)
                snellen_prediction = prediction_snellen #self.etdrs_to_snellen(prediction_etdrs)
                error = abs(snellen_ground_truth - snellen_prediction)
                errors.append(error)
            except Exception:
                errors.append(abs(self.etdrs_to_snellen(ground_truth_etdrs) - 2000)) 
                continue
        if not errors:
            return float('inf') 
        mean_snellen_error = np.mean(errors) if errors else float('inf')
        return mean_snellen_error if pd.notna(mean_snellen_error) else float('inf')

    def calculate_rmse(self, algorithm_func, metadata_df: pd.DataFrame, data_dir: str) -> float:
        squared_errors = []
        for _, row in metadata_df.iterrows():
            file_id = str(row.get('ID', row.get('id')))
            ground_truth_etdrs = float(row.get('VA ETDRS Score', row.get('score')))
            input_data_path = row.get('path', os.path.join(data_dir, f"{file_id}.csv"))
            input_data_string = self.load_and_prepare_input_string(input_data_path)

            if not input_data_string:
                squared_errors.append((self.etdrs_to_snellen(ground_truth_etdrs) - 2000)**2)
                continue
            try:
                prediction_snellen = algorithm_func(input_data_string)
                if not isinstance(prediction_snellen, (int, float)) or math.isnan(prediction_snellen) or math.isinf(prediction_snellen):
                    squared_errors.append((self.etdrs_to_snellen(ground_truth_etdrs) - 2000)**2)
                    continue
                snellen_ground_truth = self.etdrs_to_snellen(ground_truth_etdrs)
                snellen_prediction = prediction_snellen #self.etdrs_to_snellen(prediction_etdrs)
                squared_error = (snellen_ground_truth - snellen_prediction)**2
                squared_errors.append(squared_error)
            except Exception:
                squared_errors.append((self.etdrs_to_snellen(ground_truth_etdrs) - 2000)**2)
                continue
        if not squared_errors:
            return float('inf')
        mean_squared_error = np.mean(squared_errors)
        return math.sqrt(mean_squared_error) if pd.notna(mean_squared_error) else float('inf')


    def load_and_prepare_input_string(self, file_path: str) -> str:
        try:
            df = pd.read_csv(file_path, skiprows=1) 
            if df.shape[1] < 8: 
                 return ""
            data_for_prediction = df.iloc[:, :8] 
            return data_for_prediction.to_string(index=False, header=False)
        except FileNotFoundError:
            return ""
        except Exception:
            return ""

    def _get_best_survivor_info_from_last_round(self) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, float]]]:
        if self.current_round <= 1:
            return None, None, None

        prev_round = self.current_round - 1
        prev_round_dir = Path(self.competition_data['output_dir']) / f"round{prev_round}"
        results_path = prev_round_dir / f"results_round{prev_round}.csv"
        ranking_metric = 'RMSE' if self.competition_data['metric_mode'] == 'RMSE' else 'MAE'

        if not results_path.exists():
            print(f"Warning: Could not find results file for round {prev_round} to get best survivor info.")
            return None, None, None

        try:
            df = pd.read_csv(results_path, dtype={'Contestant': str})
            if df.empty:
                return None, None, None

            last_round_survivors = self.survivors
            df_survivors = df[df['Contestant'].isin(last_round_survivors)]

            if df_survivors.empty:
                print(f"Warning: No survivors from round {prev_round} found in results file.")
                df_survivors = df
            
            if df_survivors.empty:
                return None, None, None

            best_survivor_row = df_survivors.sort_values(by=ranking_metric).iloc[0]
            best_survivor_id = best_survivor_row['Contestant']
            
            score_cols = ['MAE', 'RMSE']
            best_survivor_scores = {col: best_survivor_row[col] for col in score_cols if col in df.columns}

            gpt_output_path = prev_round_dir / "outputs" / f"{best_survivor_id}_gpt_output_round{prev_round}.txt"
            step_by_step_content = None
            if gpt_output_path.exists():
                gpt_output = gpt_output_path.read_text(encoding='utf-8')
                step_by_step_content = self.extract_section(gpt_output, "step-by-step")

            data_path = Path(self.competition_data['output_dir']) / "round1" / "data" / f"{best_survivor_id.split('_')[0]}_data.txt"
            data_content = None
            if data_path.exists():
                data_content = data_path.read_text(encoding='utf-8')

            return step_by_step_content, data_content, best_survivor_scores

        except Exception as e:
            print(f"Error getting best survivor info from round {prev_round}: {e}")
            return None, None, None

    def _get_all_rivals_info_from_last_round(self, current_contestant_id: str) -> List[Dict[str, Any]]:
        if self.current_round <= 1:
            return []

        prev_round = self.current_round - 1
        prev_round_dir = Path(self.competition_data['output_dir']) / f"round{prev_round}"
        results_path = prev_round_dir / f"results_round{prev_round}.csv"
        ranking_metric = 'RMSE' if self.competition_data['metric_mode'] == 'RMSE' else 'MAE'

        if not results_path.exists():
            print(f"Warning: Could not find results file for round {prev_round} to get rival info.")
            return []

        try:
            df = pd.read_csv(results_path, dtype={'Contestant': str})
            if df.empty:
                return []

            all_prev_round_contestants = df['Contestant'].tolist()
            rival_ids = [cid for cid in all_prev_round_contestants if cid != current_contestant_id]

            rival_info_list = []
            for rival_id in rival_ids:
                rival_row = df[df['Contestant'] == rival_id]
                if rival_row.empty:
                    continue

                score_cols = ['MAE', 'RMSE']
                rival_scores = {col: rival_row.iloc[0][col] for col in score_cols if col in df.columns}

                gpt_output_path = prev_round_dir / "outputs" / f"{rival_id}_gpt_output_round{prev_round}.txt"
                step_by_step_content = None
                if gpt_output_path.exists():
                    gpt_output = gpt_output_path.read_text(encoding='utf-8')
                    step_by_step_content = self.extract_section(gpt_output, "step-by-step")
                
                rival_info_list.append({
                    "id": rival_id,
                    "scores": rival_scores,
                    "step": step_by_step_content
                })
            
            rival_info_list.sort(key=lambda x: x.get('scores', {}).get(ranking_metric, float('inf')))

            return rival_info_list

        except Exception as e:
            print(f"Error getting rival info from round {prev_round}: {e}")
            return []

    async def _generate_revised_algorithm(
        self, 
        contestant_id_to_create: str, 
        loser_data_content: str,
        own_last_round_scores: Optional[Dict[str, float]],
        history_source_id: Optional[str] = None
    ) -> str:
        print(f"    Generating revised algorithm for new contestant '{contestant_id_to_create}' (Round {self.current_round})...")
        gpt_chatbot = GPTChatbot()
        
        effective_history_source_id = history_source_id if history_source_id is not None else contestant_id_to_create
        
        print(f"      Building history for contestant using source ID: '{effective_history_source_id}'")
        for r in range(1, self.current_round):
            base_id_for_history = effective_history_source_id.split('_champ')[0]
            history_prompt_path = f"{self.competition_data['output_dir']}/round{r}/prompts/{base_id_for_history}_gpt_prompt_round{r}.txt"
            history_output_path = f"{self.competition_data['output_dir']}/round{r}/outputs/{base_id_for_history}_gpt_output_round{r}.txt"
            if os.path.exists(history_prompt_path) and os.path.exists(history_output_path):
                try:
                    with open(history_prompt_path, 'r', encoding='utf-8') as hp_file: hist_prompt = hp_file.read()
                    with open(history_output_path, 'r', encoding='utf-8') as ho_file: hist_output = ho_file.read()
                    if hasattr(gpt_chatbot, 'add_history') and callable(getattr(gpt_chatbot, 'add_history')):
                        gpt_chatbot.add_history(hist_prompt, hist_output)
                except Exception as e:
                    print(f"      Error loading history for Round {r} for source {base_id_for_history}: {e}")

        prompt_mode = self.competition_data.get("prompt_mode", "default")
        metric_mode = self.competition_data.get("metric_mode", "MAE")
        ranking_metric_name = 'RMSE' if metric_mode == 'RMSE' else 'MAE'
        additional_content = ""

        if prompt_mode == "add_best_step":
            best_step_content, _, best_competitor_scores = self._get_best_survivor_info_from_last_round()
            if best_competitor_scores:
                best_score_val = best_competitor_scores.get(ranking_metric_name)
                if best_score_val is not None:
                    score_report = f"{ranking_metric_name}: {best_score_val:.4f}"
                    additional_content += f"\nFor reference, the score from the best competitor in the last round was: {score_report}\n"
            if best_step_content:
                additional_content += f"""
This is the step-by-step from the best performing algorithm from the last round:
<step-by-step>
{best_step_content}
</step-by-step>
"""
        elif prompt_mode == "add_all_contestants_step":
            rival_info = self._get_all_rivals_info_from_last_round(contestant_id_to_create)
            if rival_info:
                additional_content += "\nHere is the step-by-step approach and scores from all other rivals in the last round. Use this to improve your own algorithm.\n"
                for rival in rival_info:
                    rival_id, rival_scores, rival_step = rival.get('id'), rival.get('scores', {}), rival.get('step')
                    rival_score_val = rival_scores.get(ranking_metric_name)
                    score_report = f"{ranking_metric_name}: {rival_score_val:.4f}" if rival_score_val is not None else "N/A"
                    additional_content += f"\n--- Rival: {rival_id} | Last Round Score: {score_report} ---\n"
                    if rival_step and "No content found" not in rival_step:
                        additional_content += f"<step-by-step>\n{rival_step}\n</step-by-step>\n"
                    else:
                        additional_content += "[No step-by-step content was available for this rival.]\n"

        own_score_report = ""
        if own_last_round_scores:
            score_value = own_last_round_scores.get(ranking_metric_name)
            if score_value is not None and not math.isinf(score_value):
                own_score_report = f"Your algorithm's score in the last round was ({ranking_metric_name}: {score_value:.4f}). The goal is to achieve a lower {ranking_metric_name}."
            else:
                own_score_report = "Your algorithm failed or produced an invalid result in the last round."
        else:
            own_score_report = "This is your algorithm's first revision attempt."

        revision_instruction = f"""You are in a competition to refine a heuristic algorithm. Your goal is to create a new algorithm that is demonstrably superior to your last to be the winner in this competition. A simple tweak is insufficient; you must re-evaluate your entire approach to win.

**Your Performance Last Round:**
{own_score_report}
"""
            
        re_prompt_text = f"""{revision_instruction}
**Analysis Material:**
1.  `loser_data_content`: An example where your previous algorithm performed poorly, or a style of data you must now master to defend your title.
{loser_data_content}
2.  `rival_heuristic`: he <step-by-step> logic and performance scores from all other competitors in the last round. Analyze them for strengths to emulate and weaknesses to exploit.
{additional_content}

**Task:**
Your task is to perform a rigorous analysis and then produce a refined algorithm. Your new heuristic MUST be a clear improvement. Your thinking process must be laid out in the tags below.

**Output Structure:**
Your final output must strictly adhere to this structure.

<self_reflection>
[1. **Internal Rubric:** First, define the 5 key categories of a competition-winning heuristic for this specific problem (e.g., Threshold Detection Accuracy, Scoring Plausibility, Robustness to Noise, Penalty/Bonus Logic, Generalizability).
 2. **Self-Assessment:** Briefly state why your previous algorithm failed against this rubric.
 3. **Commitment to Improvement:** State your commitment to iterating internally until your new proposed heuristic meets the highest standards of your rubric, ensuring it is a fundamental improvement and not just a minor change. This new heuristic must outperform your previous one.]
</self_reflection>

<analysis>
[**1. Self-Critique (Failure Analysis):**
   - My previous heuristic's core logic was: [Summarize your last algorithm in 1-2 sentences].
   - Applying my old logic to the `loser_data_content`, the predicted score would be [Show calculation and result]. The ground truth is [Insert ground truth score].
   - The failure point, based on my rubric, was: [Pinpoint the specific reason for the error, linking it to one of your rubric categories from <self_reflection>. e.g., "My 'Threshold Detection' was too simplistic and was thrown off by a single correct answer at a small size."]

**2. Rival Analysis:**
   - The rival's heuristic differs from mine primarily in: [Describe the key logical difference, e.g., "It calculates a base score using a weighted average of all lines, rather than finding one specific threshold line."]
   - This approach is superior against my rubric category [Category Name] because: [Explain why the rival's logic was more robust for this specific case.]

**3. Proposed Refinement (New Strategy):**
   - Based on this, I will discard/radically revise my logic. My new strategy is: [State the specific, targeted change or new approach. e.g., "I will adopt a per-letter scoring model. I will assign a base score of 100 and subtract penalty points for each line not read, weighted by the number of trials, plus a per-letter penalty for misses on lines easier than the threshold."]
   - This new strategy directly addresses the failure point by [Explain how the change fixes the problem] and aims to score perfectly on my internal rubric.]
</analysis>

<step-by-step>
[Clearly outline the general steps of your EVOLVED heuristic method.
**Crucially, any constants, value mappings (e.g., size-to-score tables), or mathematical formulas used in the calculation must be explicitly defined and justified within these steps.** The `<calculation>` block must only apply the rules defined here, with no unexplained "magic numbers."]
</step-by-step>

<calculation>
[Demonstrate the superiority and robustness of your evolved heuristic with two proofs.

**Proof 1: Mastery of New Challenges**
Apply your new method to the `loser_data_content`. Show the calculations to prove that your evolved logic now correctly handles this difficult case that challenged your previous approach.

**Proof 2: Retention of Old Strengths (Backward-Compatibility)**
Recall a data example from your history where your *old* algorithm performed well. Apply your new method to that same data. Show the calculations to prove that your score remains accurate, demonstrating that your evolution has not introduced new weaknesses (no regressions).]
</calculation>
"""
        current_round_dir = f"{self.competition_data['output_dir']}/round{self.current_round}"
        current_gpt_prompt_path = f"{current_round_dir}/prompts/{contestant_id_to_create}_gpt_prompt_round{self.current_round}.txt"
        with open(current_gpt_prompt_path, 'w', encoding='utf-8') as f: f.write(re_prompt_text)

        gpt_response_text = None
        timeout = 7200.0
        loop = asyncio.get_running_loop()

        print(f"      Calling GPT for refined heuristic for {contestant_id_to_create}...")
        try:
            gpt_response_text = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    functools.partial(gpt_chatbot.generate_response, re_prompt_text, temp=0.0, top_p=1.0, top_k=100, clear=False)
                ),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            print(f"      Initial GPT generation for {contestant_id_to_create} timed out.")
            gpt_response_text = None
        except Exception as e:
            print(f"      Initial GPT generation for {contestant_id_to_create} failed with error: {e}")
            gpt_response_text = None

        retries = 0
        max_retries = 10
        while (not gpt_response_text or gpt_response_text.strip() == "") and retries < max_retries:
            retries += 1
            print(f"    Retrying GPT for contestant {contestant_id_to_create} (R{self.current_round})... (Attempt {retries}/{max_retries})")

            presence_penalty = 0.0
            while presence_penalty <= 1.0:
                try:
                    print(f"      Trying edit_last_response with presence_penalty={presence_penalty:.1f}...")
                    response = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            functools.partial(gpt_chatbot.edit_last_response, re_prompt_text, temp=0.0, top_p=1.0, top_k=100, presence_penalty=presence_penalty, clear=False)
                        ),
                        timeout=timeout
                    )
                    if response and response.strip():
                        gpt_response_text = response
                        print(f"      Success on attempt {retries} with presence_penalty={presence_penalty:.1f}.")
                        break
                    else:
                        print(f"      Got empty response with presence_penalty={presence_penalty:.1f}. Increasing presence_penalty.")
                        presence_penalty += 0.1
                except asyncio.TimeoutError:
                    print(f"      Timed out with presence_penalty={presence_penalty:.1f}. Increasing presence_penalty.")
                    presence_penalty += 0.1
                except Exception as e:
                    print(f"      Error during edit_last_response with presence_penalty={presence_penalty:.1f}: {e}. Breaking inner loop.")
                    break

            if gpt_response_text:
                break
        
        if not gpt_response_text or gpt_response_text.strip() == "":
            gpt_response_text = f"FAILED_TO_GENERATE_GPT_RESPONSE_FOR_CONTESTANT_{contestant_id_to_create}_ROUND_{self.current_round}"
            print(f"  GPT failed for contestant {contestant_id_to_create} (R{self.current_round}) after all retries (result is inf).")

        current_gpt_output_path = f"{current_round_dir}/outputs/{contestant_id_to_create}_gpt_output_round{self.current_round}.txt"
        with open(current_gpt_output_path, 'w', encoding='utf-8') as f: f.write(gpt_response_text)

        print(f"      Calling Gemini for Python code for new contestant '{contestant_id_to_create}'...")
        algorithm_code = self.generate_algorithm_code_with_gemini(gpt_response_text, contestant_id_to_create)
        
        return algorithm_code

    async def _run_tournament_round(self) -> Tuple[List[str], List[str], List[Dict]]:
        """
        Pairs contestants, runs matches, evaluates, updates Elo, and returns results for a tournament round.
        Returns (winners_list, losers_list, results_summary_list)
        """
        round_dir = Path(self.competition_data['output_dir']) / f"round{self.current_round}"
        round_dir.mkdir(parents=True, exist_ok=True)
        
        contestants_in_round = list(self.survivors)
        
        pairs = []
        if self.current_round == 1:
            random.shuffle(contestants_in_round)
            for i in range(0, len(contestants_in_round), 2):
                pairs.append((contestants_in_round[i], contestants_in_round[i+1]))
            print(f"Round 1: Randomly paired {len(pairs)} pairs.")
        else:
            sorted_contestants = sorted(contestants_in_round, key=lambda cid: self._get_elo(cid), reverse=True)
            for i in range(0, len(sorted_contestants), 2):
                pairs.append((sorted_contestants[i], sorted_contestants[i+1]))
            print(f"Round {self.current_round}: Paired {len(pairs)} pairs based on Elo rating.")

        winners = []
        losers = []
        results_summary = []
        all_results_for_csv = []
        
        ranking_metric = 'RMSE' if self.competition_data['metric_mode'] == 'RMSE' else 'MAE'

        for c1, c2 in pairs:
            pair_dir = round_dir / f"pair_{c1}_vs_{c2}"
            pair_dir.mkdir(exist_ok=True)
            print(f"  Match: {c1} (Elo: {self._get_elo(c1):.0f}) vs. {c2} (Elo: {self._get_elo(c2):.0f})")

            c1_results, c2_results = await self._evaluate_pair(c1, c2, pair_dir)
            c1_scores, c2_scores = c1_results['scores'], c2_results['scores']
            c1_iteration, c2_iteration = c1_results['iteration'], c2_results['iteration']
            c1_best_iteration, c2_best_iteration = c1_results['best_iteration'], c2_results['best_iteration']

            c1_score = c1_scores.get(ranking_metric, float('inf'))
            c2_score = c2_scores.get(ranking_metric, float('inf'))

            if c1_score <= c2_score:
                winner, loser, winner_score, loser_score = c1, c2, c1_score, c2_score
                winner_scores_dict, loser_scores_dict = c1_scores, c2_scores
                winner_iteration, loser_iteration = c1_iteration, c2_iteration
                winner_best_iteration, loser_best_iteration = c1_best_iteration, c2_best_iteration
            else:
                winner, loser, winner_score, loser_score = c2, c1, c2_score, c1_score
                winner_scores_dict, loser_scores_dict = c2_scores, c1_scores
                winner_iteration, loser_iteration = c2_iteration, c1_iteration
                winner_best_iteration, loser_best_iteration = c2_best_iteration, c1_best_iteration
                
            winners.append(winner)
            losers.append(loser)
            
            new_winner_elo, new_loser_elo = self._update_elo(winner, loser, winner_score, loser_score)
            
            if loser in self.contestant_ancestry:
                self.contestant_ancestry[winner].update(self.contestant_ancestry.pop(loser))

            print(f"    Winner: {winner} (Score: {winner_score:.4f}). Loser: {loser} (Score: {loser_score:.4f})")
            print(f"    Elo Update: {winner} -> {new_winner_elo:.0f}, {loser} -> {new_loser_elo:.0f}")

            match_result = {
                "winner": winner, "loser": loser,
                "winner_score": winner_score, "loser_score": loser_score,
                "winner_scores_dict": winner_scores_dict,
                "winner_new_elo": new_winner_elo, "loser_new_elo": new_loser_elo,
            }
            results_summary.append(match_result)
            
            pair_results_df = pd.DataFrame([
                {"Contestant": winner, **winner_scores_dict, "Elo": new_winner_elo, "Outcome": "Win", "iteration": winner_iteration, "best_iteration": winner_best_iteration},
                {"Contestant": loser, **loser_scores_dict, "Elo": new_loser_elo, "Outcome": "Loss", "iteration": loser_iteration, "best_iteration": loser_best_iteration},
            ])
            pair_results_df.to_csv(pair_dir / "match_results.csv", index=False)
            
            all_results_for_csv.extend(pair_results_df.to_dict('records'))

        if all_results_for_csv:
            round_results_df = pd.DataFrame(all_results_for_csv).sort_values(by=ranking_metric)
            round_results_df.to_csv(round_dir / f"results_round{self.current_round}.csv", index=False)

        return winners, losers, results_summary

    async def _prepare_next_tournament_round(self, results_summary: List[Dict]):
        """
        Takes the results of a tournament round and generates the revised algorithms
        for the winners, who will compete in the next round.
        """
        next_round = self.current_round + 1
        next_round_dir = Path(self.competition_data['output_dir']) / f"round{next_round}"
        for subdir in ["prompts", "outputs", "algorithms"]:
            (next_round_dir / subdir).mkdir(parents=True, exist_ok=True)
            
        for match_result in results_summary:
            winner_id = match_result['winner']
            loser_id = match_result['loser']
            
            print(f"  Revising winner '{winner_id}' using loser '{loser_id}'s info for Round {next_round}.")
            
            base_loser_id = str(loser_id).split('_')[0]
            loser_data_file_path = f"{self.competition_data['output_dir']}/round1/data/{base_loser_id}_data.txt"
            try:
                with open(loser_data_file_path, 'r', encoding='utf-8') as f:
                    loser_data_content = f.read()
            except FileNotFoundError:
                print(f"    Warning: Data file for loser '{loser_id}' not found. Using a generic prompt.")
                loser_data_content = "The previous opponent's data was not available. Please refine your algorithm based on general principles."

            winner_scores = match_result['winner_scores_dict']

            self.current_round = next_round
            revised_code = await self._generate_revised_algorithm(
                contestant_id_to_create=winner_id,
                loser_data_content=loser_data_content,
                own_last_round_scores=winner_scores,
                history_source_id=winner_id
            )
            self.current_round = next_round - 1

            algo_path = next_round_dir / "algorithms" / f"{winner_id}_algorithm.py"
            algo_path.write_text(revised_code, encoding='utf-8')

    async def _run_final_testing(self) -> Optional[str]:
        test_data_path = self.competition_data.get("test_data_path")
        if not test_data_path:
            return None

        try:
            test_df = pd.read_csv(test_data_path)
            print(f"\n--- Running Final Testing on {test_data_path} ---")
        except FileNotFoundError:
            msg = f"\n- Warning: Test data file not found at {test_data_path}. Skipping final testing."
            print(msg)
            return msg

        evaluate_mode = self.competition_data.get("evaluate_mode", "normal")
        output_dir = Path(self.competition_data['output_dir'])
        final_round_dir = output_dir / f"round{self.current_round}"
        data_dir = self.competition_data['data_dir']
        
        results_to_test = []

        # 1. Test the tournament winner
        if self.survivors:
            winner_id = self.survivors[0]
            winner_round = self.current_round
            winner_algo_path = final_round_dir / "algorithms" / f"{winner_id}_algorithm.py"
            if winner_algo_path.exists():
                print(f"  - Testing Tournament Winner: {winner_id} from round {winner_round}")
                winner_scores = await self._evaluate_algorithm_from_path(winner_algo_path, winner_id, test_df, data_dir)
                results_to_test.append({"Contestant": f"Tournament Winner (r{winner_round}_{winner_id})", **winner_scores})
            else:
                print(f"  - Warning: Algorithm for tournament winner {winner_id} not found.")

        # 2. If in 'pair' mode, test the global best from the evaluation phase
        if evaluate_mode == 'pair':
            global_eval_path = final_round_dir / "final_global_evaluation.csv"
            if global_eval_path.exists():
                try:
                    global_df = pd.read_csv(global_eval_path)
                    if not global_df.empty:
                        global_best_name = global_df.iloc[0]['Contestant'] # e.g., "r2_4"
                        
                        match = re.match(r"r(\d+)_(\w+)", global_best_name)
                        if match:
                            global_best_round, global_best_id = int(match.group(1)), match.group(2)
                            global_best_algo_path = output_dir / f"round{global_best_round}" / "algorithms" / f"{global_best_id}_algorithm.py"
                            
                            if global_best_algo_path.exists():
                                print(f"  - Testing Global Best (from eval set): {global_best_id} from round {global_best_round}")
                                global_best_scores = await self._evaluate_algorithm_from_path(global_best_algo_path, global_best_id, test_df, data_dir)
                                results_to_test.append({"Contestant": f"Global Best ({global_best_name})", **global_best_scores})
                            else:
                                 print(f"  - Warning: Algorithm for global best {global_best_name} not found.")
                        else:
                            print(f"  - Warning: Could not parse global best contestant name '{global_best_name}'.")
                except Exception as e:
                    print(f"  - Error processing global evaluation file for testing: {e}")
            else:
                print("  - Warning: final_global_evaluation.csv not found, cannot test global best.")

        if not results_to_test:
            msg = "\n- No algorithms could be identified for final testing."
            print(msg)
            return msg

        test_results_df = pd.DataFrame(results_to_test)
        final_test_csv_path = final_round_dir / "final_test_results.csv"
        test_results_df.to_csv(final_test_csv_path, index=False)
        
        message = f"\n- Final Testing on Test Data Complete. Results saved to {final_test_csv_path}"
        message += f"\n--- Test Results ---\n{test_results_df.to_string(index=False)}"
        return message

    async def _advance_round_tournament(self) -> list[TextContent]:
        print(f"\n--- Advancing to Round {self.current_round} (Tournament Mode) ---")
        
        is_final_round = len(self.survivors) == 2
        
        winners, losers, results_summary = await self._run_tournament_round()
        
        if not winners:
            self.survivors = []
            self.losers_history.append(losers)
            return [TextContent(type="text", text=f"Tournament Round {self.current_round} failed. No winners.")]

        self.survivors = winners
        self.losers_history.append(losers)
        
        message = (
            f"Tournament Round {self.current_round} completed!\n"
            f"- Winners (advancing): {len(winners)} ({winners})\n"
            f"- Losers of this round: {len(losers)} ({losers})"
        )
        
        if is_final_round:
            if self.competition_data.get("evaluate_mode") == "pair" and self.competition_data.get("evaluation_data_path"):
                print("\n--- Running Final Comprehensive Evaluation on Global Test Data ---")
                
                if not self.competition_data.get('evaluation_samples'):
                    message += "\n- Warning: Cannot run final global evaluation because the evaluation dataset is missing."
                    print("Warning: evaluation_samples not found in competition_data. Skipping comprehensive evaluation.")
                else:
                    global_eval_df = pd.DataFrame(self.competition_data['evaluation_samples'])
                    all_time_results = []
                    output_dir = Path(self.competition_data['output_dir'])
                    evaluated_contestants = set()

                    for r in range(1, self.current_round + 1):
                        round_path = output_dir / f"round{r}"
                        algo_dir = round_path / "algorithms"
                        if not algo_dir.is_dir(): continue

                        for algo_file in sorted(algo_dir.glob("*_algorithm.py")):
                            contestant_id = algo_file.stem.replace("_algorithm", "")
                            formatted_name = f"r{r}_{contestant_id}"
                            if algo_file in evaluated_contestants: continue
                            
                            print(f"  - Evaluating historical contestant: {formatted_name} from {algo_file}")
                            scores = await self._evaluate_historical_contestant(contestant_id, r, global_eval_df)
                            all_time_results.append({"Contestant": formatted_name, **scores})
                            evaluated_contestants.add(algo_file)

                    if not all_time_results:
                        message += "\n- Warning: Could not generate the final comprehensive evaluation report."
                    else:
                        ranking_metric = 'RMSE' if self.competition_data['metric_mode'] == 'RMSE' else 'MAE'
                        final_df = pd.DataFrame(all_time_results).sort_values(by=ranking_metric)
                        final_csv_path = Path(self.competition_data['output_dir']) / f"round{self.current_round}" / "final_global_evaluation.csv"
                        final_df.to_csv(final_csv_path, index=False)
                        print(f"--- Comprehensive final evaluation saved to {final_csv_path} ---")
                        champion = final_df.iloc[0]['Contestant']
                        message += f"\n- Championship Result (Global Test): Contestant {champion} is the overall champion!"
                        print(f"--- Contestant {champion} is the overall champion based on comprehensive global evaluation. ---")

            test_results_message = await self._run_final_testing()
            if test_results_message:
                message += test_results_message
                print(test_results_message)

        if len(winners) > 1:
            print(f"\n--- Preparing Winners of Round {self.current_round} for the Next Round ---")
            await self._prepare_next_tournament_round(results_summary)
            message += f"\n- Winners have been revised for Round {self.current_round + 1}."
        else:
            if not is_final_round:
                message += f"\n- Final winner determined: {winners[0]}!"

        return [TextContent(type="text", text=message)]

    async def _advance_round_logic(self) -> list[TextContent]:
        if not self.survivors:
            return [TextContent(type="text", text="Cannot advance round: No survivors from the previous round.")]
        
        is_complete = len(self.survivors) <= 1 and self.current_round > 0

        if is_complete:
            winner = self.survivors[0] if self.survivors else "N/A"
            return [TextContent(type="text", text=f"Competition is complete. Final winner/survivor is {winner}. Use get_results to see final details.")]

        self.current_round += 1
        return await self._advance_round_tournament()

    async def get_competition_status(self) -> list[TextContent]:
        status = {
            "current_round_completed": self.current_round,
            "survivors_for_next_round_or_final": self.survivors,
            "num_survivors_for_next_round_or_final": len(self.survivors),
            "losers_history_per_round": self.losers_history,
            "is_gemini_initialized": self.gemini_model is not None,
            "competition_parameters": {
                k:v for k,v in self.competition_data.items()
                if k not in ['metadata_path', 'data_dir', 'revision_mode', 'ranking_rule', 'evaluation_samples']
            },
            "tournament_status": {
                "elo_ratings": {cid: round(elo) for cid, elo in sorted(self.elo_ratings.items(), key=lambda item: item[1], reverse=True)}
            },
            "competition_considered_complete": (self.current_round > 0 and len(self.survivors) <= 1)
        }
        return [TextContent(type="text", text=f"Competition Status:\n{json.dumps(status, indent=2)}")]


    async def get_results(self) -> list[TextContent]:
        if not self.competition_data or self.current_round == 0:
             return [TextContent(type="text", text="Competition has not started or no rounds have been run yet.")]

        final_round_completed = self.current_round
        metric_mode = self.competition_data.get("metric_mode", "MAE")
        ranking_metric = 'RMSE' if metric_mode == 'RMSE' else 'MAE'
        
        message_parts = [f"Competition Concluded after Round {final_round_completed}."]
        message_parts.append(f"Evolution Strategy Used: tournament")
        message_parts.append(f"Ranking Metric Used: {ranking_metric}")

        final_survivors = self.survivors
        final_round_dir = Path(self.competition_data['output_dir']) / f"round{final_round_completed}"
        results_csv_path = final_round_dir / f"results_round{final_round_completed}.csv"

        if len(final_survivors) > 1: 
            message_parts.append(f"Multiple survivors remain ({len(final_survivors)}): {final_survivors}.")
            if results_csv_path.exists():
                try:
                    df_final_results = pd.read_csv(results_csv_path, dtype={'Contestant': str})
                    survivor_results_df = df_final_results[df_final_results['Contestant'].isin(map(str,final_survivors))].sort_values(ranking_metric)
                    if not survivor_results_df.empty:
                        message_parts.append(f"Final scores for these survivors in Round {final_round_completed} (sorted by performance):\n{survivor_results_df.to_string(index=False)}")
                except Exception as e:
                    message_parts.append(f"Could not read or process final scores from {results_csv_path}: {e}")
            return [TextContent(type="text", text="\n".join(message_parts))]

        if not final_survivors and final_round_completed > 0: 
            message_parts.append("No survivors remain.")
            if results_csv_path.exists():
                 message_parts.append(f"Results from the last round ({final_round_completed}) can be found at: {results_csv_path}")
            return [TextContent(type="text", text="\n".join(message_parts))]

        if len(final_survivors) == 1: 
            winner = final_survivors[0]
            
            global_eval_path = final_round_dir / "final_global_evaluation.csv"
            if global_eval_path.exists():
                message_parts.insert(0, f"🏆 TOURNAMENT CHAMPION (GLOBAL EVALUATION) 🏆")
                df_global = pd.read_csv(global_eval_path)
                champion_row = df_global.iloc[0]
                winner = champion_row['Contestant']
                scores = {col: champion_row[col] for col in ['MAE', 'RMSE'] if col in df_global.columns}
                winner_scores_str = ", ".join([f"{k}: {v:.4f}" for k, v in scores.items()])
                message_parts.append(f"Final Champion: Algorithm {winner}")
                message_parts.append(f"Champion's Scores on Global Test Data: {winner_scores_str}")
                message_parts.append(f"\nFull global evaluation results:\n{df_global.to_string(index=False)}")
            else:
                message_parts.insert(0, f"🏆 COMPETITION WINNER: Algorithm {winner} 🏆")
                winner_scores_str = "N/A"
                if results_csv_path.exists():
                    try:
                        df_results = pd.read_csv(results_csv_path, dtype={'Contestant': str})
                        winner_row = df_results[df_results['Contestant'] == str(winner)]
                        if not winner_row.empty:
                            scores = {col: winner_row.iloc[0][col] for col in ['MAE', 'RMSE'] if col in df_results.columns}
                            winner_scores_str = ", ".join([f"{k}: {v:.4f}" for k, v in scores.items()])
                    except Exception as e:
                        message_parts.append(f"Could not read scores for winner {winner} from {results_csv_path}: {e}")
                message_parts.append(f"Winner's Final Scores (Round {final_round_completed}): {winner_scores_str}")

            final_elo = self._get_elo(winner)
            message_parts.append(f"Winner's Final Elo Rating: {final_elo:.0f}")
            
            final_test_path = final_round_dir / "final_test_results.csv"
            if final_test_path.exists():
                try:
                    df_test = pd.read_csv(final_test_path)
                    message_parts.append(f"\n--- Final Test Results ---\n{df_test.to_string(index=False)}")
                except Exception as e:
                    message_parts.append(f"Could not read final test results file: {e}")

            return [TextContent(type="text", text="\n".join(message_parts))]
        
        return [TextContent(type="text", text="Competition results are inconclusive or state is unexpected. Please check status.")]


    async def run_full_competition_logic(self, n_total_examples: float, m_contestants: int,
                                         metadata_path: str = "../dataset/rawdata.xlsx",
                                         data_dir: str = "../dataset/rawdata/",
                                         output_dir: str = "./competition_results",
                                         max_rounds: Optional[int] = None,
                                         evolution_strategy: str = "tournament",
                                         rand_mode: str = "fully_random",
                                         ranking_rule_params: Optional[Dict[str, Any]] = None,
                                         prompt_mode: str = "default",
                                         train_data: Optional[str] = None,
                                         evaluation_data: Optional[str] = None,
                                         test_data: Optional[str] = None,
                                         metric_mode: str = "MAE",
                                         evaluate_mode: str = "normal"
                                         ) -> list[TextContent]:
        all_messages: List[TextContent] = []
        effective_max_rounds = max_rounds if max_rounds is not None else self.default_max_rounds
        effective_ranking_rule_params = ranking_rule_params if ranking_rule_params is not None else {}
    
        all_messages.append(TextContent(type="text", text=f"--- Starting Full Competition (Max Rounds: {effective_max_rounds}, Strategy: {evolution_strategy}) ---"))
    
        start_messages = await self.start_competition(
            n_total_examples=n_total_examples,
            m_contestants=m_contestants,
            metadata_path=metadata_path,
            data_dir=data_dir,
            output_dir=output_dir,
            evolution_strategy=evolution_strategy,
            rand_mode=rand_mode,
            ranking_rule_params=effective_ranking_rule_params,
            prompt_mode=prompt_mode,
            train_data=train_data,
            evaluation_data=evaluation_data,
            test_data=test_data,
            metric_mode=metric_mode,
            evaluate_mode=evaluate_mode
        )
        all_messages.extend(start_messages)
    
        if any("Error:" in msg.text for msg in start_messages) or not self.survivors:
            all_messages.append(TextContent(type="text", text="Full competition aborted after start due to error or no initial survivors."))
            all_messages.extend(await self.get_results())
            return all_messages
    
        def is_competition_over():
            return len(self.survivors) <= 1

        while not is_competition_over() and self.current_round < effective_max_rounds:
            all_messages.append(TextContent(type="text", text=f"Advancing from round {self.current_round}. Current Survivors: {self.survivors}"))
            
            advance_messages = await self._advance_round_logic()
            all_messages.extend(advance_messages)
            
            if any("Error:" in msg.text for msg in advance_messages) or not self.survivors:
                all_messages.append(TextContent(type="text", text=f"Full competition aborted during round {self.current_round} due to error or no survivors."))
                break

        if is_competition_over():
            all_messages.append(TextContent(type="text", text=f"Competition loop ended: A final winner was determined after round {self.current_round}."))
        elif self.current_round >= effective_max_rounds:
            all_messages.append(TextContent(type="text", text=f"Competition loop ended: Reached max_rounds ({effective_max_rounds})."))
    
        all_messages.append(TextContent(type="text", text="\n--- Fetching Final Competition Results ---"))
        results_messages = await self.get_results()
        all_messages.extend(results_messages)
        return all_messages

    async def run_multiple_competitions(self, competitions: List[Dict[str, Any]]) -> list[TextContent]:
        all_messages: List[TextContent] = []
        base_output_dir = Path("./competition_results")
        
        if not competitions:
            return [TextContent(type="text", text="Error: No competitions provided.")]

        rules = [c.get("evolution_strategy", self.default_evolution_strategy) for c in competitions]
        all_rules_are_same = len(set(rules)) == 1

        shared_train_data = None
        shared_eval_data = None
        first_run = True

        rule_counts: Dict[str, int] = {}

        for i, config in enumerate(competitions):
            config['evolution_strategy'] = 'tournament'
            rule_name = 'tournament'
            
            if all_rules_are_same:
                rule_counts[rule_name] = rule_counts.get(rule_name, 0) + 1
                output_dir_name = f"{rule_counts[rule_name]}_{rule_name}_results"
            else:
                output_dir_name = f"{rule_name}_results"
            
            output_dir = base_output_dir / output_dir_name
            config['output_dir'] = str(output_dir)

            all_messages.append(TextContent(type="text", text=f"\n\n{'='*80}\n--- Starting Competition {i+1}/{len(competitions)}: Rule '{rule_name}' ---\n{'='*80}"))
            
            if not all_rules_are_same:
                if first_run:
                    run_messages = await self.run_full_competition_logic(**config)
                    all_messages.extend(run_messages)
                    
                    first_output_path = Path(config['output_dir'])
                    shared_train_data = first_output_path / "training_data_split.csv"
                    shared_eval_data = first_output_path / "evaluation_data_split.csv"
                    
                    if not shared_train_data.exists() or not shared_eval_data.exists():
                        msg = "Error: First competition did not generate split data files. Cannot proceed with subsequent competitions."
                        all_messages.append(TextContent(type="text", text=msg))
                        return all_messages
                    first_run = False
                else:
                    config['train_data'] = str(shared_train_data)
                    config['evaluation_data'] = str(shared_eval_data)
                    run_messages = await self.run_full_competition_logic(**config)
                    all_messages.extend(run_messages)
            else:
                run_messages = await self.run_full_competition_logic(**config)
                all_messages.extend(run_messages)

        all_messages.append(TextContent(type="text", text=f"\n\n{'='*80}\n--- All Competitions Finished ---\n{'='*80}"))
        return all_messages


async def main():
    try:
        q_test = GPTChatbot() 
        if not hasattr(q_test, 'add_history') or not callable(getattr(q_test, 'add_history')):
            print("WARNING: Your GPTChatbot class does not seem to have an 'add_history' method. The history loading feature may not work as expected.")
        del q_test 
    except NameError:
        print("ERROR: GPTChatbot class not found. Make sure gpt3_mlx.py is in your PYTHONPATH or the same directory.")
        return
    except Exception as e:
        print(f"ERROR during GPTChatbot pre-check: {e}.")

    competition = AlgorithmCompetitionMCP()
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await competition.server.run(
            read_stream,
            write_stream,
            competition.server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())