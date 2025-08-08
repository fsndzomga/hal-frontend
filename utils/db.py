import json
from pathlib import Path
import sqlite3
import pickle
from functools import lru_cache
import threading
import pandas as pd
import ast
from scipy import stats
import yaml
import numpy as np
import re


# Define column schemas
PARSED_RESULTS_COLUMNS = {
    'benchmark_name': 'TEXT',
    'agent_name': 'TEXT', 
    'date': 'TEXT',
    'run_id': 'TEXT',
    'successful_tasks': 'TEXT',
    'failed_tasks': 'TEXT',
    'total_cost': 'REAL',
    'accuracy': 'REAL',
    'average_score': 'REAL', #assistantbench
    'exact_matches': 'REAL', #assistantbench
    'answer_rate': 'REAL', #assistantbench
    'total_tasks': 'REAL', #mind2web
    'accuracy_easy': 'REAL', #mind2web
    'total_easy': 'REAL', #mind2web
    'accuracy_medium': 'REAL', #mind2web
    'total_medium': 'REAL', #mind2web
    'accuracy_hard': 'REAL', #mind2web
    'total_hard': 'REAL', #mind2web
    'average_correctness': 'REAL', #colbench_backend_frontend
    'subtask_accuracy': 'REAL', #scicode
    'codebert_score': 'REAL', #scienceagentbench
    'success_rate': 'REAL', #scienceagentbench
    'valid_program_rate': 'REAL', #scienceagentbench
    'trace_stem': 'TEXT', # Stores file.stem from the JSON
    'precision': 'REAL',
    'recall': 'REAL',
    'f1_score': 'REAL',
    'auc': 'REAL',
    'overall_score': 'REAL',
    'vectorization_score': 'REAL',
    'fathomnet_score': 'REAL',
    'feedback_score': 'REAL',
    'house_price_score': 'REAL',
    'spaceship_titanic_score': 'REAL',
    'amp_parkinsons_disease_progression_prediction_score': 'REAL',
    'cifar10_score': 'REAL',
    'imdb_score': 'REAL',
    'level_1_accuracy': 'REAL',
    'level_2_accuracy': 'REAL',
    'level_3_accuracy': 'REAL',
    'task_goal_completion': 'REAL',  # New column
    'scenario_goal_completion': 'REAL',  # New column
    'combined_scorer_inspect_evals_avg_refusals': 'REAL',
    'combined_scorer_inspect_evals_avg_score_non_refusals': 'REAL',
    'accuracy_ci': 'TEXT',  # Using TEXT since it stores formatted strings like "-0.123/+0.456"
    'cost_ci': 'TEXT',
    'model_name': 'TEXT',
}

# Define which columns should be included in aggregation and how
AGGREGATION_RULES = {
    'date': 'first',
    'total_cost': 'mean',
    'accuracy': 'mean',
    'average_score': 'mean',
    'exact_matches': 'mean',
    'answer_rate': 'mean',
    'total_tasks': 'mean',
    'accuracy_easy': 'mean',
    'total_easy': 'mean',
    'accuracy_medium': 'mean',
    'total_medium': 'mean',
    'accuracy_hard': 'mean',
    'total_hard': 'mean',
    'average_correctness': 'mean',
    'subtask_accuracy': 'mean',
    'codebert_score': 'mean',
    'success_rate': 'mean',
    'valid_program_rate': 'mean',
    'precision': 'mean',
    'recall': 'mean',
    'f1_score': 'mean',
    'auc': 'mean',
    'overall_score': 'mean',
    'vectorization_score': 'mean',
    'fathomnet_score': 'mean',
    'feedback_score': 'mean',
    'house_price_score': 'mean',
    'spaceship_titanic_score': 'mean',
    'amp_parkinsons_disease_progression_prediction_score': 'mean',
    'cifar10_score': 'mean',
    'imdb_score': 'mean',
    'level_1_accuracy': 'mean',
    'level_2_accuracy': 'mean',
    'level_3_accuracy': 'mean',
    'task_goal_completion': 'mean',
    'scenario_goal_completion': 'mean',
    'combined_scorer_inspect_evals_avg_refusals': 'mean',
    'combined_scorer_inspect_evals_avg_score_non_refusals': 'mean',
    'Verified': 'first',
    'Runs': 'first',
    'Traces': 'first',
    'accuracy_ci': 'first',
    'cost_ci': 'first',
    'run_id': 'first',
    'trace_stem': 'first',
    'model_name': 'first',
}

# Define column display names
COLUMN_DISPLAY_NAMES = {
    'agent_name': 'Agent Name',
    'url': 'URL',
    'date': 'Date',
    'total_cost': 'Total Cost',
    'accuracy': 'Accuracy',
    'average_score': 'Average Score',
    'exact_matches': 'Exact Matches',
    'answer_rate': 'Answer Rate',
    'total_tasks': 'Total Tasks',
    'accuracy_easy': 'Accuracy (Easy)',
    'total_easy': 'Total (Easy)',
    'accuracy_medium': 'Accuracy (Medium)',
    'total_medium': 'Total (Medium)',
    'accuracy_hard': 'Accuracy (Hard)',
    'total_hard': 'Total (Hard)',
    'average_correctness': 'Average Correctness',
    'subtask_accuracy': 'Subtask Accuracy',
    'codebert_score': 'CodeBERT Score',
    'success_rate': 'Success Rate',
    'valid_program_rate': 'Valid Program Rate',
    'precision': 'Precision',
    'recall': 'Recall',
    'f1_score': 'F1 Score',
    'auc': 'AUC',
    'overall_score': 'Overall Score',
    'vectorization_score': 'Vectorization Score',
    'fathomnet_score': 'Fathomnet Score',
    'feedback_score': 'Feedback Score',
    'house_price_score': 'House Price Score',
    'spaceship_titanic_score': 'Spaceship Titanic Score',
    'amp_parkinsons_disease_progression_prediction_score': 'AMP Parkinsons Disease Progression Prediction Score',
    'cifar10_score': 'CIFAR10 Score',
    'imdb_score': 'IMDB Score',
    'level_1_accuracy': 'Level 1 Accuracy',
    'level_2_accuracy': 'Level 2 Accuracy',
    'level_3_accuracy': 'Level 3 Accuracy',
    'task_goal_completion': 'Task Goal Completion',
    'scenario_goal_completion': 'Scenario Goal Completion',
    'accuracy_ci': 'Accuracy CI',
    'cost_ci': 'Total Cost CI',
    'combined_scorer_inspect_evals_avg_refusals': 'Refusals',
    'combined_scorer_inspect_evals_avg_score_non_refusals': 'Non-Refusal Harm Score',
    'trace_stem': 'Trace Stem',
    'model_name': 'Model Name',
}

# DEFAULT_PRICING = {
#     "text-embedding-3-small": {"prompt_tokens": 0.02, "completion_tokens": 0},
#     "text-embedding-3-large": {"prompt_tokens": 0.13, "completion_tokens": 0},
#     "gpt-4o-2024-05-13": {"prompt_tokens": 2.5, "completion_tokens": 10},
#     "gpt-4o-2024-08-06": {"prompt_tokens": 2.5, "completion_tokens": 10},
#     "gpt-4o-2024-11-20": {"prompt_tokens": 2.5, "completion_tokens": 10},
#     "gpt-3.5-turbo-0125": {"prompt_tokens": 0.5, "completion_tokens": 1.5},
#     "gpt-3.5-turbo": {"prompt_tokens": 0.5, "completion_tokens": 1.5},
#     "gpt-4-turbo-2024-04-09": {"prompt_tokens": 10, "completion_tokens": 30},
#     "gpt-4-turbo": {"prompt_tokens": 10, "completion_tokens": 30},
#     "gpt-4o-mini-2024-07-18": {"prompt_tokens": 0.15, "completion_tokens": 0.6},
#     "gpt-4-turbo-2024-04-09": {"prompt_tokens": 10, "completion_tokens": 30},
#     "o1-2024-12-17": {"prompt_tokens": 15, "completion_tokens": 60},
#     "meta-llama/Meta-Llama-3.1-8B-Instruct": {"prompt_tokens": 0.18, "completion_tokens": 0.18},
#     "meta-llama/Meta-Llama-3.1-70B-Instruct": {"prompt_tokens": 0.88, "completion_tokens": 0.88},
#     "meta-llama/Meta-Llama-3.1-405B-Instruct": {"prompt_tokens": 5, "completion_tokens": 15},
#     "meta-llama/Llama-3-70b-chat-hf": {"prompt_tokens": 0.88, "completion_tokens": 0.88},
#     "deepseek-ai/deepseek-coder-33b-instruct": {"prompt_tokens": 0.18, "completion_tokens": 0.18}, # these are set to the llama 8n prices
#     "gpt-4o": {"prompt_tokens": 2.5, "completion_tokens": 10},
#     "gpt-4.5-preview-2025-02-27": {"prompt_tokens": 75, "completion_tokens": 150},
#     "o1-mini-2024-09-12": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
#     "o1-preview-2024-09-12": {"prompt_tokens": 15, "completion_tokens": 60},
#     "o3-mini-2025-01-14": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
#     "o3-mini-2025-01-31": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
#     "claude-3-5-sonnet-20240620": {"prompt_tokens": 3, "completion_tokens": 15},
#     "claude-3-5-sonnet-20241022": {"prompt_tokens": 3, "completion_tokens": 15},
#     "us.anthropic.claude-3-5-sonnet-20240620-v1:0": {"prompt_tokens": 3, "completion_tokens": 15},
#     "us.anthropic.claude-3-5-sonnet-20241022-v2:0": {"prompt_tokens": 3, "completion_tokens": 15},
#     "claude-3-5-haiku-20241022": {"prompt_tokens": 0.8, "completion_tokens": 4},
#     "us.anthropic.claude-3-5-haiku-20241022-v1:0": {"prompt_tokens": 0.8, "completion_tokens": 4},
#     "openai/gpt-4o-2024-11-20": {"prompt_tokens": 2.5, "completion_tokens": 10},
#     "openai/gpt-4o-2024-08-06": {"prompt_tokens": 2.5, "completion_tokens": 10},
#     "openai/gpt-4o-mini-2024-07-18": {"prompt_tokens": 0.15, "completion_tokens": 0.6},
#     "openai/gpt-4.5-preview-2025-02-27": {"prompt_tokens": 75, "completion_tokens": 150},
#     "openai/o1-mini-2024-09-12": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
#     "openai/o1-preview-2024-09-12": {"prompt_tokens": 15, "completion_tokens": 60},
#     "openai/o1-2024-12-17": {"prompt_tokens": 15, "completion_tokens": 60},
#     "openai/o3-mini-2025-01-14": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
#     "openai/o3-mini-2025-01-31": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
#     "anthropic/claude-3-5-sonnet-20240620": {"prompt_tokens": 3, "completion_tokens": 15},
#     "anthropic/claude-3-5-sonnet-20241022": {"prompt_tokens": 3, "completion_tokens": 15},
#     "google/gemini-1.5-pro": {"prompt_tokens": 1.25, "completion_tokens": 5},
#     "google/gemini-1.5-flash": {"prompt_tokens": 0.075, "completion_tokens": 0.3},
#     "together/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": {"prompt_tokens": 3.5, "completion_tokens": 3.5},
#     "Meta-Llama-3.3-70B-Instruct-Turbo": {"prompt_tokens": 0.88, "completion_tokens": 0.88},
#     "us.meta.llama3-3-70b-instruct-v1:0": {"prompt_tokens": 0.88, "completion_tokens": 0.88},
#     "Meta-Llama-3.1-405B-Instruct-Turbo": {"prompt_tokens": 3.5, "completion_tokens": 3.5},
#     "together/meta-llama/Meta-Llama-3.1-70B-Instruct": {"prompt_tokens": 0.88, "completion_tokens": 0.88},
#     "claude-3-7-sonnet-20250219": {"prompt_tokens": 3, "completion_tokens": 15},
#     "anthropic/claude-3-7-sonnet-20250219": {"prompt_tokens": 3, "completion_tokens": 15},
#     "o4-mini-2025-04-16": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
#     "o3-2025-04-16":{"prompt_tokens": 2, "completion_tokens": 8},
#     "gpt-4.1-2025-04-14":{"prompt_tokens": 2, "completion_tokens": 8},
#     "claude-3-7-sonnet-2025-02-19": {"prompt_tokens": 3, "completion_tokens": 15},
#     "deepseek-ai/DeepSeek-V3": {"prompt_tokens": 0.27, "completion_tokens": 1.1},
#     "gemini-2.0-flash": {"prompt_tokens": 0.1, "completion_tokens": 0.4},
#     "deepseek-ai/DeepSeek-R1":{"prompt_tokens": 0.55, "completion_tokens": 2.19},
# }

DEFAULT_PRICING = {
    "Text-Embedding-3 Small": {"prompt_tokens": 0.02, "completion_tokens": 0},
    "text-embedding-3-large": {"prompt_tokens": 0.13, "completion_tokens": 0},
    "GPT-4o (May 2024)": {"prompt_tokens": 2.5, "completion_tokens": 10},
    "GPT-4o (August 2024)": {"prompt_tokens": 2.5, "completion_tokens": 10},
    "GPT-4o (November 2024)": {"prompt_tokens": 2.5, "completion_tokens": 10},
    "gpt-3.5-turbo-0125": {"prompt_tokens": 0.5, "completion_tokens": 1.5},
    "gpt-3.5-turbo": {"prompt_tokens": 0.5, "completion_tokens": 1.5},
    "gpt-4-turbo-2024-04-09": {"prompt_tokens": 10, "completion_tokens": 30},
    "gpt-4-turbo": {"prompt_tokens": 10, "completion_tokens": 30},
    "gpt-4o-mini-2024-07-18": {"prompt_tokens": 0.15, "completion_tokens": 0.6},
    "o1 Medium (December 2024)": {"prompt_tokens": 15, "completion_tokens": 60},
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {"prompt_tokens": 0.18, "completion_tokens": 0.18},
    "meta-llama/Meta-Llama-3.1-70B-Instruct": {"prompt_tokens": 0.88, "completion_tokens": 0.88},
    "meta-llama/Meta-Llama-3.1-405B-Instruct": {"prompt_tokens": 5, "completion_tokens": 15},
    "meta-llama/Llama-3-70b-chat-hf": {"prompt_tokens": 0.88, "completion_tokens": 0.88},
    "deepseek-ai/deepseek-coder-33b-instruct": {"prompt_tokens": 0.18, "completion_tokens": 0.18},
    "o1-mini Medium (September 2024)": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
    "o1-preview-2024-09-12": {"prompt_tokens": 15, "completion_tokens": 60},
    "o3-mini Medium (January 2025)": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
    "claude-3-5-sonnet-20240620": {"prompt_tokens": 3, "completion_tokens": 15},
    "claude-3-5-sonnet-20241022": {"prompt_tokens": 3, "completion_tokens": 15},
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0": {"prompt_tokens": 3, "completion_tokens": 15},
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0": {"prompt_tokens": 3, "completion_tokens": 15},
    "claude-3-5-haiku-20241022": {"prompt_tokens": 0.8, "completion_tokens": 4},
    "us.anthropic.claude-3-5-haiku-20241022-v1:0": {"prompt_tokens": 0.8, "completion_tokens": 4},
    "o4-mini Medium (April 2025)": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
    "o3 Medium (April 2025)": {"prompt_tokens": 2, "completion_tokens": 8},
    "GPT-4.1 (April 2025)": {"prompt_tokens": 2, "completion_tokens": 8},
    "Claude-3.7 Sonnet (February 2025)": {"prompt_tokens": 3, "completion_tokens": 15},
    "DeepSeek V3": {"prompt_tokens": 0.27, "completion_tokens": 1.1},
    "Gemini 2.0 Flash": {"prompt_tokens": 0.1, "completion_tokens": 0.4},
    "DeepSeek R1": {"prompt_tokens": 0.55, "completion_tokens": 2.19},
    "o4-mini Low (April 2025)": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
    "Claude-3.7 Sonnet Low (February 2025)": {"prompt_tokens": 3, "completion_tokens": 15},
    "o4-mini High (April 2025)": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
    "o3 Low (April 2025)": {"prompt_tokens": 2, "completion_tokens": 8},
    "o3-mini Low (January 2025)": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
    "o3-mini High (January 2025)": {"prompt_tokens": 1.1, "completion_tokens": 4.4},
    "Gemini 2.5 Pro Preview (March 2025)": {"prompt_tokens": 1.25, "completion_tokens": 5},
    "Claude-3.7 Sonnet High (February 2025)": {"prompt_tokens": 3, "completion_tokens": 15},
    "gpt-4.5-preview-2025-02-27": {"prompt_tokens": 75, "completion_tokens": 150},
    "Claude Opus 4 (May 2025)": {"prompt_tokens": 15, "completion_tokens": 75},
    "Claude Opus 4 High (May 2025)": {"prompt_tokens": 15, "completion_tokens": 75},
    "Claude Opus 4.1 High (August 2025)": {"prompt_tokens": 15, "completion_tokens": 75},
    "Claude Opus 4.1 (August 2025)": {"prompt_tokens": 15, "completion_tokens": 75},
    "GPT-5 High": {"prompt_tokens": 1.25, "completion_tokens": 10},
    "GPT-5 Medium": {"prompt_tokens": 1.25, "completion_tokens": 10},
    "GPT-5 Low": {"prompt_tokens": 1.25, "completion_tokens": 10},
    "GPT-OSS-120B": {"prompt_tokens": 0.15, "completion_tokens": 0.6},
    "GPT-OSS-120B High": {"prompt_tokens": 0.15, "completion_tokens": 0.6},
}


MODEL_MAPPING = [
    ("o4-mini-2025-04-16", "o4-mini Medium (April 2025)", "o4-mini-2025-04-16"),
    ("gpt-4.1-2025-04-14", "GPT-4.1 (April 2025)", "gpt-4.1-2025-04-14"),
    ("o4-mini-2025-04-16 low", "o4-mini Low (April 2025)", "o4-mini-2025-04-16"),
    ("claude-3-7-sonnet-20250219 low", "Claude-3.7 Sonnet Low (February 2025)", "claude-3-7-sonnet-20250219"),
    ("o4-mini-2025-04-16 high", "o4-mini High (April 2025)", "o4-mini-2025-04-16"),
    ("deepseek-ai/DeepSeek-V3", "DeepSeek V3", "deepseek-ai/DeepSeek-V3"),
    ("claude-3-7-sonnet-20250219", "Claude-3.7 Sonnet (February 2025)", "claude-3-7-sonnet-20250219"),
    ("claude-opus-4-20250514", "Claude Opus 4 (May 2025)", "claude-opus-4-20250514"),
    ("Claude-Opus 4 High (May 2025)", "Claude Opus 4 High (May 2025)", "claude-opus-4-20250514"),
    ("gemini-2.0-flash", "Gemini 2.0 Flash", "gemini-2.0-flash"),
    ("claude-3-7-sonnet-20250219 high", "Claude-3.7 Sonnet High (February 2025)", "claude-3-7-sonnet-20250219"),
    ("o3-2025-04-16", "o3 Medium (April 2025)", "o3-2025-04-16"),
    ("claude-3-7-sonnet-2025-02-19", "Claude-3.7 Sonnet (February 2025)", "claude-3-7-sonnet-20250219"),
    ("claude-opus-4-20250514 high", "Claude Opus 4 High (May 2025)", "claude-opus-4-20250514"),
    ("openrouter/anthropic/claude-opus-4.1", "Claude Opus 4.1 (August 2025)", "openrouter/anthropic/claude-opus-4.1"),
    ("deepseek-ai/DeepSeek-R1", "DeepSeek R1", "deepseek-ai/DeepSeek-R1"),
    ("DeepSeek-R1", "DeepSeek R1", "deepseek-ai/DeepSeek-R1"),
    ("claude-3-7-sonnet-2025-02-19 low", "Claude-3.7 Sonnet Low (February 2025)", "claude-3-7-sonnet-20250219"),
    ("DeepSeek-V3", "DeepSeek V3", "deepseek-ai/DeepSeek-V3"),
    ("together_ai/deepseek-ai/DeepSeek-V3", "DeepSeek V3", "deepseek-ai/DeepSeek-V3"),
    ("together_ai/deepseek-ai/DeepSeek-R1", "DeepSeek R1", "deepseek-ai/DeepSeek-R1"),
    ("openrouter/anthropic/claude-opus-4.1 high", "Claude Opus 4.1 High (August 2025)", "openrouter/anthropic/claude-opus-4.1-high"),
    ("Sonnet3.7", "Claude-3.7 Sonnet (February 2025)", "claude-3-7-sonnet-20250219"),
    ("O4-mini-high", "o4-mini High (April 2025)", "o4-mini-2025-04-16"),
    ("o4-mini-high", "o4-mini High (April 2025)", "o4-mini-2025-04-16"),
    ("GPT4.1", "GPT-4.1 (April 2025)", "gpt-4.1-2025-04-14"),
    ("O3-low", "o3 Low (April 2025)", "o3-2025-04-16"),
    ("o3-low", "o3 Low (April 2025)", "o3-2025-04-16"),
    ("Sonnet 3.7", "Claude-3.7 Sonnet (February 2025)", "claude-3-7-sonnet-20250219"),
    ("o4-mini-low", "o4-mini Low (April 2025)", "o4-mini-2025-04-16"),
    ("o4-mini-2025-04-16 medium", "o4-mini Medium (April 2025)", "o4-mini-2025-04-16"),
    ("o3-mini-2025-01-31 low", "o3-mini Low (January 2025)", "o3-mini-2025-01-31"),
    ("o3-mini-2025-01-31 medium", "o3-mini Medium (January 2025)", "o3-mini-2025-01-31"),
    ("claude-3-7-sonnet-2025-02-19 high", "Claude-3.7 Sonnet High (February 2025)", "claude-3-7-sonnet-20250219"),
    ("gemini/gemini-2.5-pro-preview-03-25", "Gemini 2.5 Pro Preview (March 2025)", "gemini-2.5-pro-preview-03-25"),
    ("o3-mini-2025-01-31 high", "o3-mini High (January 2025)", "o3-mini-2025-01-31"),
    ("claude-3-7-sonnet-20250219_thinking_high_4096", "Claude-3.7 Sonnet High (February 2025)", "claude-3-7-sonnet-20250219"),
    ("gemini-2.5-pro-preview-03-25", "Gemini 2.5 Pro Preview (March 2025)", "gemini-2.5-pro-preview-03-25"),
    ("o4-mini-2025-04-16_high_reasoning_effort", "o4-mini High (April 2025)", "o4-mini-2025-04-16"),
    ("o4-mini-2025-04-16_low_reasoning_effort", "o4-mini Low (April 2025)", "o4-mini-2025-04-16"),
    ("gemini-2.5-pro-preview", "Gemini 2.5 Pro Preview (March 2025)", "gemini-2.5-pro-preview-03-25"),
    ("o3-mini", "o3-mini Medium (January 2025)", "o3-mini-2025-01-31"),
    ("gpt-4o-2024-11-20", "GPT-4o (November 2024)", "gpt-4o-2024-11-20"),
    ("gpt-4o", "GPT-4o (August 2024)", "gpt-4o-2024-08-06"),
    ("o1", "o1 Medium (December 2024)", "o1-2024-12-17"),
    ("gpt-4.1", "GPT-4.1 (April 2025)", "gpt-4.1-2025-04-14"),
    ("o3-mini-2025-01-31", "o3-mini Medium (January 2025)", "o3-mini-2025-01-31"),
    ("o3-2025-04-03", "o3 Medium (April 2025)", "o3-2025-04-03"),
    ("o3-2025-04-16 medium", "o3 Medium (April 2025)", "o3-2025-04-16"),
    ("o3-2025-04-16 low", "o3 Low (April 2025)", "o3-2025-04-16"),
    ("openai/o3-2025-04-16 medium", "o3 Medium (April 2025)", "o3-2025-04-16"),
    ("o3-mini low", "o3-mini Low (January 2025)", "o3-mini-2025-01-31"),
    ("o3-mini high", "o3-mini High (January 2025)", "o3-mini-2025-01-31"),
    ("gpt-4o-2024-08-06", "GPT-4o (August 2024)", "gpt-4o-2024-08-06"),
    ("o1-2024-12-17", "o1 Medium (December 2024)", "o1-2024-12-17"),
    ("text-embedding-3-small", "Text-Embedding-3 Small", "text-embedding-3-small"),
    ("gpt-5 high", "GPT-5 Medium", "gpt-5-high"), # Temporary hack because of errors in taubench
    ("gpt-5 minimal", "GPT-5 Medium", "gpt-5-high"), # Temporary hack because of errors in taubench
    ("gpt-5", "GPT-5 Medium", "gpt-5"),
    ("gpt-5 low", "GPT-5 Low", "gpt-5-low"),
    ("gpt-5 medium", "GPT-5 Medium", "gpt-5-medium"),
    ("gemini/gemini-2.0-flash", "Gemini 2.0 Flash", "gemini/gemini-2.0-flash"),
    ("openai/o3-mini-2025-01-31 low", "o3-mini Low (January 2025)", "o3-mini-2025-01-31"),
    ("openai/gpt-5-2025-08-07", "GPT-5 Medium", "gpt-5"),
    ("openrouter/openai/gpt-oss-120b high", "GPT-OSS-120B High", "openrouter/openai/gpt-oss-120b"),
    ("openrouter/openai/gpt-oss-120b", "GPT-OSS-120B", "openrouter/openai/gpt-oss-120b"),
    ("openai/gpt-oss-120b high", "GPT-OSS-120B High", "openai/gpt-oss-120b"),
    ("openai/gpt-oss-120b", "GPT-OSS-120B", "openai/gpt-oss-120b"),
    ("claude-opus-4-1-20250805", "Claude Opus 4.1 (August 2025)", "claude-opus-4.1-20250805"),
    ("claude-opus-4-1-20250514", "Claude Opus 4.1 (August 2025)", "claude-opus-4.1-20250514"),
    ("claude-opus-4-1-20250514 high", "Claude Opus 4.1 High (August 2025)", "claude-opus-4.1-20250514"),
    ("claude-opus-4", "Claude Opus 4 (May 2025)", "claude-opus-4"),
    ("anthropic/claude-opus-4.1", "Claude Opus 4.1 (August 2025)", "anthropic/claude-opus-4.1"),
    ("anthropic/claude-opus-4", "Claude Opus 4 (May 2025)", "anthropic/claude-opus-4"),
    ("gpt-5-2025-08-07", "GPT-5 Medium", "gpt-5"),
    ("claude-opus-4.1-20250514 high", "Claude Opus 4.1 High (August 2025)", "claude-opus-4.1-20250514"),
    ("claude-opus-4.1-20250514", "Claude Opus 4.1 (August 2025)", "claude-opus-4.1-20250514"),
]

MODELS_TO_SKIP = [
'Gemini 2.5 Pro Preview (March 2025)',
'o1 Medium (December 2024)',
'o3-mini Low (January 2025)',
'o3-mini Medium (January 2025)',
'o3-mini High (January 2025)',
'GPT-4o (November 2024)',
'GPT-4o (August 2024)',
'o3 Low (April 2025)',
'Claude-3.7 Sonnet Low (February 2025)',
'o4-mini Medium (April 2025)',
]

class TracePreprocessor:
    def __init__(self, db_dir='preprocessed_traces'):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(exist_ok=True)
        self.local = threading.local()
        self.connections = {}
    
    @staticmethod
    def get_fallback_accuracy(results):
        if 'accuracy' in results and results['accuracy'] is not None:
            return results['accuracy']
        elif 'average_correctness' in results and results['average_correctness'] is not None:
            return results['average_correctness']
        elif 'success_rate' in results and results['success_rate'] is not None:
            return results['success_rate']
        elif 'average_score' in results and results['average_score'] is not None:
            return results['average_score']
        else:
            return None
    
    @staticmethod
    def get_model_show_name(model_name):
        for mapping in MODEL_MAPPING:
            if model_name == mapping[0]:
                return mapping[1]
        return model_name
    
    def get_all_runs(self):
        records = []
        for db_file in self.db_dir.glob('*.db'):
            benchmark_name = db_file.stem
            with self.get_conn(benchmark_name) as conn:
                # Try to get all runs from parsed_results
                df = pd.read_sql_query(
                    "SELECT benchmark_name, agent_name, model_name, run_id FROM parsed_results",
                    conn
                )
                if not df.empty:
                    records.append(df)
        if not records:
            return pd.DataFrame(columns=['benchmark_name', 'agent_name', 'model_name', 'run_id'])
        all_df = pd.concat(records, ignore_index=True)
        return all_df
    
    def get_model_benchmark_accuracies(self):
        EXCLUDE_BENCHMARKS = [
            'colbench_backend_programming',
            'colbench_frontend_design',
            'scienceagentbench',
        ]

        BENCHMARK_ALIAS = {
            "usaco":                 "USACO",
            "taubench_airline":      "TAU-bench Airline",
            "swebench_verified_mini":"SWE-bench Verified Mini",
            "scicode":               "Scicode",
            "online_mind2web":       "Online Mind2Web",
            "gaia":                  "GAIA",
            "corebench_hard":        "CORE-Bench Hard",
        }

        MODEL_ALIAS = {
            "Claude-3.7 Sonnet High (February 2025)":   "Claude-3.7 High Feb 25",
            "Claude-3.7 Sonnet (February 2025)":   "Claude-3.7 Feb 25",
            "DeepSeek R1":                             "DeepSeek R1",
            "DeepSeek V3":                             "DeepSeek V3",
            "GPT-4.1 (April 2025)":                    "GPT-4.1 Apr 25",
            "GPT-4o (August 2024)":                    "GPT-4o Aug 24",
            "GPT-4o (November 2024)":                  "GPT-4o Nov 24",
            "Gemini 2.0 Flash":                        "Gemini 2.0 Flash",
            "Gemini 2.5 Pro Preview (March 2025)":     "Gemini 2.5 Mar 5",
            "o1 Medium (December 2024)":               "o1 Med Dec 24",
            "o3 Medium (April 2025)":                  "o3 Med Apr 25",
            "o3-mini Low (January 2025)":              "o3-mini Low Jan 25",
            "o4-mini Medium (April 2025)":             "o4-mini Med Apr 25",
            "o4-mini Low (April 2025)":             "o4-mini Low Apr 25",
            "o4-mini High (April 2025)":             "o4-mini High Apr 25",
            "Claude Opus 4 High (May 2025)":         "Claude Opus 4 High May 25",
            "Claude Opus 4 (May 2025)":              "Claude Opus 4 May 25",
            "Claude Opus 4.1 (August 2025)":         "Claude Opus 4.1 Aug 25",
            "Claude Opus 4.1 High (August 2025)":    "Claude Opus 4.1 High Aug 25",
        }

        # MODEL_ALIAS = {
        #     "Claude-3.7 Sonnet Low (February 2025)":   "Claude-3.7 Sonnet Low (February 2025)",
        #     "DeepSeek R1":                             "DeepSeek R1",
        #     "DeepSeek V3":                             "DeepSeek V3",
        #     "GPT-4.1 (April 2025)":                    "GPT-4.1 (April 2025)",
        #     "GPT-4o (August 2024)":                    "GPT-4o (August 2024)",
        #     "GPT-4o (November 2024)":                  "GPT-4o (November 2024)",
        #     "Gemini 2.0 Flash":                        "Gemini 2.0 Flash",
        #     "Gemini 2.5 Pro Preview (March 2025)":     "Gemini 2.5 Pro Preview (March 2025)",
        #     "o1 Medium (December 2024)":               "o1 Medium (December 2024)",
        #     "o3 Medium (April 2025)":                  "o3 Medium (April 2025)",
        #     "o3-mini Low (January 2025)":              "o3-mini Low (January 2025)",
        #     "o4-mini Medium (April 2025)":             "o4-mini Medium (April 2025)",
        # }

        records = []
        for db_file in self.db_dir.glob('*.db'):
            benchmark_name = db_file.stem
            if benchmark_name in EXCLUDE_BENCHMARKS:
                continue
            try:
                with self.get_conn(benchmark_name) as conn:
                    # Only select rows with model_name and accuracy
                    df = pd.read_sql_query(
                        "SELECT model_name, accuracy FROM parsed_results",
                        conn
                    )
                    if df.empty:
                        continue
                    # Group by model_name and compute mean accuracy
                    grouped = df.groupby('model_name')['accuracy'].mean().reset_index()
                    grouped['benchmark_name'] = benchmark_name
                    records.append(grouped)
            except Exception as e:
                print(f"Error processing {db_file}: {e}")
                continue
        if not records:
            return pd.DataFrame(columns=['benchmark_name', 'model_name', 'accuracy'])
        all_df = pd.concat(records, ignore_index=True)
        # Clean up names if needed
        all_df = all_df[['benchmark_name', 'model_name', 'accuracy']]
        all_df["benchmark_name"] = all_df["benchmark_name"].replace(BENCHMARK_ALIAS)
        all_df["model_name"] = all_df["model_name"].replace(MODEL_ALIAS)
        return all_df
        
    def get_conn(self, benchmark_name):
        # Sanitize benchmark name for filename
        safe_name = benchmark_name.replace('/', '_').replace('\\', '_')
        db_path = self.db_dir / f"{safe_name}.db"
        
        # Get thread-specific connections dictionary
        if not hasattr(self.local, 'connections'):
            self.local.connections = {}
            
        # Create new connection if not exists for this benchmark
        if safe_name not in self.local.connections:
            self.local.connections[safe_name] = sqlite3.connect(db_path)
            
        return self.local.connections[safe_name]

    def create_tables(self, benchmark_name):
        with self.get_conn(benchmark_name) as conn:
            # Create parsed_results table dynamically from schema
            columns = [f"{col} {dtype}" for col, dtype in PARSED_RESULTS_COLUMNS.items()]
            create_parsed_results = f'''
                CREATE TABLE IF NOT EXISTS parsed_results (
                    {', '.join(columns)},
                    PRIMARY KEY (benchmark_name, agent_name, run_id)
                )
            '''
            conn.execute(create_parsed_results)
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS preprocessed_traces (
                    benchmark_name TEXT,
                    agent_name TEXT,
                    date TEXT,
                    run_id TEXT,
                    raw_logging_results BLOB,
                    PRIMARY KEY (benchmark_name, agent_name, run_id)
                )
            ''')
            # conn.execute('''
            #     CREATE TABLE IF NOT EXISTS failure_reports (
            #         benchmark_name TEXT,
            #         agent_name TEXT,
            #         date TEXT,
            #         run_id TEXT,
            #         failure_report BLOB,
            #         PRIMARY KEY (benchmark_name, agent_name, run_id)
            #     )
            # ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS token_usage (
                    benchmark_name TEXT,
                    agent_name TEXT,
                    run_id TEXT,
                    model_name TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    input_tokens_cache_write INTEGER,
                    input_tokens_cache_read INTEGER,
                    is_primary INTEGER DEFAULT 0,
                    PRIMARY KEY (benchmark_name, agent_name, run_id, model_name)
                )
            ''')

    def preprocess_traces(self, processed_dir="evals_live", skip_existing=True):
        processed_dir = Path(processed_dir)
        
        # Track processed files to avoid reprocessing
        processed_files_log = self.db_dir / "processed_files.txt"
        processed_files = set()
        if skip_existing and processed_files_log.exists():
            with open(processed_files_log, 'r') as f:
                processed_files = set(line.strip() for line in f)

        for file in processed_dir.glob('*.json'):
            # Skip if file already processed
            if skip_existing and str(file) in processed_files:
                print(f"Skipping already processed file: {file}")
                continue
            print(f"Processing {file}")
            primary_model_name = None
            show_primary_model_name = None
            agent_name_with_model = None
            model_show_name = None 
            with open(file, 'r') as f:
                data = json.load(f)
                stem = file.stem
                agent_name = data['config']['agent_name']
                benchmark_name = data['config']['benchmark_name']
                if "inspect" in benchmark_name:
                    benchmark_name = benchmark_name.split("/")[-1]
                date = data['config']['date']
                config = data['config']

            # Create tables for this benchmark if they don't exist
            self.create_tables(benchmark_name)

            # try:
            #     failure_report = pickle.dumps(data['failure_report'])
            #     with self.get_conn(benchmark_name) as conn:
            #         conn.execute('''
            #             INSERT INTO failure_reports 
            #             (benchmark_name, agent_name, date, run_id, failure_report)
            #             VALUES (?, ?, ?, ?, ?)
            #         ''', (benchmark_name, agent_name, date, config['run_id'], failure_report))
            # except Exception as e:
            #     print(f"Error preprocessing failure_report in {file}: {e}")

            try:
                total_usage = data.get('total_usage', {})
                print(f"Total usage is: {total_usage}")

                if benchmark_name in ['corebench_hard', 'swebench_verified_mini', 'taubench_airline']:
                    # Use get to avoid KeyError if 'model_name_short' is not present
                    primary_model_name = data['config'].get('model_name_short')
                    if primary_model_name is None:
                        primary_model_name = data['config']['agent_args'].get('model_name')
                        # get reasoning effort if any
                        reasoning_effort = data['config']['agent_args'].get('reasoning_effort')
                        if reasoning_effort:
                            primary_model_name = f"{primary_model_name} {reasoning_effort}"
                else:
                    # Find primary model from agent_name knowing Agent name is in the format "AgentName (ModelName)"
                    primary_model_name = agent_name.split('(')[-1].strip(' )') if '(' in agent_name else None

                show_primary_model_name = self.get_model_show_name(primary_model_name) if primary_model_name else None
                
                # Find the primary model based on total tokens
                if show_primary_model_name is None:
                    max_tokens = -1
                    for model_name, usage in total_usage.items():
                        completion_tokens = usage.get('completion_tokens', 0)
                        if completion_tokens > max_tokens:
                            max_tokens = completion_tokens
                            primary_model_name = model_name

                    show_primary_model_name = self.get_model_show_name(primary_model_name) if primary_model_name else primary_model_name

                # save in csv for debugging
                with open('primary_model.csv', 'a') as f:
                    f.write(f"{benchmark_name},{agent_name},{primary_model_name},{show_primary_model_name}\n")

                # If show_primary_model_name is part of models to  skip, skip this agent
                if show_primary_model_name in MODELS_TO_SKIP:
                    print(f"Skipping agent {agent_name} for benchmark {benchmark_name} due to primary model {show_primary_model_name} being in MODELS_TO_SKIP")
                    continue # This will skip this agent for this benchmark and continue to the next file

                # Rename agent_name with primary model show name (only once)
                base_agent_name = re.sub(r'\s*\(.*?\)$', '', agent_name)

                # Simple string replacements
                simple_replacements = [
                    ('Browser-Use_test', 'Browser-Use'),
                    ('hal', 'HAL'),
                    ('Hal', 'HAL'),
                    ('HAl', 'HAL'),
                ]
                
                # Apply simple replacements
                for old, new in simple_replacements:
                    base_agent_name = base_agent_name.replace(old, new)
                
                # Pattern-based mappings with case-insensitive matching
                # HAL Generalist patterns
                if 'hal' in base_agent_name.lower() and 'generalist' in base_agent_name.lower():
                    base_agent_name = 'HAL Generalist Agent'
                
                # Self-Debug patterns
                elif 'self-debug' in base_agent_name.lower() or 'selfdebug' in base_agent_name.lower():
                    base_agent_name = 'SAB Self-Debug'
                
                # TAU-bench patterns
                elif any(pattern in base_agent_name.lower() for pattern in ['few shot', 'fewshot']):
                    base_agent_name = 'TAU-bench Few Shot'
                
                # USACO patterns
                elif 'usaco' in base_agent_name.lower():
                    if 'episodic' in base_agent_name.lower() and 'semantic' in base_agent_name.lower():
                        base_agent_name = 'USACO Episodic + Semantic'
                    else:
                        base_agent_name = 'USACO Agent'
                
                # Browser/Assistant patterns
                elif any(pattern in base_agent_name.lower() for pattern in ['browser', 'assistantbench']):
                    base_agent_name = 'Browser-Use'
                
                # CORE-Agent patterns
                elif 'coreagent' in base_agent_name.lower() or 'core-agent' in base_agent_name.lower():
                    base_agent_name = 'CORE-Agent'
                
                # Col-bench patterns
                elif 'colbench' in base_agent_name.lower():
                        base_agent_name = 'Col-bench Text'
                
                # SWE-Agent patterns
                elif any(pattern in base_agent_name.lower() for pattern in ['my_agent', 'my agent', 'sweagent', 'swe-agent']):
                    base_agent_name = 'SWE-Agent'
                
                # SciCode patterns
                
                # HF Open Deep Research patterns
                elif 'hf_open_deep_research' in base_agent_name.lower() or 'hf open deep research' in base_agent_name.lower():
                    base_agent_name = 'HF Open Deep Research'
                
                # SeeAct patterns
                elif 'seeact' in base_agent_name.lower():
                    base_agent_name = 'SeeAct'
                
                # If no pattern matches, use exact mappings as fallback
                else:
                    exact_mappings = {
                        'HAL Generalist': 'HAL Generalist Agent',
                        'HAL Generalist High Reasoning': 'HAL Generalist Agent',
                        'HAL Generalist No Reasoning': 'HAL Generalist Agent',
                        'HAL Generalist Minimal Reasoning': 'HAL Generalist Agent',
                        'TauBench Few Shot High Reasoning': 'TAU-bench Few Shot',
                        'TauBench Few Shot': 'TAU-bench Few Shot',
                        'TauBench Few-Shot High Reasoning': 'TAU-bench Few Shot',
                        'TauBench Few-Shot Minimal Reasoning': 'TAU-bench Few Shot',
                        'TAU-bench Few-shot No Reasoning': 'TAU-bench Few Shot',
                        'TAU-bench FewShot No Reasoning': 'TAU-bench Few Shot',
                        'TAU-bench FewShot': 'TAU-bench Few Shot',
                        'Taubench FewShot High Reasoning': 'TAU-bench Few Shot',
                        'Taubench FewShot No Reasoning': 'TAU-bench Few Shot',
                        'TAU-bench Few Shot High Reasoning': 'TAU-bench Few Shot',
                        'Assistantbench Browser Agent': 'Browser-Use',
                        'Browser Agent': 'Browser-Use',
                        'coreagent': 'CORE-Agent',
                        'CORE-Agent': 'CORE-Agent',
                        'colbench_text_sonnet37': 'Col-bench Text',
                        'SAB Self-Debug Claude-3-7 low': 'SAB Self-Debug',
                        'My Agent': 'SWE-Agent',
                        'SAB Example Agent': 'SAB Self-Debug'
                    }
                    
                    # Apply exact mappings
                    if base_agent_name in exact_mappings:
                        base_agent_name = exact_mappings[base_agent_name]

                agent_name_with_model = f"{base_agent_name} ({show_primary_model_name})" if show_primary_model_name else base_agent_name

                for model_name, usage in total_usage.items():
                    model_show_name = self.get_model_show_name(model_name)
                    with self.get_conn(benchmark_name) as conn:
                        conn.execute('''
                            INSERT OR REPLACE INTO token_usage 
                            (benchmark_name, agent_name, run_id, model_name, 
                            prompt_tokens, completion_tokens, input_tokens, output_tokens, total_tokens,
                            input_tokens_cache_write, input_tokens_cache_read, is_primary)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            benchmark_name,
                            agent_name_with_model,
                            config['run_id'],
                            model_show_name,
                            usage.get('prompt_tokens', 0),
                            usage.get('completion_tokens', 0),
                            usage.get('input_tokens', 0),
                            usage.get('output_tokens', 0),
                            usage.get('total_tokens', 0),
                            usage.get('input_tokens_cache_write', 0),
                            usage.get('input_tokens_cache_read', 0),
                            1 if model_name == primary_model_name else 0
                        ))
                        print(f"{benchmark_name + agent_name + config['run_id'] + model_name}")
            except Exception as e:
                print(f"Error preprocessing token usage in {file}: {e}")
                print(f"{benchmark_name + agent_name + config['run_id'] + model_show_name}")
            

            try:
                # raw_logging_results = pickle.dumps(data['raw_logging_results'])
                with self.get_conn(benchmark_name) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO preprocessed_traces 
                        (benchmark_name, agent_name, date, run_id) 
                        VALUES (?, ?, ?, ?)
                    ''', (benchmark_name, agent_name_with_model, date, config['run_id']))
            except Exception as e:
                print(f"Error preprocessing raw_logging_results in {file}: {e}")

            try:
                results = data['results']

                # Ensure 'accuracy' key exists with a fallback
                if 'accuracy' not in results or results['accuracy'] is None:
                    fallback = self.get_fallback_accuracy(results)
                    if fallback is not None:
                        results['accuracy'] = fallback

                results['model_name'] = show_primary_model_name
                results['trace_stem'] = stem

                with self.get_conn(benchmark_name) as conn:
                    columns = [col for col in PARSED_RESULTS_COLUMNS.keys() 
                            if col not in ['benchmark_name', 'agent_name', 'date', 'run_id']]
                    placeholders = ','.join(['?'] * (len(columns) + 4)) # +4 for benchmark_name, agent_name, date, run_id
                    values = [
                        benchmark_name,
                        agent_name_with_model,
                        config['date'],
                        config['run_id']
                    ] + [str(results.get(col)) if col in ['successful_tasks', 'failed_tasks'] 
                        else results.get(col) for col in columns]

                    query = f'''
                        INSERT OR REPLACE INTO parsed_results  
                        ({', '.join(PARSED_RESULTS_COLUMNS.keys())})
                        VALUES ({placeholders})
                    '''
                    conn.execute(query, values)
                    
                    # After successful processing, create a reduced version of the JSON file
                    # Keep only the essential keys used during preprocessing
                    reduced_data = {
                        'config': data['config'],
                        'results': data['results'],
                        'total_usage': data.get('total_usage', {}),
                        # Keep any other small essential keys if they exist
                        'metadata': {
                            'processed': True,
                            'original_file_size': len(json.dumps(data)),
                            'processed_date': date
                        }
                    }
                    
                    # Overwrite the original file with the reduced version
                    try:
                        with open(file, 'w') as f:
                            json.dump(reduced_data, f, indent=2)
                        print(f"Reduced file size for {file}")
                    except Exception as e:
                        print(f"Error saving reduced file {file}: {e}")
                    
                    # Mark file as processed
                    if skip_existing:
                        with open(processed_files_log, 'a') as f:
                            f.write(f"{file}\n")
            except Exception as e:
                print(f"Error preprocessing parsed results in {file}: {e}")

    @lru_cache(maxsize=100)
    def get_analyzed_traces(self, agent_name, benchmark_name):
        with self.get_conn(benchmark_name) as conn:
            query = '''
                SELECT agent_name, raw_logging_results, date FROM preprocessed_traces 
                WHERE benchmark_name = ? AND agent_name = ?
            '''
            df = pd.read_sql_query(query, conn, params=(benchmark_name, agent_name))

        # check for each row if raw_logging_results is not None
        df = df[df['raw_logging_results'].apply(lambda x: pickle.loads(x) is not None and x != 'None')]

        if len(df) == 0:
            return None

        # select latest run
        df = df.sort_values('date', ascending=False).groupby('agent_name').first().reset_index()

        return pickle.loads(df['raw_logging_results'][0])

    @lru_cache(maxsize=100)
    def get_failure_report(self, agent_name, benchmark_name):
        with self.get_conn(benchmark_name) as conn:
            query = '''
                SELECT agent_name, date, failure_report FROM failure_reports 
                WHERE benchmark_name = ? AND agent_name = ?
            '''
            df = pd.read_sql_query(query, conn, params=(benchmark_name, agent_name))

        df = df[df['failure_report'].apply(lambda x: pickle.loads(x) is not None and x != 'None')]

        if len(df) == 0:
            return None

        df = df.sort_values('date', ascending=False).groupby('agent_name').first().reset_index()

        return pickle.loads(df['failure_report'][0])

    def _calculate_ci(self, data, confidence=0.95, type='minmax'):
        data = data[np.isfinite(data)]

        if len(data) < 2:
            return '', '', '' # No CI for less than 2 samples
        n = len(data)

        mean = np.mean(data)

        if type == 't':
            sem = stats.sem(data)
            ci = stats.t.interval(confidence, n-1, loc=mean, scale=sem)

        elif type == 'minmax':
            min = np.min(data)
            max = np.max(data)
            ci = (min, max)
        return mean, ci[0], ci[1]

    
    def get_parsed_results(self, benchmark_name, aggregate=True):
        with self.get_conn(benchmark_name) as conn:
            query = '''
                SELECT * FROM parsed_results 
                WHERE benchmark_name = ?
                ORDER BY accuracy DESC
            '''
            df = pd.read_sql_query(query, conn, params=(benchmark_name,))
        

        # Load metadata
        with open('agents_metadata.yaml', 'r') as f:
            metadata = yaml.safe_load(f)
        
        # Create URL mapping
        url_mapping = {}
        if benchmark_name in metadata:
            for agent in metadata[benchmark_name]:
                if 'url' in agent and agent['url']:  # Only add if URL exists and is not empty
                    url_mapping[agent['agent_name']] = agent['url']

        # Add 'Verified' column
        # verified_agents = self.load_verified_agents()
        # Temporary hack TO DO: Restore logic with yaml file later 
        df['Verified'] = '✓'
        # df['Verified'] = df.apply(lambda row: '✓' if (benchmark_name, row['agent_name']) in verified_agents else '', axis=1)

        # Add URLs to agent names if they exist
        df['url'] = df['agent_name'].apply(lambda x: url_mapping.get(x, ''))

        # Add column for how many times an agent_name appears in the DataFrame
        df['Runs'] = df.groupby('agent_name')['agent_name'].transform('count')

        # Compute the 95% confidence interval for accuracy and cost for agents that have been run more than once
        df['accuracy_ci'] = None
        df['cost_ci'] = None

        # Before dropping run_id, create new column from it with download link
        # First create a temporary dataframe with agent_name and max accuracy
        max_accuracy_df = df.groupby('agent_name')['accuracy'].transform('max')
        # Create mask for rows with max accuracy in their group
        max_accuracy_mask = df['accuracy'] == max_accuracy_df
        # Create the Traces column, only setting values for max accuracy rows
        df['Traces'] = ''
        df.loc[max_accuracy_mask, 'Traces'] = df.loc[max_accuracy_mask, 'trace_stem'].apply(
            lambda x: f'https://huggingface.co/datasets/agent-evals/hal_traces/resolve/main/{x}.zip?download=true' if x else ''
        )
        # df['Traces'] = ''
        # df.loc[max_accuracy_mask, 'Traces'] = df.loc[max_accuracy_mask, 'run_id'].apply(
        #     lambda x: f'https://huggingface.co/datasets/agent-evals/agent_traces/resolve/main/{x}.zip?download=true'
        #     if x else ''
        # )
        
        df = df.drop(columns=['successful_tasks', 'failed_tasks'], axis=1)
        
        if aggregate:
            df = df.groupby('agent_name').agg(AGGREGATION_RULES).reset_index()
            
        # Rename columns using the display names mapping
        df = df.rename(columns=COLUMN_DISPLAY_NAMES)
        
        # Multiply accuracy by 100
        df['Accuracy'] = df['Accuracy'] * 100
        df['Scenario Goal Completion'] = df['Scenario Goal Completion'] * 100
        df['Task Goal Completion'] = df['Task Goal Completion'] * 100
        df['Level 1 Accuracy'] = df['Level 1 Accuracy'] * 100
        df['Level 2 Accuracy'] = df['Level 2 Accuracy'] * 100
        df['Level 3 Accuracy'] = df['Level 3 Accuracy'] * 100
        df['Refusals'] = df['Refusals'] * 100
        df['Non-Refusal Harm Score'] = df['Non-Refusal Harm Score'] * 100
        
        return df
    
    def get_task_success_data(self, benchmark_name):
        with self.get_conn(benchmark_name) as conn:
            query = '''
                SELECT agent_name, accuracy, successful_tasks, failed_tasks
                FROM parsed_results 
                WHERE benchmark_name = ?
            '''
            df = pd.read_sql_query(query, conn, params=(benchmark_name,))
        
        # Get all unique task IDs
        task_ids = set()
        for tasks in df['successful_tasks']:
            if ast.literal_eval(tasks) is not None:
                task_ids.update(ast.literal_eval(tasks))
        for tasks in df['failed_tasks']:
            if ast.literal_eval(tasks) is not None:
                task_ids.update(ast.literal_eval(tasks))

        # Create a DataFrame with agent_name, task_ids, and success rates
        data_list = []
        for task_id in task_ids:
            for agent_name in df['agent_name'].unique():
                agent_runs = df[df['agent_name'] == agent_name]
                # Count how many times this task was successful across all runs
                successes = sum(1 for _, row in agent_runs.iterrows() 
                              if ast.literal_eval(row['successful_tasks']) is not None 
                              and task_id in ast.literal_eval(row['successful_tasks']))
                total_runs = len(agent_runs)
                success_rate = successes / total_runs if total_runs > 0 else 0
                
                data_list.append({
                    'agent_name': agent_name,
                    'task_id': task_id,
                    'success': success_rate
                })
        df = pd.DataFrame(data_list)

        df = df.rename(columns={
            'agent_name': 'Agent Name',
            'task_id': 'Task ID',
            'success': 'Success'
        })

        return df
    
    def load_verified_agents(self, file_path='agents_metadata.yaml'):
        with open(file_path, 'r') as f:
            metadata = yaml.safe_load(f)
        
        verified_agents = set()
        for benchmark, agents in metadata.items():
            for agent in agents:
                if 'verification_date' in agent:  # Only add if verified
                    verified_agents.add((benchmark, agent['agent_name']))
        
        return verified_agents

    def get_token_usage_with_costs(self, benchmark_name, pricing_config=None):
        """Get token usage data with configurable pricing"""
        if pricing_config is None:
            pricing_config = DEFAULT_PRICING

        with self.get_conn(benchmark_name) as conn:
            query = '''
                SELECT agent_name, model_name, run_id,
                SUM(prompt_tokens) as prompt_tokens,
                SUM(completion_tokens) as completion_tokens,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens,
                SUM(total_tokens) as total_tokens,
                SUM(input_tokens_cache_write) as input_tokens_cache_write,
                SUM(input_tokens_cache_read) as input_tokens_cache_read
                FROM token_usage
                WHERE benchmark_name = ?
                GROUP BY agent_name, model_name, run_id
            '''
            df = pd.read_sql_query(query, conn, params=(benchmark_name,))
                    
        # Calculate costs based on pricing config (prices are per 1M tokens)
        df['total_cost'] = 0.0
        for model, prices in pricing_config.items():
            mask = df['model_name'] == model
            df.loc[mask, 'total_cost'] = (
                df.loc[mask, 'input_tokens'] * prices['prompt_tokens'] / 1e6 +
                df.loc[mask, 'output_tokens'] * prices['completion_tokens'] / 1e6 +
                df.loc[mask, 'input_tokens_cache_read'] * prices['prompt_tokens'] / 1e6 +
                df.loc[mask, 'input_tokens_cache_write'] * prices['prompt_tokens'] / 1e6 +
                df.loc[mask, 'prompt_tokens'] * prices['prompt_tokens'] / 1e6 +
                df.loc[mask, 'completion_tokens'] * prices['completion_tokens'] / 1e6
            )
            
        # Sum total_cost for each run_id (if agents use multiple models, this will be the total cost for that run)
        df_temp = df.groupby('run_id')['total_cost'].sum().reset_index()
        df_temp = df_temp.rename(columns={'total_cost': 'total_cost_temp'})
        df = df.merge(df_temp, on='run_id', how='left')
        df['total_cost'] = df['total_cost_temp']
        df = df.drop('total_cost_temp', axis=1)
                                
        return df

    def get_parsed_results_with_costs(self, benchmark_name, pricing_config=None, aggregate=False):
        """Get parsed results with recalculated costs based on token usage"""
        # Get base results with URLs
        results_df = self.get_parsed_results(benchmark_name, aggregate=False)
        benchmark_name = results_df['benchmark_name'].iloc[0]
        
        # Get token usage with new costs
        token_costs = self.get_token_usage_with_costs(benchmark_name, pricing_config)
        # import pdb; pdb.set_trace()

        for agent_name in results_df['Agent Name'].unique():
            agent_df = results_df[results_df['Agent Name'] == agent_name]
            
            if agent_name not in token_costs['agent_name'].unique():
                token_costs_df = results_df[results_df['Agent Name'] == agent_name]
            else:
                token_costs_df = token_costs[token_costs['agent_name'] == agent_name]
                
            if len(agent_df) > 1:
                accuracy_mean, accuracy_lower, accuracy_upper = self._calculate_ci(agent_df['Accuracy'], type='minmax')
                if agent_name not in token_costs['agent_name'].unique():
                    cost_mean, cost_lower, cost_upper = self._calculate_ci(token_costs_df['Total Cost'], type='minmax')
                else:
                    cost_mean, cost_lower, cost_upper = self._calculate_ci(token_costs_df['total_cost'], type='minmax')
                
                # Round CI values to 2 decimals
                accuracy_ci = f"-{abs(accuracy_mean - accuracy_lower):.2f}/+{abs(accuracy_mean - accuracy_upper):.2f}"
                cost_ci = f"-{abs(cost_mean - cost_lower):.2f}/+{abs(cost_mean - cost_upper):.2f}"
                
                results_df.loc[results_df['Agent Name'] == agent_name, 'Accuracy CI'] = accuracy_ci
                results_df.loc[results_df['Agent Name'] == agent_name, 'Total Cost CI'] = cost_ci
            
            if agent_name == 'Inspect ReAct Agent (o3-mini-2025-01-14)':
                results_df.loc[results_df['Agent Name'] == agent_name, 'Total Cost CI'] = "lower bound: $16.04*"
    
        # Group token costs by agent
        agent_costs = token_costs.groupby('agent_name')['total_cost'].mean().reset_index()

        agent_costs = agent_costs.rename(columns={
            'agent_name': 'agent_name_temp',
            'total_cost': 'Total Cost'
        })
                        
        # Drop existing Total Cost column if it exists
        if 'Total Cost' in results_df.columns:
            results_df['total_cost_temp'] = results_df['Total Cost']
            results_df = results_df.drop('Total Cost', axis=1)
            
        # Create temp column for matching, preserving the original Agent Name with URL
        results_df['agent_name_temp'] = results_df['Agent Name'].apply(lambda x: x.split('[')[1].split(']')[0] if '[' in x else x)
        
        # Update costs in results
        results_df = results_df.merge(agent_costs, on='agent_name_temp', how='left')
                
        # if Total Cost is NaN, set it to the value from total_cost_temp if it exists
        results_df['Total Cost'] = results_df['Total Cost'].fillna(results_df['total_cost_temp'])
        
        # If there is no token usage data, set Total Cost to total_cost from results key
        if len(token_costs) < 1:
            results_df['Total Cost'] = results_df['Total Cost'].fillna(results_df['total_cost_temp'])
                
        # Drop temp column
        results_df = results_df.drop('agent_name_temp', axis=1)
        results_df = results_df.drop('total_cost_temp', axis=1)
        
                    
        if aggregate:
            # Aggregate results while preserving URLs in Agent Name
            results_df = results_df.groupby('Agent Name', as_index=False).agg({
                'Date': 'first',
                'Total Cost': 'mean',
                'Accuracy': 'mean',
                'Precision': 'mean',
                'Recall': 'mean',
                'F1 Score': 'mean',
                'AUC': 'mean',
                'Overall Score': 'mean',
                'Vectorization Score': 'mean',
                'Fathomnet Score': 'mean',
                'Feedback Score': 'mean',
                'House Price Score': 'mean',
                'Spaceship Titanic Score': 'mean',
                'AMP Parkinsons Disease Progression Prediction Score': 'mean',
                'CIFAR10 Score': 'mean',
                'IMDB Score': 'mean',
                'Scenario Goal Completion': 'mean',
                'Task Goal Completion': 'mean',
                'Level 1 Accuracy': 'mean',
                'Level 2 Accuracy': 'mean',
                'Level 3 Accuracy': 'mean',
                'Verified': 'first',
                'Traces': 'first',
                'Runs': 'first',
                'Accuracy CI': 'first',
                'Total Cost CI': 'first',
                'URL': 'first',
                'Refusals': 'mean',
                'Non-Refusal Harm Score': 'mean',  # Preserve URL
                'Model Name': 'first',
            })
        
        # Round float columns to 2 decimal places
        float_columns = [
            'Accuracy',
            'Precision',
            'Recall',
            'F1 Score',
            'AUC',
            'Overall Score',
            'Vectorization Score',
            'Fathomnet Score',
            'Feedback Score',
            'House Price Score',
            'Spaceship Titanic Score',
            'AMP Parkinsons Disease Progression Prediction Score',
            'CIFAR10 Score',
            'IMDB Score',
            'Level 1 Accuracy',
            'Level 2 Accuracy',
            'Level 3 Accuracy',
            'Total Cost'
        ]
        
        for column in float_columns:
            if column in results_df.columns:
                try:
                    results_df[column] = results_df[column].round(2)
                except Exception as e:
                    print(f"Error rounding {column}: {e}")
        
        return results_df

    def check_token_usage_data(self, benchmark_name):
        """Debug helper to check token usage data"""
        with self.get_conn(benchmark_name) as conn:
            query = '''
                SELECT * FROM token_usage
                WHERE benchmark_name = ?
            '''
            df = pd.read_sql_query(query, conn, params=(benchmark_name,))
        return df

    def get_models_for_benchmark(self, benchmark_name):
        """Get list of unique model names used in a benchmark"""
        with self.get_conn(benchmark_name) as conn:
            query = '''
                SELECT DISTINCT model_name
                FROM token_usage
                WHERE benchmark_name = ?
            '''
            df = pd.read_sql_query(query, conn, params=(benchmark_name,))
        return df['model_name'].tolist()

    def get_all_agents(self, benchmark_name):
        """Get list of all agent names for a benchmark"""
        with self.get_conn(benchmark_name) as conn:
            query = '''
                SELECT DISTINCT agent_name
                FROM parsed_results
                WHERE benchmark_name = ?
            '''
            df = pd.read_sql_query(query, conn, params=(benchmark_name,))
        return df['agent_name'].tolist()

    def get_total_benchmarks(self):
        """Get the total number of unique benchmarks in the database"""
        benchmarks = set()
        for db_file in self.db_dir.glob('*.db'):
            benchmarks.add(db_file.stem.replace('_', '/'))
        return len(benchmarks) - 3 # TODO hardcoded -3 because of benchmarks not added for now

    def get_total_agents(self):
        """Get the total number of unique agents across all benchmarks"""
        total_agents = set()
        # Use the parsed_results table since it's guaranteed to have all benchmark-agent pairs
        for db_file in self.db_dir.glob('*.db'):
            # skip colbench, scienceagentbench
            if db_file.stem in ['colbench_backend_programming', 'colbench_frontend_design', 'scienceagentbench']:
                continue # TODO remove hardcoded skip once these benchmarks are added
            benchmark_name = db_file.stem.replace('_', '/')
            with self.get_conn(benchmark_name) as conn:
                query = '''
                    SELECT DISTINCT benchmark_name, agent_name 
                    FROM parsed_results
                '''
                
                results = conn.execute(query).fetchall()
                # Add each benchmark-agent pair to the set
                total_agents.update(results)
        return len(total_agents)

    def get_agent_url(self, agent_name, benchmark_name):
        """Get the URL for an agent from the metadata file."""
        try:
            with open('agents_metadata.yaml', 'r') as f:
                metadata = yaml.safe_load(f)
                if benchmark_name in metadata:
                    for agent in metadata[benchmark_name]:
                        if agent['agent_name'] == agent_name:
                            return agent.get('url', '')
        except Exception as e:
            print(f"Error getting agent URL: {e}")
        return ''

    def get_highlight_results(self, limit_per_benchmark=3):
        """Get highlight results organized by benchmark and agent for the landing page"""
        EXCLUDE_BENCHMARKS = [
            'colbench_backend_programming',
            'colbench_frontend_design', 
            'scienceagentbench',
        ]
        
        BENCHMARK_DISPLAY_NAMES = {
            "usaco": "USACO",
            "taubench_airline": "TAU-bench Airline",
            "swebench_verified_mini": "SWE-bench Verified Mini",
            "scicode": "Scicode",
            "online_mind2web": "Online Mind2Web",
            "gaia": "GAIA",
            "corebench_hard": "CORE-Bench Hard",
            "assistantbench": "AssistantBench"
        }
        
        BENCHMARK_CATEGORIES = {
            "usaco": "Programming",
            "taubench_airline": "Customer Service",
            "swebench_verified_mini": "Software Engineering",
            "scicode": "Scientific Programming",
            "online_mind2web": "Web Assistance",
            "gaia": "Web Assistance",
            "corebench_hard": "Scientific Programming",
            "assistantbench": "Web Assistance",
            "scienceagentbench": "Scientific Programming",
        }
        
        highlights = []
        
        for db_file in self.db_dir.glob('*.db'):
            benchmark_name = db_file.stem
            if benchmark_name in EXCLUDE_BENCHMARKS:
                continue
                
            display_name = BENCHMARK_DISPLAY_NAMES.get(benchmark_name, benchmark_name.replace('_', ' ').title())
            category = BENCHMARK_CATEGORIES.get(benchmark_name, "Other")
            
            try:
                with self.get_conn(benchmark_name) as conn:
                    # Get top performing agent-model combinations
                    query = '''
                        SELECT agent_name, model_name, accuracy, total_cost
                        FROM parsed_results 
                        WHERE benchmark_name = ? AND accuracy IS NOT NULL
                        ORDER BY accuracy DESC
                        LIMIT ?
                    '''
                    df = pd.read_sql_query(query, conn, params=(benchmark_name, limit_per_benchmark))
                    
                    if df.empty:
                        continue
                    
                    # Convert to list of agent-model combinations
                    top_agents = []
                    for _, row in df.iterrows():
                        # Extract base agent name (without model info)
                        base_agent = re.sub(r'\s*\(.*?\)$', '', row['agent_name']).strip()
                        
                        top_agents.append({
                            'agent_name': row['agent_name'],
                            'base_agent': base_agent,
                            'model_name': row['model_name'],
                            'accuracy': row['accuracy'] * 100,  # Convert to percentage
                            'total_cost': row['total_cost'] if row['total_cost'] else 0,
                        })
                    
                    if top_agents:
                        highlights.append({
                            'benchmark': display_name,
                            'benchmark_key': benchmark_name,
                            'category': category,
                            'agents': top_agents
                        })
                        
            except Exception as e:
                print(f"Error processing highlights for {benchmark_name}: {e}")
                continue
        
        return highlights

if __name__ == '__main__':
    preprocessor = TracePreprocessor()
    preprocessor.preprocess_traces()