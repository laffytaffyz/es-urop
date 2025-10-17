from __future__ import annotations

import ast
import json
import re
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

GRID = list[list[int]]

ARC_SYSTEM_PROMPT = (
    "You are an expert puzzle solver. Carefully study the training examples, "
    "infer the transformation rule, and apply it to the test input grid. "
    "Respond with only the transformed output grid as a Python list of lists of integers."
)

ARC_USER_PREAMBLE = """
Find the common transformation that maps each input grid to its output grid in the training examples.
Apply the exact same rule to the test input grid.
Write the resulting output grid using Python list-of-lists syntax (e.g., [[0, 1], [2, 3]]) with no extra commentary.
""".strip()


@dataclass(frozen=True)
class ArcExample:
    input: GRID
    output: GRID


@dataclass(frozen=True)
class ArcTaskSample:
    task_id: str
    test_index: int
    train_examples: list[ArcExample]
    test_input: GRID
    target: GRID
    messages: list[dict[str, str]]

    def build_prompt(self, tokenizer) -> str:
        """Render chat messages with the provided tokenizer."""
        return tokenizer.apply_chat_template(
            self.messages, add_generation_prompt=True, tokenize=False
        )


def grid_to_str(grid: GRID) -> str:
    """Render a grid as whitespace-separated rows."""
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def grid_to_literal(grid: GRID) -> str:
    """Render a grid as a Python list literal."""
    row_literals = [", ".join(str(c) for c in row) for row in grid]
    return "[" + ", ".join(f"[{row}]" for row in row_literals) + "]"


def _format_training_examples(examples: Sequence[ArcExample]) -> str:
    lines: list[str] = []
    for idx, example in enumerate(examples, start=1):
        lines.append(f"Example {idx}:")
        lines.append("Input:")
        lines.append(grid_to_str(example.input))
        lines.append("Output:")
        lines.append(grid_to_str(example.output))
        if idx != len(examples):
            lines.append("")
    return "\n".join(lines)


def build_arc_user_prompt(examples: Sequence[ArcExample], test_input: GRID) -> str:
    example_block = _format_training_examples(examples)
    lines = [
        ARC_USER_PREAMBLE,
        "",
        example_block,
        "",
        "Test Input Grid:",
        grid_to_str(test_input),
        "",
        "Output Grid:",
    ]
    return "\n".join(lines)


def build_arc_chat_messages(
    examples: Sequence[ArcExample], test_input: GRID
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ARC_SYSTEM_PROMPT},
        {"role": "user", "content": build_arc_user_prompt(examples, test_input)},
    ]


def load_arc_agi_dataset(
    challenges_path: str | Path,
    solutions_path: str | Path,
    *,
    limit: int | None = None,
) -> list[ArcTaskSample]:
    """Load ARC-AGI training data and pair each test grid with its solution."""
    challenges_data = json.loads(Path(challenges_path).read_text())
    solutions_data = json.loads(Path(solutions_path).read_text())

    samples: list[ArcTaskSample] = []
    for task_id, challenge in challenges_data.items():
        train_examples = [
            ArcExample(input=example["input"], output=example["output"])
            for example in challenge["train"]
        ]
        test_inputs = [test_case["input"] for test_case in challenge["test"]]
        solutions = solutions_data.get(task_id, [])

        if len(solutions) != len(test_inputs):
            raise ValueError(
                f"Mismatched number of solutions for task {task_id}: "
                f"{len(solutions)} solutions vs {len(test_inputs)} test inputs."
            )

        for test_index, test_input in enumerate(test_inputs):
            messages = build_arc_chat_messages(train_examples, test_input)
            samples.append(
                ArcTaskSample(
                    task_id=task_id,
                    test_index=test_index,
                    train_examples=train_examples,
                    test_input=test_input,
                    target=solutions[test_index],
                    messages=messages,
                )
            )
            if limit is not None and len(samples) >= limit:
                return samples

    return samples


def parse_grid_from_text(
    text: str, *, expected_shape: tuple[int, int] | None = None
) -> GRID | None:
    """
    Extract a grid from model text output. Tries literal parsing first, then falls back to regex.
    """

    def _coerce_grid(obj: object) -> GRID | None:
        if not isinstance(obj, list) or not obj:
            return None
        coerced: list[list[int]] = []
        for row in obj:
            if not isinstance(row, list) or not row:
                return None
            coerced.append([int(cell) for cell in row])
        return coerced

    def _try_literal(candidate: str) -> GRID | None:
        try:
            parsed = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            return None
        return _coerce_grid(parsed)

    text = text.strip()

    for block in re.findall(r"```(?:json|python)?\s*(.*?)```", text, flags=re.DOTALL):
        start = block.find("[")
        end = block.rfind("]")
        if start != -1 and end != -1 and end > start:
            grid = _try_literal(block[start : end + 1])
            if grid is not None:
                return grid

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        grid = _try_literal(text[start : end + 1])
        if grid is not None:
            return grid

    numbers = [int(token) for token in re.findall(r"-?\d+", text)]
    if not numbers:
        return None

    if expected_shape is not None:
        rows, cols = expected_shape
        if rows * cols == len(numbers):
            return [
                numbers[row * cols : (row + 1) * cols] for row in range(rows)
            ]
        return None

    side = int(len(numbers) ** 0.5)
    if side * side == len(numbers):
        return [numbers[i * side : (i + 1) * side] for i in range(side)]

    return None


def grid_accuracy(predicted: GRID | None, target: GRID) -> float:
    """Compute cell-level accuracy between predicted and target grids."""
    if predicted is None:
        return 0.0
    if len(predicted) != len(target) or any(
        len(predicted[row]) != len(target[row]) for row in range(len(target))
    ):
        return 0.0

    total = 0
    correct = 0
    for row_idx, target_row in enumerate(target):
        pred_row = predicted[row_idx]
        for col_idx, target_val in enumerate(target_row):
            total += 1
            if pred_row[col_idx] == target_val:
                correct += 1
    return correct / total if total else 0.0




class SignalCheckpoint:
    """Handle preemption/time-limit signals and invoke a user-supplied checkpoint callback on the main process."""

    def __init__(self, accelerator, signals=None):
        self.accelerator = accelerator
        self._save_callback = None
        self.iteration = 0
        self._shutdown_requested = False # new
        default_signals = [signal.SIGTERM, signal.SIGINT]
        for optional in (getattr(signal, 'SIGUSR1', None), getattr(signal, 'SIGHUP', None)):
            if optional is not None:
                default_signals.append(optional)
        if signals is None:
            self._signals = tuple(dict.fromkeys(default_signals))
        else:
            cleaned = [sig for sig in signals if sig is not None]
            self._signals = tuple(dict.fromkeys(cleaned)) or tuple(dict.fromkeys(default_signals))

    def bind(self, save_callback):
        """Register the callback and attach signal handlers (main process only)."""
        self._save_callback = save_callback
        if not self.accelerator.is_main_process:
            return
        for sig in self._signals:
            signal.signal(sig, self._handle_signal)

    def update_iteration(self, iteration):
        self.iteration = iteration
    
    def should_shutdown(self): # new
        return self._shutdown_requested

    def _handle_signal(self, signum, frame): # new
        # if not self.accelerator.is_main_process:
        #     os._exit(0)
        # print(f'Received signal {signum}; attempting to save checkpoint before exit.', flush=True)
        # try:
        #     if self._save_callback is not None:
        #         self._save_callback(self.iteration)
        # except Exception as exc:
        #     print(f'Failed to save checkpoint in signal handler: {exc}', flush=True)
        # finally:
        #     os._exit(0)
        print(f"[main] Received signal {signum}; will checkpoint at end of iteration.", flush=True)
        self._shutdown_requested = True


__all__ = [
    "ArcExample",
    "ArcTaskSample",
    "ARC_SYSTEM_PROMPT",
    "build_arc_chat_messages",
    "build_arc_user_prompt",
    "grid_accuracy",
    "grid_to_literal",
    "grid_to_str",
    "grids_match",
    "load_arc_agi_dataset",
    "parse_grid_from_text",
    "SignalCheckpoint",
]
