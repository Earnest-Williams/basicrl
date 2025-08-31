"""
Scripting Engine - A module for script processing and command automation

This module provides tools for processing game commands, defining macros,
and interpreting scripts. It includes:
- MacroManager: Handles defining, expanding, and executing command macros
- BrainfuckRunner: A simple Brainfuck interpreter for script execution
"""

import re
import warnings
from io import StringIO

import numpy as np
import structlog

log = structlog.get_logger(__name__)

# Suppress numpy overflow warnings that are expected with uint64 operations
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="overflow encountered"
)


# --- EBF Interpreter ---
def _sanitize(code):
    """Clean Brainfuck code by removing any non-command characters"""
    return "".join(c for c in code if c in "><+-.,[]")


def _generate_bf_func(code, tape_size=30000):
    """
    Dynamically generate a Python function that interprets the Brainfuck code.

    Args:
        code (str): The Brainfuck code to interpret
        tape_size (int): Size of the virtual memory tape

    Returns:
        function: A Python function that runs the Brainfuck code
    """
    # This function generates Python code dynamically, which can be complex
    # Consider using a more direct interpreter if possible, or ensure thorough testing
    lines = [
        "def bf_program(tape, input_stream, output_stream):",
        "    ptr = 0",
        "    input_pos = 0",
    ]
    indent = "    "
    stack = []

    for c in _sanitize(code):
        if c == ">":
            lines.append(f"{indent}ptr = (ptr + 1) % len(tape)")  # Wrap pointer
        elif c == "<":
            lines.append(
                f"{indent}ptr = (ptr - 1 + len(tape)) % len(tape)"
            )  # Wrap pointer
        elif c == "+":
            lines.append(f"{indent}tape[ptr] = (tape[ptr] + 1) % 256")
        elif c == "-":
            # Ensure result is non-negative before modulo
            lines.append(f"{indent}tape[ptr] = (tape[ptr] - 1 + 256) % 256")
        elif c == ".":
            lines.append(f"{indent}output_stream.write(chr(tape[ptr]))")
        elif c == ",":
            # Read from input buffer if available
            lines.append(f"{indent}if input_pos < len(input_stream):")
            lines.append(f"{indent}    tape[ptr] = ord(input_stream[input_pos]) % 256")
            lines.append(f"{indent}    input_pos += 1")
            lines.append(f"{indent}else:")
            # Define behavior for EOF - often 0 or -1 (here 0)
            lines.append(f"{indent}    tape[ptr] = 0")
        elif c == "[":
            lines.append(f"{indent}while tape[ptr] != 0:")
            stack.append(indent)
            indent += "    "
        elif c == "]":
            if not stack:
                raise SyntaxError("Unmatched ']'")
            indent = stack.pop()
            lines.append(f"{indent}# End of loop")  # Add comment for clarity

    if stack:
        raise SyntaxError("Unmatched '['")

    # Add return statement or final processing if needed
    lines.append(f"{indent}return ''.join(output_stream.getvalue())")

    full_code = "\n".join(lines)
    # print("Generated BF Code:\n", full_code) # For debugging generated code
    namespace = {}
    try:
        exec(full_code, namespace)
        return namespace["bf_program"]
    except Exception as e:
        log.error("Error executing generated BF code", error=str(e), exc_info=True)
        raise  # Re-raise the error


def _should_jit(code):
    """
    Determine if JIT compilation should be used for the given Brainfuck code.

    Args:
        code (str): The Brainfuck code

    Returns:
        bool: True if JIT should be used, False otherwise
    """
    # Simple heuristic, may need refinement
    clean = _sanitize(code)
    length = len(clean)
    loop_count = clean.count("[")
    io_count = clean.count(".") + clean.count(",")

    # Avoid JIT for very short code or code with lots of I/O
    if length < 50 or io_count > length // 5:
        return False
    # Favor JIT for longer code with loops
    if loop_count > 2 and length > 100:
        return True
    return False  # Default to no JIT


class BrainfuckRunner:
    """
    A simple and efficient Brainfuck interpreter with optional JIT compilation.
    """

    def __init__(self, tape_size=30000):
        """
        Initialize the Brainfuck interpreter.

        Args:
            tape_size (int): Size of the virtual memory tape
        """
        self.tape_size = tape_size
        # Remove self.output state, run should return output directly

    def run(self, code, input_data="", jit=None):
        """
        Execute Brainfuck code and return the result.

        Args:
            code (str): The Brainfuck code to execute
            input_data (str): Input data to feed to the ',' command
            jit (bool or None): Whether to use JIT compilation (None = auto-detect)

        Returns:
            dict: A dictionary with 'success' status and 'output' or 'error'
        """

        # Check if numba is available
        has_njit = False
        if jit is not False:  # Allow forcing JIT off
            try:
                from numba import njit

                has_njit = True
            except ImportError:
                jit = False  # Force JIT off if numba not available

        output_stream = StringIO()

        try:
            use_jit = _should_jit(code) if jit is None else jit

            # Pass input/output streams to generated function
            bf_func = _generate_bf_func(code, self.tape_size)
            tape = np.zeros(self.tape_size, dtype=np.uint8)  # Use uint8 for BF tape

            if use_jit and has_njit:
                try:
                    # Numba doesn't easily support StringIO, need wrapper or different approach
                    # For now, disable JIT if I/O is involved, or handle I/O outside JITted part
                    if "." in code or "," in code:
                        log.warning(
                            "JIT compilation with I/O might be slow or unsupported. Running interpreted."
                        )
                        output = bf_func(tape, input_data, output_stream)
                    else:
                        # JIT compile the function without I/O arguments initially
                        # This is complex; direct interpretation might be better
                        # compiled_func = njit(lambda t: bf_func(t, "", StringIO()))
                        # compiled_func(tape)
                        # output = output_stream.getvalue() # This won't work as StringIO isn't updated by JIT
                        log.warning(
                            "JIT with I/O needs complex handling. Running interpreted."
                        )
                        output = bf_func(tape, input_data, output_stream)

                except Exception as e:
                    log.warning(
                        "Numba JIT compilation failed. Falling back to interpreter",
                        error=str(e),
                        exc_info=True,
                    )
                    # Fallback to non-JIT
                    output = bf_func(tape, input_data, output_stream)
            else:
                # Run interpreted
                output = bf_func(tape, input_data, output_stream)

            return {"success": True, "output": output}
        except Exception as e:

            # print(f"Brainfuck execution error: {e}\n{traceback.format_exc()}") # More detailed error
            return {"success": False, "error": str(e)}
        # No finally needed as StringIO closes automatically


class MacroManager:
    """
    Manages the definition, expansion, and execution of command macros.
    Also handles Brainfuck code execution through integration with BrainfuckRunner.
    """

    def __init__(self, game_state=None):
        """
        Initialize the MacroManager.

        Args:
            game_state: Reference to the game state for command execution
        """
        self.macros = {}
        self.game_state = (
            game_state  # Ensure this is updated if game_state changes (e.g., on load)
        )
        self.bf_runner = BrainfuckRunner()

    def define(self, name, sequence):
        """
        Define a new macro with the given name and command sequence.

        Args:
            name (str): The macro name (must start with !)
            sequence (str): The command sequence for this macro

        Returns:
            str: Confirmation message or error message
        """
        # Add basic validation for macro names
        if not re.match(r"^!\w+$", name):
            return f"Error: Invalid macro name '{name}'. Must start with ! and contain only letters, numbers, or _."
        self.macros[name] = sequence
        return f"Defined macro {name} = {sequence}"

    def expand_macros(self, text, expansion_limit=10):
        """
        Expand all macros in the given text, with limits to prevent infinite recursion.

        Args:
            text (str): The text containing macros to expand
            expansion_limit (int): Maximum recursive expansion depth

        Returns:
            str: The text with all macros expanded
        """
        # Prevent infinite recursion more robustly
        pattern = re.compile(r"!\w+")
        expanded_text = text
        depth = 0
        seen_macros = set()  # Track macros used in current expansion chain

        while depth < expansion_limit:
            found_macro = False
            current_expansion = ""
            last_end = 0

            for match in pattern.finditer(expanded_text):
                macro_name = match.group(0)
                start, end = match.span()

                # Append text before the macro
                current_expansion += expanded_text[last_end:start]

                if macro_name in seen_macros:
                    # Prevent direct recursion within this expansion path
                    log.warning(
                        "Detected recursion for macro; stopping expansion", macro=macro_name
                    )
                    current_expansion += macro_name  # Keep the macro name as is
                elif macro_name in self.macros:
                    # Expand the macro
                    replacement = self.macros[macro_name]
                    current_expansion += replacement
                    found_macro = True
                    # We could add 'macro_name' to seen_macros here for stricter recursion check
                    # but simple depth limit might be enough
                else:
                    # Macro not found, keep it as is
                    current_expansion += macro_name

                last_end = end

            # Append any remaining text after the last macro
            current_expansion += expanded_text[last_end:]

            if not found_macro:
                # No more macros found in this pass
                break

            expanded_text = current_expansion
            depth += 1
            seen_macros.clear()  # Reset seen set for the next level of expansion if needed, depends on desired recursion depth handling

        if depth >= expansion_limit:
            log.warning(
                "Macro expansion reached depth limit", expansion_limit=expansion_limit
            )

        return expanded_text

    def execute_command_sequence(self, command_string):
        """
        Execute a sequence of game commands.

        Args:
            command_string (str): The command string to execute

        Returns:
            str: The game state display text after execution
        """
        # Split by semicolons to handle command sequences
        command_groups = [
            cmd.strip() for cmd in command_string.split(";") if cmd.strip()
        ]
        outputs = []

        # Ensure game_state is valid before processing
        if not self.game_state:
            return "Error: Game state not available."

        for group in command_groups:
            # Process each character in the command group individually
            for char in group:
                if not char:
                    continue

                # Each single character is a valid game command
                # Check game over before processing turn
                if self.game_state.is_game_over():
                    outputs.append(self.game_state.get_display_text())
                    break  # Stop processing sequence if game ended

                result = self.game_state.process_turn(char)
                outputs.append(
                    result
                )  # Append the display text returned by process_turn

            # If game ended, break the outer loop too
            if self.game_state.is_game_over():
                break

        # Return the final display state text
        return outputs[-1] if outputs else "No valid commands executed."

    def process_line(self, line):
        """
        Process a line of input, which could be a macro definition, macro execution,
        Brainfuck code, or direct game commands.

        Args:
            line (str): The input line to process

        Returns:
            str or dict: The result of processing (usually display text or error info)
        """
        line = line.strip()

        # Define a new macro
        if line.startswith("!") and "=" in line:
            # Ensure correct splitting, handle potential '=' in sequence
            parts = line.split("=", 1)
            if len(parts) == 2:
                name, seq = map(str.strip, parts)
                if name.startswith("!"):  # Validate name starts with !
                    return self.define(name, seq)
                else:
                    return "Error: Macro name must start with '!'"
            else:  # Handle case like "!foo=" or just "!foo"
                return "Error: Invalid macro definition format. Use !name=sequence"

        # Expand potential macros in the line first
        # Limit expansion depth to prevent infinite loops
        try:
            expanded_line = self.expand_macros(line, expansion_limit=10)
        except Exception as e:
            return f"Error during macro expansion: {e}"

        # Execute a macro (check if the fully expanded line is just a known macro name)
        if expanded_line.startswith("!") and expanded_line in self.macros:
            # This case might be redundant if expand_macros handles it,
            # but could be kept for clarity or direct single macro execution.
            # We re-expand here in case the macro itself contains other macros.
            final_sequence = self.expand_macros(
                self.macros[expanded_line], expansion_limit=10
            )
            return self.execute_command_sequence(final_sequence)

        # Check if the expanded line looks like Brainfuck code
        # Use a stricter check: must contain BF chars and potentially brackets
        is_bf_chars = set(expanded_line) <= set("><+-.,[]")
        has_brackets = "[" in expanded_line and "]" in expanded_line
        likely_bf = is_bf_chars or has_brackets  # Adjust logic as needed

        if likely_bf and len(expanded_line) > 0:  # Check length > 0
            try:
                # Execute Brainfuck code
                # Pass empty string "" as default input for now
                result = self.bf_runner.run(expanded_line, input_data="")

                if result["success"]:
                    bf_output = result["output"]
                    if bf_output:
                        # Convert Brainfuck output to game commands
                        # Note: BF output might not be valid game commands!
                        print(f"Brainfuck output: {bf_output}")  # Log BF output
                        return self.execute_command_sequence(bf_output)
                    else:
                        return "Brainfuck program executed with no command output."
                else:
                    # Return dict for clearer error handling in UI
                    return {
                        "error": f"Brainfuck Error: {result['error']}",
                        "is_error": True,
                    }
            except Exception as e:

                # print(f"Error processing Brainfuck line: {e}\n{traceback.format_exc()}")
                return {"error": f"Brainfuck Processing Error: {e}", "is_error": True}

        # Otherwise, assume it's a sequence of game commands
        else:
            return self.execute_command_sequence(expanded_line)
