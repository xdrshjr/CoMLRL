import ast
import contextlib
import io
import re
import signal
from typing import List


def cleanup_code(code):
    """Remove markdown code blocks and other non-executable parts from code."""
    code = re.sub(r"```python\s*", "", code)
    code = re.sub(r"```\s*", "", code)

    lines = code.split("\n")
    cleaned_lines = []
    for line in lines:
        if re.match(r"^[A-Z][^:=]*$", line.strip()) or re.match(
            r"^Here is", line.strip()
        ):
            continue
        if line.strip() and not re.match(
            r"^(import|from|def|class|if|else|elif|for|while|try|except|#|\s+|return|print)",
            line.strip(),
        ):
            continue
        cleaned_lines.append(line)

    code = "\n".join(cleaned_lines)

    code = re.split(
        r"In this example:|As you can see:|The function is defined|This function|We can use",
        code,
    )[0]

    code = re.sub(r"Example usage:", "# Example usage", code)
    code = re.sub(r"Now we can test", "# Test", code)

    return code.strip()


def extract_print_statements(code):
    """
    Extract print statements and their expected outputs from code.
    Returns a list of tuples (print_line, expected_output)
    """
    # Find all print lines with expected output comments
    lines = code.split("\n")
    print_statements = []

    for line in lines:
        line = line.strip()
        if line.startswith("print(") and "#" in line:
            # Split by comment
            code_part, comment_part = line.split("#", 1)

            print_stmt = code_part.strip()

            expected_match = re.search(
                r"Expected\s+output:?\s*(.*)", comment_part, re.IGNORECASE
            )
            if expected_match:
                expected_output = expected_match.group(1).strip()
                print_statements.append((print_stmt, expected_output))
            elif comment_part.strip():
                expected_output = comment_part.strip()
                print_statements.append((print_stmt, expected_output))

    return print_statements


class TimeoutException(Exception):
    """Exception raised when code execution times out."""

    pass


def timeout_handler(signum, frame):
    """Signal handler for timeouts."""
    raise TimeoutException("Code execution timed out")


def deduplicate_tests(print_statements):
    """Deduplicate test cases by normalizing and comparing them."""
    unique_tests = []
    seen_tests = set()

    for stmt, expected in print_statements:
        func_call_match = re.search(r"print\((.*)\)", stmt)
        if func_call_match:
            func_call = func_call_match.group(1).strip()
            normalized_call = re.sub(r"\s+", "", func_call)

            if normalized_call not in seen_tests:
                seen_tests.add(normalized_call)
                unique_tests.append((stmt, expected))

    return unique_tests


def extract_imported_libraries(import_code):
    """
    Extract library names from import statements.
    Returns a set of imported library names.
    """
    libraries = set()
    if not import_code:
        return libraries

    lines = import_code.split("\n")
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("import "):
            lib_part = line[7:].strip()
            for lib in lib_part.split(","):
                lib_name = lib.strip().split(".")[0].split(" as ")[0]
                if lib_name and not lib_name.startswith("#"):
                    libraries.add(lib_name)
        elif line.startswith("from "):
            match = re.match(r"from\s+(\w+)", line)
            if match:
                lib_name = match.group(1)
                libraries.add(lib_name)

    if libraries:
        print(f"DEBUG: Extracted libraries: {libraries}")
    else:
        print(f"DEBUG: No libraries extracted from code: '{import_code}'")

    return libraries


def check_library_usage(function_code, imported_libraries):
    """
    Check if the function code uses any of the imported libraries.
    Returns True if at least one imported library is used.
    """
    if not imported_libraries or not function_code:
        return False

    function_lower = function_code.lower()

    for lib in imported_libraries:
        if re.search(rf"\b{re.escape(lib.lower())}\b\.", function_lower):
            print(f"DEBUG: Found library usage: {lib} (dot notation)")
            return True

        common_aliases = {
            "numpy": ["np"],
            "pandas": ["pd"],
            "matplotlib": ["plt"],
            "seaborn": ["sns"],
        }

        if lib.lower() in common_aliases:
            for alias in common_aliases[lib.lower()]:
                if re.search(rf"\b{re.escape(alias)}\b\.", function_lower):
                    print(f"DEBUG: Found library usage: {lib} via alias {alias}")
                    return True

    print(f"DEBUG: No library usage found for {imported_libraries} in function code")
    return False


def execution_reward(completion1: List[str], completion2: List[str]) -> List[float]:
    """
    Reward function:
    - +0.5 reward if the combined code is runnable
    - +0.1 for each correct test output (up to 5 unique tests total)
    - 300 second timeout for initial code execution
    - If timeout occurs, treat as failing all tests
    """
    rewards = []
    # Set timeout threshold (300 seconds for initial execution)
    TIMEOUT_SECONDS = 300

    for c1, c2 in zip(completion1, completion2):
        reward = 0.0

        c1_clean = cleanup_code(c1)
        c2_clean = cleanup_code(c2)

        print("\n--- Testing Code Generation ---")

        function_match = re.search(r"def\s+(\w+)\s*\(", c1_clean)
        if not function_match:
            print("No function definition found in first completion")
            rewards.append(reward)
            continue

        function_name = function_match.group(1)

        function_used = (
            re.search(r"\b" + re.escape(function_name) + r"\s*\(", c2_clean) is not None
        )
        function_redefined = (
            re.search(r"def\s+" + re.escape(function_name) + r"\s*\(", c2_clean)
            is not None
        )

        if not function_used:
            print(f"Function {function_name} is not used in second completion")
            rewards.append(reward)
            continue

        if function_redefined:
            print(f"Function {function_name} is redefined in second completion")
            rewards.append(reward)
            continue

        test_code = c2_clean if function_redefined else (c1_clean + "\n\n" + c2_clean)

        if len(test_code) < 1000:
            print("\n--- Combined Test Code ---")
            print(test_code)

        try:
            ast.parse(test_code)

            reward += 0.5
            print(f"✓ Syntax valid: +0.5 reward")

            print_statements = extract_print_statements(test_code)
            unique_print_statements = deduplicate_tests(print_statements)
            unique_print_statements = unique_print_statements[:5]

            if unique_print_statements:
                print(f"Found {len(unique_print_statements)} unique test(s) to run")

                local_vars = {}
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(TIMEOUT_SECONDS)

                    exec(test_code, local_vars)

                    signal.alarm(0)
                    print(
                        f"✓ Initial code execution completed within {TIMEOUT_SECONDS}s"
                    )

                    for i, (print_stmt, expected_output) in enumerate(
                        unique_print_statements
                    ):
                        test_env = dict(local_vars)

                        try:
                            stdout_buffer = io.StringIO()
                            with contextlib.redirect_stdout(stdout_buffer):
                                exec(print_stmt, test_env)

                            actual_output = stdout_buffer.getvalue().strip()
                            expected_output = expected_output.strip()

                            def extract_number(text):
                                """Extract the first number from text, return as string"""
                                match = re.search(r"-?\d+(?:\.\d+)?", text)
                                return match.group(0) if match else text.strip()

                            expected_num = extract_number(expected_output)
                            actual_num = extract_number(actual_output)

                            outputs_match = expected_num == actual_num

                            result_symbol = "✓" if outputs_match else "✗"
                            print(f"Test {i + 1}: {result_symbol} Input: {print_stmt}")
                            print(
                                f"  Expected: {expected_output} | Actual: {actual_output}"
                            )

                            if outputs_match:
                                reward += 0.1
                                print(f"  Passed: +0.1 reward, current total: {reward}")

                        except Exception as e:
                            print(
                                f"Test {i + 1}: ✗ Input: {print_stmt} (Error: {str(e)})"
                            )

                except TimeoutException:
                    print(
                        f"✗ Initial code execution timed out after {TIMEOUT_SECONDS}s - failing all tests"
                    )
                    signal.alarm(0)

                except RecursionError as e:
                    print(
                        f"✗ Initial code execution failed due to recursion limit: {str(e)}"
                    )
                    signal.alarm(0)

                except Exception as e:
                    print(f"✗ Initial code execution failed: {str(e)}")
                    signal.alarm(0)
            else:
                print("No test prints with expected outputs found")

        except SyntaxError as e:
            print(f"✗ Syntax error: {str(e)}")
            rewards.append(reward)
            continue

        print(f"Final reward: {reward}")
        rewards.append(reward)

    return rewards
