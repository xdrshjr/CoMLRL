"""Utility functions for code manipulation and processing."""

import re


def extract_imports_from_prompt(prompt):
    """Extract import statements from the prompt text."""
    if not prompt:
        return ""

    # Find all import statements in the prompt
    import_patterns = [
        r"^from\s+[\w\.]+\s+import\s+.*$",  # from module import ...
        r"^import\s+[\w\.,\s]+$",  # import module, module2
    ]

    imports = []
    lines = prompt.split("\n")

    for line in lines:
        line = line.strip()
        for pattern in import_patterns:
            if re.match(pattern, line):
                imports.append(line)
                break

    # Remove duplicates while preserving order
    seen = set()
    unique_imports = []
    for imp in imports:
        if imp not in seen:
            seen.add(imp)
            unique_imports.append(imp)

    return "\n".join(unique_imports)


def cleanup_code(code):
    """
    Extract function definitions from code that may contain explanatory text,
    markdown formatting, or other non-code content.
    """
    if not code:
        return ""

    # Remove markdown code blocks
    code = re.sub(r"```[a-zA-Z]*\n?", "", code)
    code = re.sub(r"```", "", code)

    # Split into lines and filter
    lines = code.split("\n")
    cleaned_lines = []
    in_function = False
    function_indent = 0

    for line in lines:
        # Check if this is a function definition
        if re.match(r"^\s*def\s+\w+\s*\(", line):
            in_function = True
            # Calculate the indentation level
            function_indent = len(line) - len(line.lstrip())
            cleaned_lines.append(line)
        elif in_function:
            # Check if we're still in the function based on indentation
            current_indent = len(line) - len(line.lstrip())
            
            # Empty lines are included
            if not line.strip():
                cleaned_lines.append(line)
            # Lines with greater indent are part of the function
            elif current_indent > function_indent:
                cleaned_lines.append(line)
            # Lines with equal indent might be continuation
            elif current_indent == function_indent and line.strip() and not line.strip().startswith('#'):
                # Check if it's likely a new statement/function
                if re.match(r"^\s*(def|class|if|for|while|with|try|import|from)\s+", line):
                    in_function = False
                else:
                    cleaned_lines.append(line)
            else:
                # We've exited the function
                in_function = False

    return "\n".join(cleaned_lines)


def concatenate_functions(aux_completion, main_completion, imports=""):
    """Concatenate imports, aux and main functions."""
    aux_clean = cleanup_code(aux_completion)
    main_clean = cleanup_code(main_completion)

    # Build the combined code with imports first
    parts = []

    if imports:
        parts.append(imports)

    if aux_clean:
        parts.append(aux_clean)

    if main_clean:
        parts.append(main_clean)

    combined_code = "\n\n".join(parts)
    return combined_code