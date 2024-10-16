import re


def convert_to_snake_case(pascal_case):
    """Convert string in PascalCase to snake_case

    Args:
        pascal_case (str): string to convert

    Returns:
        str: converted string
    """
    camel_case = re.sub(r'(?<!^)(?=[A-Z])', '_', pascal_case).lower()
    return re.sub(r'_{2,}', '_', camel_case)  # Removes duplicate underscores


def convert_to_pascal_case(snake_case):
    """Convert string in snake_case to PascalCase

    Args:
        snake_case (str): string to convert

    Returns:
        str: converted string
    """
    return ''.join(c.title() for c in snake_case.split('_'))
