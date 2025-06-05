import sys

def get_method_name(s: str) -> str:
    parts = s.split("/")
    if len(parts) < 2:
        raise ValueError("The string does not have enough parts separated by '/'")
    second_to_last = parts[-2]
    subparts = second_to_last.split(".")
    return subparts[-1]

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <your_string>")
        sys.exit(1)

    input_string = sys.argv[1]
    try:
        result = get_method_name(input_string)
        print("method name:", result)
    except ValueError as e:
        print("Error:", e)

if __name__ == "__main__":
    main()

