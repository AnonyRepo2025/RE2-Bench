def get_random_string(length, allowed_chars=RANDOM_STRING_CHARS):
    return "".join(secrets.choice(allowed_chars) for i in range(length))