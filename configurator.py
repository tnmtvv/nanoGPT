import sys
from ast import literal_eval

# Skip the script name
argv = sys.argv[1:]

# Process config file if provided as positional argument
config_file = None
for i, arg in enumerate(argv):
    if arg == '--config' and i + 1 < len(argv):
        config_file = argv[i + 1]
        break

if config_file:
    print(f"Overriding config with {config_file}:")
    with open(config_file) as f:
        content = f.read()
        print(content)
    exec(content)

# Then handle --key=value overrides
for arg in argv:
    if '=' in arg and arg.startswith('--'):
        key, val = arg.split('=', 1)
        key = key[2:]
        if key in globals():
            try:
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                attempt = val
            assert type(attempt) == type(globals()[key])
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
