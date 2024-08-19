import sys
from datetime import datetime
from time import sleep, time


def print_log_to_file(log_file, *args, also_print_to_console, add_timestamp):
    timestamp = time()
    dt_object = datetime.fromtimestamp(timestamp)

    if add_timestamp:
        args = (f"{dt_object}:", *args)

    successful = False
    max_attempts = 5
    ctr = 0
    while not successful and ctr < max_attempts:
        try:
            with open(log_file, "a+") as f:
                for a in args:
                    f.write(str(a))
                    f.write(" ")
                f.write("\n")
            successful = True
        except IOError:
            print(
                f"{datetime.fromtimestamp(timestamp)}: failed to log: ", sys.exc_info()
            )
            sleep(0.5)
            ctr += 1
    if also_print_to_console:
        print(*args)
    elif also_print_to_console:
        print(*args)
