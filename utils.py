from datetime import datetime
import sys


def log(msg, file=sys.stdout):
    now = datetime.now()
    display_now = str(now).split(" ")[1][:-3]
    print(display_now + ' ' + msg, file=file)
