from flask import Flask
import os
import sys

app = Flask(__name__)


@app.get("/")
def ok() -> str:
    return "ok\n"


@app.post("/fail")
def fail() -> str:
    sys.stderr.write("FAIL requested\n")
    sys.stderr.flush()
    os._exit(1)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
