from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, Staging! This should be the newly deployed version now"
