from flask import Flask, render_template, jsonify
import yaml
from stream_and_transcribe import main
import whisper

whisper.load_model("medium.en")  # Download model at the start of session
app = Flask(__name__)


def read_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


@app.route("/")
def home():
    data = read_yaml("output.yaml")
    return render_template("form.html", data=data)


# Route to run decoding separately
@app.route("/run-decoding", methods=["POST"])
def run_decoding():
    # Call your decoding function here
    main()  # This will run your decoding process
    return jsonify({"message": "Decoding process completed!"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
