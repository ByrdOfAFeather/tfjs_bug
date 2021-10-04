from flask import Flask, render_template, send_file, send_from_directory, safe_join, abort

app = Flask(__name__)

@app.route("/")
def serve_model(name=None):
	return render_template("val_model.html", name=name)

@app.route("/download/<file_name>")
def load_model(file_name):
	try:
		return send_from_directory(directory="longleaf_model/tfjs_conversion", path="model.json")
	except FileNotFoundError:
		abort(404)

app.run()