import os
import time
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from caption_engine.model import generate_caption

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "change_this_in_production"


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    image_url = None

    if request.method == "POST":
        style = request.form.get("style", "realistic").lower()

        if "image" not in request.files:
            flash("No file part in the request.")
            return redirect(request.url)

        file = request.files["image"]

        if file.filename == "":
            flash("No image selected.")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            original = secure_filename(file.filename)
            unique_name = f"{int(time.time())}_{original}"
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
            file.save(save_path)

            try:
                caption = generate_caption(save_path, style=style)
            except Exception as e:
                print("Error while generating caption:", e)
                caption = "Sorry, something went wrong while generating the caption."

            image_url = url_for("static", filename=f"uploads/{unique_name}")
        else:
            flash("Unsupported file type. Please upload PNG, JPG, JPEG, or GIF.")
            return redirect(request.url)

    return render_template("index.html", caption=caption, image_url=image_url)


if __name__ == "__main__":
    app.run(debug=True)
