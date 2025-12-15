"""
Flask web server for the AI Nutrition Assistant.

Provides web interface for recipe generation, viewing results,
and downloading recipes as PDFs.
"""

from flask import Flask, request, render_template, send_file
from fpdf import FPDF
from dotenv import load_dotenv
load_dotenv()

from app import run_agent
import json
import os
import tempfile

# ============================================================================
# Helper Functions
# ============================================================================
def new_memory():
    """Fresh memory snapshot for the web layer."""
    return {
        "normalized_ingredients": [],
        "ingredient_categories": {},
        "recipes_fetched": [],
        "images_found": [],
        "summaries_created": [],
        "reflections": [],
        "adapted_recipes": [],
        "constraints_applied": [],
        "nutrition_values": []
    }

# ============================================================================
# Flask Application Setup
# ============================================================================
# Run this code and open http://127.0.0.1:5000/ in your browser to access the AI nutrition assistant.
app = Flask(__name__)

# ============================================================================
# Route Handlers
# ============================================================================
@app.route("/")
def home():
    return render_template("results.html", goal=None, memory={
    "normalized_ingredients": [],
    "ingredient_categories": {},
    "recipes_fetched": [],
    "images_found": [],
    "summaries_created": [],
    "reflections": [],
    "adapted_recipes": [],
    "constraints_applied": [],
    "nutrition_values": []
}, error=None)


# Runs the agent with the user's input and displays the generated results
@app.route("/run", methods=["GET", "POST"])
def run():
    global memory
    memory = new_memory()  # ensure clean state per request
    
    goal = request.form.get("goal", "") or request.args.get("goal", "")
    image_path = None

    # Handle image upload
    if "ingredient_image" in request.files:
        img = request.files["ingredient_image"]
        if img and img.filename:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            img.save(tmp.name)
            image_path = tmp.name
            print(f"📸 Received image upload: {image_path}")

    if not goal and not image_path:
        return render_template(
            "results.html",
            goal=None,
            memory={},
            error="Please enter a dish or upload an image."
        )

    # Run the agent (now supports text or image)
    result = run_agent(goal, image_path=image_path)
    memory = result["memory"]

    # Clean up temp image
    if image_path and os.path.exists(image_path):
        os.remove(image_path)

    rendered_page = render_template(
        "results.html",
        goal=goal,
        memory=memory,
        error=None
    )

    return rendered_page

# Displays previously saved results from recipe_results.json
@app.route("/results")
def show_results():
    json_path = "recipe_results.json"

    if not os.path.exists(json_path):
        return "⚠️ No results file found. Run the agent first."

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    goal = data.get("goal", "")
    memory = data.get("memory", {})

    return render_template("results.html", goal=goal, memory=memory, error=None)


@app.route("/download_recipe")
def download_recipe():
    title = request.args.get("title", "Recipe")
    ingredients = request.args.getlist("ingredient")
    instructions = request.args.get("instructions", "")
    image_url = request.args.get("image", None)
    
    calories = request.args.get("calories")
    protein_g = request.args.get("protein_g")
    carbs_g = request.args.get("carbs_g")
    fat_g = request.args.get("fat_g")
    summary = request.args.get("summary")
    health_verdict = request.args.get("health_verdict")


    # --- helper: make text safe for FPDF's latin-1 fonts ---
    def to_latin1(text: str) -> str:
        if text is None:
            return ""
        # replace unsupported characters with '?'
        return text.encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title banner
    pdf.set_fill_color(45, 106, 79)  
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 15, to_latin1(title), ln=True, align="C", fill=True)
    pdf.ln(4)

    pdf.set_text_color(0, 0, 0)


    # Ingredients section
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Ingredients", ln=True)
    pdf.set_draw_color(150, 200, 180)
    pdf.set_line_width(0.7)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Arial", "", 12)
    for ing in ingredients:
        pdf.multi_cell(0, 8, to_latin1(f"- {ing}"))

    pdf.ln(6)

    # Instructions section
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Instructions", ln=True)
    pdf.set_draw_color(150, 200, 180)
    pdf.set_line_width(0.7)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font("Arial", "", 12)

    cleaned_instructions = (
        instructions
        .replace("<p>", "")
        .replace("</p>", "\n\n")
        .replace("<br>", "\n")
        .replace("<br/>", "\n")
    )
    pdf.multi_cell(0, 8, to_latin1(cleaned_instructions))

    pdf.ln(6)

    if calories or protein_g or carbs_g or fat_g:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Nutritional Information", ln=True)
        pdf.set_draw_color(150, 200, 180)
        pdf.set_line_width(0.7)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(6)
        pdf.set_font("Arial", "", 12)

        if calories:
            pdf.cell(0, 8, to_latin1(f"Calories: {calories} kcal"), ln=True)
        if protein_g:
            pdf.cell(0, 8, to_latin1(f"Protein: {protein_g} g"), ln=True)
        if carbs_g:
            pdf.cell(0, 8, to_latin1(f"Carbohydrates: {carbs_g} g"), ln=True)
        if fat_g:
            pdf.cell(0, 8, to_latin1(f"Fat: {fat_g} g"), ln=True)

        pdf.ln(6)

        if summary or health_verdict:
            if summary:
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 8, "Summary:", ln=True)
                pdf.set_font("Arial", "", 12)
                pdf.multi_cell(0, 8, to_latin1(summary))
                pdf.ln(4)

            if health_verdict:
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 8, "Health Verdict:", ln=True)
                pdf.set_font("Arial", "", 12)
                pdf.multi_cell(0, 8, to_latin1(health_verdict))
                pdf.ln(6)
                
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)

    return send_file(
        tmp.name,
        as_attachment=True,
        download_name=f"{title}.pdf"
    )

# ============================================================================
# Application Entry Point
# ============================================================================
if __name__ == "__main__":
    app.run(port=5000, debug=True)