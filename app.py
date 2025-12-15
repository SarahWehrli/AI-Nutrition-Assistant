"""
AI Nutrition Assistant - Multi-Agent Recipe System

This module implements a multi-agent system for recipe generation, adaptation,
and nutrition analysis using OpenAI GPT models and Spoonacular MCP.
"""

from openai import OpenAI
import json
import os
import re
import requests
import asyncio
import base64
import mimetypes
import copy

from contextlib import AsyncExitStack
from mcp import stdio_client, ClientSession, StdioServerParameters
from security_utils import safe_get

# ============================================================================
# MCP Server Configuration
# ============================================================================
def mcp_server_params():
    return StdioServerParameters(
        command="docker",
        args=[
            "run",
            "--rm",
            "-i",

            # Sandbox hardening
            "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "--pids-limit", "64",
            "--cpus", "0.5",
            "--memory", "256m",
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",

            # Only required secret
            "-e", f"SPOONACULAR_API_KEY={os.environ['SPOONACULAR_API_KEY']}",

            # Sandbox image
            "spoonacular-mcp",
        ],
        env={},  
    )

async def safe_call_tool(session, tool, params, timeout=10):
    """Safely call an MCP tool with timeout protection."""
    return await asyncio.wait_for(
        session.call_tool(tool, params),
        timeout=timeout
    )

# ============================================================================
# OpenAI Configuration
# ============================================================================
client = OpenAI()
MODEL = "gpt-5-mini"

# ============================================================================
# Memory Management
# ============================================================================
def fresh_memory():
    """Return a brand-new, empty memory store."""
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

# Global memory Store (reset before every request)
memory = fresh_memory()

# ============================================================================
# Helper Constants and Utilities
# ============================================================================

# Natural language quantity mapping for recipe count extraction
NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "couple": 2, "a couple": 2, "a few": 3, "few": 3, "several": 4
}

# JSON Parser
def safe_json_parse(text):
    """Best-effort JSON parse that returns raw text on failure."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON", "raw_output": text}
    

def extract_requested_count(text: str, default: int = 5, hard_cap: int = 10) -> int:
    """
    Infer a user-requested count from free text like:
    'give me two recipes', '3 ideas', 'a couple', 'several', etc.
    Returns default if no signal is found. Applies a sensible hard cap.
    """
    t = text.lower().strip()

    # 1) numeric digits
    m = re.search(r"\b(\d{1,2})\b", t)
    if m:
        n = int(m.group(1))
        return max(1, min(n, hard_cap))

    # 2) number words / quantifiers
    for phrase, n in NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(phrase)}\b", t):
            return max(1, min(n, hard_cap))

    # 3) fallback to default
    return default


def detect_nutrition_intent(goal: str) -> bool:
    """
    Detect if the user's goal implies interest in nutritional values or health analysis.
    """
    keywords = [
        "nutrition", "nutritional", "nutritious", "calorie", "calories",
        "macro", "macros", "protein", "carbs", "fat", "fats",
        "healthy", "healthier", "diet", "nutrient", "values"
    ]
    text = goal.lower()
    return any(k in text for k in keywords)


def clean_summary(text: str) -> str:
    """Remove unwanted HTML and 'Try ... for similar recipes' lines from Spoonacular summaries."""
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove 'Try ... similar recipes' section
    text = re.sub(r"Try\s.+?similar recipes\.*", "", text, flags=re.IGNORECASE | re.DOTALL)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def has_explicit_constraints(goal: str) -> bool:
    keywords = [
        "vegetarian", "vegan", "gluten-free", "gluten free",
        "dairy-free", "keto", "halal", "kosher", "paleo"
    ]
    goal_l = goal.lower()
    return any(k in goal_l for k in keywords)

def get_foodish_image(title: str) -> str:
    """
    Always returns a valid Foodish image URL.
    Falls back to picsum.photos if Foodish API is unavailable.
    """
    FOODISH_BASE = "https://foodish-api.com/api/"
    category_map = {
        "burger": "burger", "pasta": "pasta", "pizza": "pizza",
        "salad": "salad", "rice": "rice", "dessert": "dessert",
        "chicken": "burger",  # chicken often shown under 'burger'
        "curry": "pasta", "soup": "soup"
    }
    try:
        title_l = title.lower()
        cat = next((v for k, v in category_map.items() if k in title_l), None)
        url = f"{FOODISH_BASE}images/{cat}" if cat else FOODISH_BASE
        r = safe_get(url, timeout=5)
        if r.ok:
            data = r.json()
            if "image" in data:
                return data["image"]
    except Exception:
        pass
    # guaranteed fallback
    safe_kw = re.sub(r"[^a-z0-9]+", "-", title.lower())
    return f"https://picsum.photos/seed/{safe_kw}/600/400"


# ============================================================================
# MCP Recipe Fetching Functions
# ============================================================================
async def fetch_recipe_details_mcp(recipe_id: int):
    """
    Calls Spoonacular MCP get_recipe_information to fetch
    ingredients, instructions, summary, image
    """
    server_params = mcp_server_params()

    async with AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(read, write))

        await session.initialize()

        params = {
            "id": recipe_id,
            "includeNutrition": False
        }

        result = await safe_call_tool(session, "get_recipe_information", params)
        return result.content

# ============================================================================
# Agent Implementations
# ============================================================================

# Ingredient Agent
def call_ingredient_agent(goal: str, context: dict, image_path: str | None = None) -> dict:
    """
    IngredientAgent with full agentic behavior including
    dietary constraint detection (vegan, vegetarian, etc.).
    """
    sys_prompt = f"""
        You are IngredientAgent.

        The user may provide a text request, an image of ingredients, or both.
        User request: "{goal}"

        Your task:
        - If text is provided: extract all ingredients mentioned in the request.
        - If an image is provided: visually identify as many recognizable ingredients as possible.
        - Normalize ingredient names.
        - Categorize each ingredient.
        - Detect dietary constraints, allergies or ingredients to avoid.

        Valid categories:
        vegetable, fruit, protein, grain, spice, dairy,
        herb, fat/oil, sweetener, other.
        
        Valid constraints:
        vegan, vegetarian, pescetarian, keto, paleo, halal,
        kosher, gluten-free, dairy-free.

        Return JSON ONLY:
        {{
          "normalized_ingredients": [str],
          "categories": {{ "ingredient": "category" }},
          "constraints": [str],
          "reflections": [str]
        }}
    """

    # Build a multimodal message with optional text and image
    user_content = []
    if goal and goal.strip():
        user_content.append({"type": "text", "text": f"Analyze this user request for ingredients: {goal}"})
    if image_path:
        mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
        data_url = f"data:{mime};base64,{image_base64}"
        user_content.append({
            "type": "image_url",
            "image_url": {"url": data_url}
        })
        
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content}
        ]
    )

    output = safe_json_parse(resp.choices[0].message.content.strip())

    # Update memory
    context["normalized_ingredients"] = output.get("normalized_ingredients", [])
    context["ingredient_categories"] = output.get("categories", {})
    context["constraints_detected"] = output.get("constraints", [])
    context["constraints_applied"] = []
    context["reflections"].extend(output.get("reflections", []))
    if image_path:
        context["images_found"].append(image_path)

    return output


# Spoonacular MCP Call
async def fetch_recipes_with_spoonacular_mcp(user_query: str, diet: str | None = None, goal_text: str | None = None):
    """
    This function performs the *search* step of the recipe pipeline. 
    It sends a query to the Spoonacular MCP "search_recipes" tool and 
    retrieves a list of matching recipes (IDs, titles, images, etc.),but NOT full recipe details.
    """
    server_params = mcp_server_params()

    async with AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(stdio_client(server_params))
        session = await stack.enter_async_context(ClientSession(read, write))

        await session.initialize()

        num = extract_requested_count(goal_text or user_query, default=5, hard_cap=10)
        
        params = {"query": user_query, "number": num}
        if diet:
            params["diet"] = diet

        result = await safe_call_tool(session, "search_recipes", params)
        return result.content


# Convert MCP to recipe dictionary
def convert_mcp_to_internal(mcp_results):
    recipes = []
    for r in mcp_results:
        recipes.append({
            "title": r.get("title", "Untitled Recipe"),
            "ingredients": [i["original"] for i in r.get("extendedIngredients", [])],
            "instructions": r.get("instructions") or "No instructions provided.",
            "image": r.get("image", ""),
            "summary": (r.get("summary") or "").replace("<b>", "").replace("</b>", "")
        })
    return recipes


# Recipe Agent 
def call_recipe_agent(goal: str, context: dict) -> dict:
    """
    Fetches recipes from Spoonacular based on the user's goal and the
    ingredients/constraints already stored in memory.

    Steps:
    1. Build a search query (using extracted ingredients or the raw goal).
    2. Fetch recipes *without* applying dietary filters.
    3. Store results and reflections.
    4. ConstraintAgent will handle adaptation later if constraints exist.
    5. If MCP fails, fall back to OpenAI-based generation.
    """

    # Build search query from extracted ingredients (fallback to raw goal)
    if context.get("normalized_ingredients"):
        user_query = ", ".join(context["normalized_ingredients"])
    else:
        user_query = goal

    requested_n = extract_requested_count(goal, default=3, hard_cap=10)
    already = len(context["recipes_fetched"])
    to_fetch = max(0, requested_n - already)

    if to_fetch == 0:
        return {"recipes": [], "reflections": ["No more recipes needed."]}

    detailed_recipes = []

    # MCP Spoonacular fetch
    try:
        raw_search = asyncio.run(fetch_recipes_with_spoonacular_mcp(user_query, diet=None, goal_text=goal))

        if isinstance(raw_search, list) and hasattr(raw_search[0], "text"):
            search_json = safe_json_parse(raw_search[0].text)
        else:
            raise ValueError("Invalid MCP response")

        results = search_json.get("results", [])

        for entry in results:
            recipe_id = entry.get("id")
            if not recipe_id:
                continue

            details_raw = asyncio.run(fetch_recipe_details_mcp(recipe_id))

            if isinstance(details_raw, list) and hasattr(details_raw[0], "text"):
                details = safe_json_parse(details_raw[0].text)
            else:
                details = details_raw

            detailed_recipes.append({
                "title": details.get("title", "Untitled Recipe"),
                "ingredients": [i.get("original", "") for i in details.get("extendedIngredients", [])],
                "instructions": details.get("instructions") or "No instructions available.",
                "image": details.get("image", ""),
                "summary": clean_summary(details.get("summary")),
                "ready_in_minutes": details.get("readyInMinutes"),
            })

        # Normal MCP path succeeded
        if detailed_recipes:
            context["recipes_fetched"].extend(detailed_recipes)
            reflection = [
                f"Fetched {len(detailed_recipes)} recipes (no dietary filtering).",
                "Recipes will be adapted later by ConstraintAgent if needed."
            ]
            context["reflections"].extend(reflection)
            return {"recipes": detailed_recipes, "reflections": reflection}
        else:
            raise RuntimeError("MCP returned no recipes.")

    except Exception as e:
        context["reflections"].append(f"⚠️ MCP fetch failed: {e}. Switching to fallback recipe generator.")

    # FALLBACK PATH: OpenAI-based generation
    fallback_prompt = f"""
    You are RecipeAgent-Fallback.

    User goal: "{goal}"
    Constraints: {context.get("constraints_applied", [])}
    requested_n = {extract_requested_count(goal, default=3, hard_cap=10)}
    
    Your task:
    - Use all the ingredients already identified in {memory["normalized_ingredients"]}.
    - Generate {requested_n} recipes that fit the user's goal.
    - Do **not** apply dietary filters, the constraints will be handled later by the constraint agent.
    - For each recipe:
    - Provide: title, ingredients, instructions, summary, ready_in_minutes, and image.
    - Leave the image field blank; it will be automatically filled.
   
    Return ONLY JSON:
    {{
      "recipes": [
        {{
          "title": str,
          "ingredients": [str],
          "instructions": str,
          "summary": str,
          "image": "",
          "ready_in_minutes": 10
        }}
      ],
      "reflections": [str]
    }}
    """

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": fallback_prompt},
            {"role": "user", "content": f"Generate the recipes matching: '{goal}'"}
        ]
    )

    fallback_data = safe_json_parse(resp.choices[0].message.content.strip())
    recipes = fallback_data.get("recipes", [])
    reflections = fallback_data.get("reflections", [])
    
    # Ensure every fallback recipe has a displayable image URL
    for r in recipes:
        r["image"] = get_foodish_image(r.get("title", "dish"))

    context["recipes_fetched"].extend(recipes)
    context["reflections"].extend(reflections)
    context["reflections"].append("✅ Used OpenAI fallback recipe generator due to MCP failure.")

    return {"recipes": recipes, "reflections": reflections}

# Constraint Agent
def call_constraint_agent(goal: str, context: dict) -> dict:
    """
    Adapts all fetched recipes to match the user's dietary constraints.
    The adapted recipes replace the originals in memory["recipes_fetched"].
    """

    if not context.get("recipes_fetched"):
        return {"adapted_recipes": [], "reflections": ["No recipes available to adapt."]}

    recipes = context["recipes_fetched"]
    constraints = context.get("constraints_applied", [])

    sys_prompt = f"""
You are ConstraintAgent.

User goal: "{goal}"

Detected constraints: {constraints}

Shared memory (for context only):
{json.dumps(context, indent=2)}

Your task:
- For each recipe in memory["recipes_fetched"], modify it so it fully complies with ALL user constraints.
- Remove any forbidden ingredients (e.g. meat, dairy, eggs, fish for vegan).
- Substitute with realistic plant-based or compliant alternatives.
- Ensure instructions reflect those substitutions.
- Keep titles meaningful, e.g., “Vegan {{recipe_name}}”.
- Return a list of adapted recipes in valid JSON.

Return ONLY JSON:
{{
  "adapted_recipes": [
    {{
      "title": "string",
      "ingredients": ["string"],
      "instructions": "string",
      "summary": "string",
      "image": "string"
    }}
  ],
  "reflections": ["string"]
}}
"""

    # Serialize all fetched recipes for the model
    recipe_text = json.dumps(recipes, indent=2)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Here are the recipes to adapt:\n{recipe_text}"}
        ]
    )

    output = safe_json_parse(resp.choices[0].message.content.strip())

    adapted_recipes = output.get("adapted_recipes", [])

    # Preserve original prep times when adaptation does not provide them
    if adapted_recipes and len(adapted_recipes) == len(recipes):
        for i, adapted in enumerate(adapted_recipes):
            if "ready_in_minutes" not in adapted:
                adapted["ready_in_minutes"] = recipes[i].get("ready_in_minutes")
    reflections = output.get("reflections", [])

    # Replace the fetched recipes with adapted ones
    context["adapted_recipes"] = adapted_recipes
    context["recipes_fetched"] = adapted_recipes
    context["reflections"].extend(reflections)
    context["reflections"].append("✅ Adapted all fetched recipes to comply with constraints.")

    return output

# Nutrition Agent
def call_nutrition_agent(goal: str, context: dict) -> dict:
    sys_prompt = f"""
        You are NutritionAgent. Your goal is: {goal}.
        You have the following memory: {json.dumps(context, indent=2)}

        Act autonomously to achieve the goal. You can:
        - Fetch nutritional information for each recipe in memory ["recipes_fetched"]
        - Infer healthiness using nutritional guidelines
        - Estimate nutritional values (calories, protein, carbs, fats)
        - Output a nutrition summary and a clear health verdict

        After generating output, reflect:
        - Did you fully satisfy the goal?
        - If not, propose a correction.

        Return JSON:
        {{
            "nutrition_values": {{
                "title": str,
                "calories": float,
                "protein_g": float,
                "carbs_g": float,
                "fat_g": float,
                "summary": str,
                "health_verdict": str
            }},
            "reflections": [str]
        }}
    """

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Execute your tasks to satisfy this goal: {goal}"}
        ]
    )

    output = safe_json_parse(resp.choices[0].message.content.strip())
    context["nutrition_values"].append(output.get("nutrition_values", {}))
    context["reflections"].extend(output.get("reflections", []))

    return output


# ============================================================================
# Tool Definitions and Mapping
# ============================================================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "process_ingredients",
            "description": "Extract, normalize, and categorize ingredients.",
            "parameters": {"type": "object", "properties": {"goal": {"type": "string"}}, "required": ["goal"]}
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_recipes",
            "description": "Fetch recipes from Spoonacular MCP.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adapt_recipe",
            "description": "Adapt a recipe based on diet constraints.",
            "parameters": {
                "type": "object",
                "properties": {"recipe_text": {"type": "string"}, "user_constraints": {"type": "string"}},
                "required": ["recipe_text", "user_constraints"]
            }
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_nutrition",
            "description": "Analyze if recipe is healthy.",
            "parameters": {
                "type": "object",
                "properties": {"recipe_text": {"type": "string"}},
                "required": ["recipe_text"]
            }
        },
    },
]

tool_map = {
    "process_ingredients": call_ingredient_agent,
    "fetch_recipes": call_recipe_agent,
    "adapt_recipe": call_constraint_agent,
    "analyze_nutrition": call_nutrition_agent
}

# ============================================================================
# Manager Agent Loop
# ============================================================================
def manager_loop(user_goal: str, max_rounds: int = 5):
    print(f"\nStarting autonomous reasoning for: {user_goal}\n")
    history = []
    
    requested_n = extract_requested_count(user_goal, default=3, hard_cap=10)

    for round in range(max_rounds):
        sys_prompt = f"""
        You are RecipeManager.
        
        User goal: "{user_goal}"
        The user requested approximately {requested_n} recipes.
        The user explicitly requested nutritional or calorie information: {detect_nutrition_intent(user_goal)}.
        
        Shared memory:
        {json.dumps(memory, indent=2)}

        You can call:
        - process_ingredients: Extract and normalize all ingredients, dietary constraints, and allergies.
        - fetch_recipes: Fetch recipes from Spoonacular MCP.
        - adapt_recipe: Adapt recipes to meet dietary constraints or allergies.
        - analyze_nutrition: Analyze nutritional values of the fetched recipes.

        Instructions:
        - Think step by step:
        - Decide what to do next to best achieve the user's goal.
        - Call only the tools necessary to fulfill the user's goal.
        - Stop when the user's goal is fully satisfied.

        Hints:
        - Before fetching recipes, make sure the ingredients are known.
        - Nutrition analysis depends on the existence of recipes. If the user explicitly requested nutritional information, plan to analyze them after recipes are available.
        - Adaptation to constraints is optional and should only occur if dietary restrictions are detected.
        - Avoid repeating the same action unless it clearly serves progress.
        
        Return JSON:
        {{
            "next_action": "process_ingredients" | "fetch_recipes" |
                           "adapt_recipe" | "analyze_nutrition" | "finish",
            "reasoning": str
        }}
        """

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Round {round+1}: decide next step."}
            ],
            tools=tools,
            tool_choice="auto"
        )

        # Extract raw decision message and tool-calls
        decision_raw = resp.choices[0].message.content or ""
        tool_calls = resp.choices[0].message.tool_calls or []

        # JSON fallback parse
        decision = safe_json_parse(decision_raw)

        # Correct SDK v1+ tool call access
        if tool_calls:
            next_action = tool_calls[0].function.name
        else:
            next_action = decision.get("next_action", "finish")

        print(f"Round {round+1}: {next_action}")
        
        if next_action == "finish":
            break

        # Run the selected agent and capture its output
        agent_fn = tool_map.get(next_action)
        result = agent_fn(goal=user_goal, context=memory)

        # Store history
        history.append({
            "round": round + 1,
            "action": next_action,
            "result": result
        })
        
             # Force constraint adaptation if constraints exist
        if (
            next_action == "fetch_recipes"
            and has_explicit_constraints(user_goal)
            and memory.get("constraints_detected")
            and memory.get("recipes_fetched")
        ):
            print("🌿 Explicit dietary request detected — adapting recipes.")
            memory["constraints_applied"] = memory["constraints_detected"]
            adapt_result = call_constraint_agent(goal=user_goal, context=memory)
            history.append({
                "round": round + 1,
                "action": "adapt_recipe",
                "result": adapt_result
            })

        if len(memory["recipes_fetched"]) >= requested_n:
            print(f"✅ Reached target: {len(memory['recipes_fetched'])} recipes.")
            if not detect_nutrition_intent(user_goal):
                break
            else:
                print("🧠 Nutrition analysis requested — continuing to analyze nutrition.")

    # Final agent state
    result_state = {
        "goal": user_goal,
        "memory": memory,
        "history": history
    }

    with open("recipe_results.json", "w", encoding="utf-8") as f:
        json.dump(result_state, f, indent=2)

    print("\nResults saved to recipe_results.json\n")

    return result_state

# ============================================================================
# Entry Point
# ============================================================================
def run_agent(user_goal: str = "", image_path: str | None = None):
    """
    Entry point for the website.

    This function is called by web_server.py and runs the entire agent pipeline.
    It returns the final result with goal, memory, and history.

    Args:
        user_goal: The user's recipe request text
        image_path: Optional path to an uploaded ingredient image
        
    Returns:
        Dictionary containing goal, memory, and history, or error dict
    """
    if not user_goal or not user_goal.strip():
        return {"error": "No goal provided"}

    # Always start with a fresh memory per request
    global memory
    memory = fresh_memory()

    # Pre-extract ingredients (text and/or image)
    call_ingredient_agent(user_goal, memory, image_path=image_path)

    # Run the manager loop using this isolated memory
    result = manager_loop(user_goal)

    # Snapshot the result memory before clearing
    result_memory = copy.deepcopy(result.get("memory", memory))
    history = result.get("history", [])

    # Hard reset memory so nothing leaks to the next request
    memory = fresh_memory()

    return {"goal": user_goal, "memory": result_memory, "history": history}