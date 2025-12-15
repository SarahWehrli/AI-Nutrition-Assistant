"""
Unit tests for individual agent functions.

Tests each agent (IngredientAgent, RecipeAgent, ConstraintAgent, NutritionAgent)
in isolation to verify correct behavior and memory updates.
"""

import json
import unittest
from unittest.mock import patch, MagicMock

import app

app.fetch_recipes_with_spoonacular_mcp = MagicMock()
app.fetch_recipe_details_mcp = MagicMock()

# ============================================================================
# Test Helpers
# ============================================================================
def fresh_memory():
    return {
        "normalized_ingredients": [],
        "ingredient_categories": {},
        "recipes_fetched": [],
        "images_found": [],
        "summaries_created": [],
        "reflections": [],
        "adapted_recipes": [],
        "constraints_applied": [],
        "nutrition_values": [],
    }

def fake_openai_response(payload):
    msg = MagicMock()
    msg.content = json.dumps(payload)
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ============================================================
# INGREDIENT AGENT — Test that ingredient extraction, categorization,
# constraint detection, and reflections all update memory correctly.
# ============================================================
class TestIngredientAgent(unittest.TestCase):

    @patch.object(app.client.chat.completions, "create")
    def test_ingredient_agent_full(self, mock_create):
        payload = {
            "normalized_ingredients": ["tomato"],
            "categories": {"tomato": "vegetable"},
            "constraints": ["vegan"],
            "reflections": ["parsed ingredients"]
        }
        mock_create.return_value = fake_openai_response(payload)

        memory = fresh_memory()
        result = app.call_ingredient_agent("vegan tomato pasta", memory, None)

        self.assertEqual(result["normalized_ingredients"], ["tomato"])
        self.assertEqual(memory["ingredient_categories"]["tomato"], "vegetable")
        self.assertIn("vegan", memory["constraints_applied"])
        self.assertIn("parsed ingredients", memory["reflections"])


# ============================================================
# RECIPE AGENT — Test both branches:
# 1) MCP success path (search + details)
# 2) Fallback path when MCP fails
# Ensures recipes and reflections update memory correctly.
# ============================================================
class TestRecipeAgent(unittest.TestCase):

    @patch("app.fetch_recipe_details_mcp")
    @patch("app.fetch_recipes_with_spoonacular_mcp")
    @patch("app.asyncio.run")
    @patch.object(app.client.chat.completions, "create")
    def test_recipe_agent_mcp_success(self, mock_create, mock_run, mock_fetch_search, mock_fetch_details):

        mock_fetch_search.return_value = [
            MagicMock(text=json.dumps({"results": [{"id": 123}]}))
        ]
        mock_fetch_details.return_value = [
            MagicMock(text=json.dumps({
                "title": "MCP Pasta",
                "extendedIngredients": [{"original": "flour"}],
                "instructions": "Cook.",
                "image": "",
                "summary": "<b>Summary</b>",
                "readyInMinutes": 8
            }))
        ]

        mock_run.side_effect = [
            mock_fetch_search.return_value,
            mock_fetch_details.return_value
        ]

        memory = fresh_memory()
        result = app.call_recipe_agent("give me 1 recipe", memory)

        self.assertEqual(memory["recipes_fetched"][0]["title"], "MCP Pasta")
        self.assertIn("Fetched 1 recipes", " ".join(memory["reflections"]))
        self.assertEqual(len(result["recipes"]), 1)

    @patch.object(app.client.chat.completions, "create")
    @patch("app.asyncio.run", side_effect=Exception("MCP failed"))
    def test_recipe_agent_fallback(self, mock_run, mock_create):
        payload = {
            "recipes": [
                {
                    "title": "Fallback Dish",
                    "ingredients": ["1 cup rice"],
                    "instructions": "Cook rice.",
                    "summary": "fallback summary",
                    "image": "https://example.com/rice.jpg",
                    "ready_in_minutes": 9,
                }
            ],
            "reflections": ["fallback used"]
        }

        mock_create.return_value = fake_openai_response(payload)

        memory = fresh_memory()
        result = app.call_recipe_agent("give me rice recipe", memory)

        self.assertEqual(memory["recipes_fetched"][0]["title"], "Fallback Dish")
        self.assertIn("fallback", " ".join(memory["reflections"]))
        self.assertEqual(len(result["recipes"]), 1)


# ============================================================
# CONSTRAINT AGENT — Test recipe adaptation when constraints exist.
# Ensures adapted recipes replace originals and reflections are recorded.
# ============================================================
class TestConstraintAgent(unittest.TestCase):

    @patch.object(app.client.chat.completions, "create")
    def test_constraint_agent_full(self, mock_create):
        memory = fresh_memory()

        memory["recipes_fetched"] = [
            {
                "title": "Chicken Curry",
                "ingredients": ["chicken"],
                "instructions": "cook chicken",
                "summary": "",
                "image": ""
            }
        ]
        memory["constraints_applied"] = ["vegan"]

        payload = {
            "adapted_recipes": [
                {
                    "title": "Vegan Chicken Curry",
                    "ingredients": ["tofu"],
                    "instructions": "cook tofu",
                    "summary": "vegan version",
                    "image": "",
                }
            ],
            "reflections": ["adapted for vegan diet"]
        }

        mock_create.return_value = fake_openai_response(payload)

        result = app.call_constraint_agent("make vegan", memory)

        self.assertEqual(memory["recipes_fetched"][0]["title"], "Vegan Chicken Curry")
        self.assertIn("adapted", " ".join(memory["reflections"]))
        self.assertEqual(result["adapted_recipes"][0]["ingredients"][0], "tofu")


# ============================================================
# NUTRITION AGENT — Test nutrition extraction and health verdict handling.
# Ensures nutrition values update memory and reflections are stored.
# ============================================================
class TestNutritionAgent(unittest.TestCase):

    @patch.object(app.client.chat.completions, "create")
    def test_nutrition_full(self, mock_create):
        payload = {
            "nutrition_values": {
                "title": "Healthy Bowl",
                "calories": 420,
                "protein_g": 20,
                "carbs_g": 40,
                "fat_g": 10,
                "summary": "Healthy choice",
                "health_verdict": "Good"
            },
            "reflections": ["nutrition computed"]
        }

        mock_create.return_value = fake_openai_response(payload)

        memory = fresh_memory()
        result = app.call_nutrition_agent("analyze health", memory)

        self.assertEqual(result["nutrition_values"]["title"], "Healthy Bowl")
        self.assertEqual(memory["nutrition_values"][0]["protein_g"], 20)
        self.assertIn("nutrition", " ".join(memory["reflections"]))


if __name__ == "__main__":
    unittest.main()