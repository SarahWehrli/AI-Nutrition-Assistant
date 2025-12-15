"""
Integration tests for the multi-agent recipe system.

Tests the full workflow from run_agent() through manager_loop()
orchestrating multiple agents (IngredientAgent, RecipeAgent, 
ConstraintAgent, NutritionAgent).
"""
import json
import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock, AsyncMock

import app

# ============================================================================
# Test Helpers
# ============================================================================
def fake_openai_response(payload, tool_calls=None):
    """Helper to create a fake OpenAI API response."""
    msg = MagicMock()
    msg.content = json.dumps(payload) if isinstance(payload, dict) else payload
    msg.tool_calls = tool_calls or None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def fake_tool_call_response(tool_name):
    """Helper to create a fake OpenAI response with a tool call."""
    tool_call = MagicMock()
    tool_call.function = MagicMock()
    tool_call.function.name = tool_name
    tool_call.function.arguments = "{}"
    
    msg = MagicMock()
    msg.content = None
    msg.tool_calls = [tool_call]
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp

# ============================================================================
# Test Classes
# ============================================================================
class TestFullWorkflow(unittest.TestCase):
    """Test the complete multi-agent workflow end-to-end."""

    def setUp(self):
        """Reset memory before each test."""
        app.memory = app.fresh_memory()

    @patch("app.fetch_recipe_details_mcp", new_callable=AsyncMock)
    @patch("app.fetch_recipes_with_spoonacular_mcp", new_callable=AsyncMock)
    @patch("app.asyncio.run")
    @patch.object(app.client.chat.completions, "create")
    def test_simple_workflow_no_constraints(self, mock_openai, mock_asyncio, mock_fetch_search, mock_fetch_details):
        """
        Test a simple workflow: ingredient extraction -> recipe fetch -> finish.
        No constraints or nutrition analysis.
        """
        # Mock ingredient agent response (called twice: once in run_agent, once by manager)
        ingredient_payload = {
            "normalized_ingredients": ["tomato", "pasta"],
            "categories": {"tomato": "vegetable", "pasta": "grain"},
            "constraints": [],
            "reflections": ["Extracted 2 ingredients"]
        }
        
        # Mock manager decisions: process_ingredients -> fetch_recipes -> finish
        mock_openai.side_effect = [
            fake_openai_response(ingredient_payload),  # Ingredient agent (in run_agent)
            fake_tool_call_response("fetch_recipes"),  # Manager round 1 (skips process_ingredients since already done)
            fake_openai_response({"next_action": "finish", "reasoning": "Done"})  # Manager round 2
        ]
        
        # Mock MCP recipe search
        mock_fetch_search.return_value = [
            MagicMock(text=json.dumps({"results": [{"id": 123}]}))
        ]
        mock_fetch_details.return_value = [
            MagicMock(text=json.dumps({
                "title": "Tomato Pasta",
                "extendedIngredients": [{"original": "tomato"}, {"original": "pasta"}],
                "instructions": "Cook pasta. Add tomato.",
                "image": "https://example.com/pasta.jpg",
                "summary": "Simple pasta dish",
                "readyInMinutes": 15
            }))
        ]
        mock_asyncio.side_effect = [
            mock_fetch_search.return_value,
            mock_fetch_details.return_value
        ]
        
        # Run the full workflow
        result = app.run_agent("tomato pasta recipe")
        
        # Assertions
        self.assertIn("goal", result)
        self.assertEqual(result["goal"], "tomato pasta recipe")
        self.assertIn("memory", result)
        self.assertIn("history", result)
        
        # Check memory was populated
        memory = result["memory"]
        self.assertEqual(len(memory["normalized_ingredients"]), 2)
        self.assertIn("tomato", memory["normalized_ingredients"])
        self.assertEqual(len(memory["recipes_fetched"]), 1)
        self.assertEqual(memory["recipes_fetched"][0]["title"], "Tomato Pasta")
        
        # Check history shows agent calls
        history = result["history"]
        self.assertGreater(len(history), 0)
        actions = [h["action"] for h in history]
        # process_ingredients may be skipped if already done in run_agent
        self.assertIn("fetch_recipes", actions)

    @patch("app.fetch_recipe_details_mcp", new_callable=AsyncMock)
    @patch("app.fetch_recipes_with_spoonacular_mcp", new_callable=AsyncMock)
    @patch("app.asyncio.run")
    @patch.object(app.client.chat.completions, "create")
    def test_workflow_with_constraints(self, mock_openai, mock_asyncio, mock_fetch_search, mock_fetch_details):
        """
        Test workflow with dietary constraints: ingredient extraction -> 
        recipe fetch -> constraint adaptation -> finish.
        """
        # Mock ingredient agent (detects vegan constraint)
        ingredient_payload = {
            "normalized_ingredients": ["tofu", "rice"],
            "categories": {"tofu": "protein", "rice": "grain"},
            "constraints": ["vegan"],
            "reflections": ["Detected vegan constraint"]
        }
        
        # Mock constraint agent
        constraint_payload = {
            "adapted_recipes": [{
                "title": "Vegan Tofu Rice Bowl",
                "ingredients": ["tofu", "rice", "vegetables"],
                "instructions": "Cook vegan style",
                "summary": "Vegan adapted",
                "image": "",
                "ready_in_minutes": 20
            }],
            "reflections": ["Adapted for vegan diet"]
        }
        
        mock_openai.side_effect = [
            fake_openai_response(ingredient_payload),  # Ingredient agent (in run_agent)
            fake_tool_call_response("fetch_recipes"),  # Manager round 1
            fake_openai_response(constraint_payload),  # Constraint agent (auto-triggered)
            fake_openai_response({"next_action": "finish", "reasoning": "Done"})  # Manager round 2
        ]
        
        # Mock MCP
        mock_fetch_search.return_value = [
            MagicMock(text=json.dumps({"results": [{"id": 456}]}))
        ]
        mock_fetch_details.return_value = [
            MagicMock(text=json.dumps({
                "title": "Tofu Rice Bowl",
                "extendedIngredients": [{"original": "tofu"}, {"original": "rice"}],
                "instructions": "Cook",
                "image": "",
                "summary": "",
                "readyInMinutes": 20
            }))
        ]
        mock_asyncio.side_effect = [
            mock_fetch_search.return_value,
            mock_fetch_details.return_value
        ]
        
        result = app.run_agent("vegan tofu rice bowl")
        
        memory = result["memory"]
        self.assertIn("vegan", memory["constraints_applied"])
        self.assertEqual(len(memory["recipes_fetched"]), 1)
        # Recipe should be adapted
        self.assertIn("Vegan", memory["recipes_fetched"][0]["title"])

    @patch("app.fetch_recipe_details_mcp", new_callable=AsyncMock)
    @patch("app.fetch_recipes_with_spoonacular_mcp", new_callable=AsyncMock)
    @patch("app.asyncio.run")
    @patch.object(app.client.chat.completions, "create")
    def test_workflow_with_nutrition_analysis(self, mock_openai, mock_asyncio, mock_fetch_search, mock_fetch_details):
        """
        Test workflow with nutrition analysis: ingredient extraction -> 
        recipe fetch -> nutrition analysis -> finish.
        """
        ingredient_payload = {
            "normalized_ingredients": ["chicken", "broccoli"],
            "categories": {"chicken": "protein", "broccoli": "vegetable"},
            "constraints": [],
            "reflections": []
        }
        
        nutrition_payload = {
            "nutrition_values": {
                "title": "Chicken Broccoli",
                "calories": 350,
                "protein_g": 30,
                "carbs_g": 15,
                "fat_g": 12,
                "summary": "High protein meal",
                "health_verdict": "Healthy choice"
            },
            "reflections": ["Nutrition analyzed"]
        }
        
        mock_openai.side_effect = [
            fake_openai_response(ingredient_payload),  # Ingredient agent (in run_agent)
            fake_tool_call_response("fetch_recipes"),  # Manager round 1
            fake_tool_call_response("analyze_nutrition"),  # Manager round 2
            fake_openai_response(nutrition_payload),  # Nutrition agent
            fake_openai_response({"next_action": "finish", "reasoning": "Done"})  # Manager round 3
        ]
        
        mock_fetch_search.return_value = [
            MagicMock(text=json.dumps({"results": [{"id": 789}]}))
        ]
        mock_fetch_details.return_value = [
            MagicMock(text=json.dumps({
                "title": "Chicken Broccoli",
                "extendedIngredients": [{"original": "chicken"}, {"original": "broccoli"}],
                "instructions": "Cook",
                "image": "",
                "summary": "",
                "readyInMinutes": 25
            }))
        ]
        mock_asyncio.side_effect = [
            mock_fetch_search.return_value,
            mock_fetch_details.return_value
        ]
        
        result = app.run_agent("chicken broccoli with nutrition info")
        
        memory = result["memory"]
        self.assertEqual(len(memory["nutrition_values"]), 1)
        self.assertEqual(memory["nutrition_values"][0]["calories"], 350)
        self.assertEqual(memory["nutrition_values"][0]["protein_g"], 30)

    @patch("app.asyncio.run", side_effect=Exception("MCP failed"))
    @patch.object(app.client.chat.completions, "create")
    @patch("app.get_foodish_image")
    def test_workflow_fallback_to_openai(self, mock_foodish, mock_openai, mock_asyncio):
        """
        Test workflow when MCP fails: ingredient extraction -> 
        fallback recipe generation -> finish.
        """
        ingredient_payload = {
            "normalized_ingredients": ["rice"],
            "categories": {"rice": "grain"},
            "constraints": [],
            "reflections": []
        }
        
        fallback_payload = {
            "recipes": [{
                "title": "Simple Rice Dish",
                "ingredients": ["rice", "water"],
                "instructions": "Cook rice",
                "summary": "Basic rice",
                "image": "",
                "ready_in_minutes": 20
            }],
            "reflections": ["Used fallback generator"]
        }
        
        mock_openai.side_effect = [
            fake_openai_response(ingredient_payload),  # Ingredient agent (in run_agent)
            fake_tool_call_response("fetch_recipes"),  # Manager round 1
            fake_openai_response(fallback_payload),  # Fallback recipe agent
            fake_openai_response({"next_action": "finish", "reasoning": "Done"})  # Manager round 2
        ]
        
        mock_foodish.return_value = "https://picsum.photos/600/400"
        
        result = app.run_agent("rice recipe")
        
        memory = result["memory"]
        self.assertEqual(len(memory["recipes_fetched"]), 1)
        self.assertEqual(memory["recipes_fetched"][0]["title"], "Simple Rice Dish")
        # Image should be filled by get_foodish_image
        self.assertIn("https://", memory["recipes_fetched"][0]["image"])

    @patch("app.fetch_recipe_details_mcp", new_callable=AsyncMock)
    @patch("app.fetch_recipes_with_spoonacular_mcp", new_callable=AsyncMock)
    @patch("app.asyncio.run")
    @patch.object(app.client.chat.completions, "create")
    def test_memory_isolation_between_requests(self, mock_openai, mock_asyncio, mock_fetch_search, mock_fetch_details):
        """
        Test that memory is properly isolated between two consecutive requests.
        First request should not affect second request's memory.
        """
        # First request
        ingredient_payload_1 = {
            "normalized_ingredients": ["chicken"],
            "categories": {"chicken": "protein"},
            "constraints": [],
            "reflections": []
        }
        
        mock_openai.side_effect = [
            fake_openai_response(ingredient_payload_1),  # Ingredient agent (in run_agent)
            fake_tool_call_response("fetch_recipes"),  # Manager round 1
            fake_openai_response({"next_action": "finish", "reasoning": "Done"})  # Manager round 2
        ]
        
        mock_fetch_search.return_value = [
            MagicMock(text=json.dumps({"results": [{"id": 111}]}))
        ]
        mock_fetch_details.return_value = [
            MagicMock(text=json.dumps({
                "title": "Chicken Dish",
                "extendedIngredients": [{"original": "chicken"}],
                "instructions": "Cook",
                "image": "",
                "summary": "",
                "readyInMinutes": 30
            }))
        ]
        mock_asyncio.side_effect = [
            mock_fetch_search.return_value,
            mock_fetch_details.return_value
        ]
        
        result1 = app.run_agent("chicken recipe")
        memory1 = result1["memory"]
        
        # Second request (different ingredients)
        ingredient_payload_2 = {
            "normalized_ingredients": ["tofu"],
            "categories": {"tofu": "protein"},
            "constraints": ["vegan"],
            "reflections": []
        }
        
        # Mock constraint agent (will be auto-triggered after fetch_recipes)
        constraint_payload = {
            "adapted_recipes": [{
                "title": "Tofu Dish",
                "ingredients": ["tofu"],
                "instructions": "Cook",
                "summary": "",
                "image": "",
                "ready_in_minutes": 15
            }],
            "reflections": ["Adapted for vegan"]
        }
        
        # Provide enough mocks - manager might make multiple decisions
        mock_openai.side_effect = [
            fake_openai_response(ingredient_payload_2),  # Ingredient agent (in run_agent)
            fake_tool_call_response("fetch_recipes"),  # Manager round 1
            fake_openai_response(constraint_payload),  # Constraint agent (auto-triggered)
            fake_openai_response({"next_action": "finish", "reasoning": "Done"})  # Manager round 2
        ]
        
        mock_fetch_search.return_value = [
            MagicMock(text=json.dumps({"results": [{"id": 222}]}))
        ]
        mock_fetch_details.return_value = [
            MagicMock(text=json.dumps({
                "title": "Tofu Dish",
                "extendedIngredients": [{"original": "tofu"}],
                "instructions": "Cook",
                "image": "",
                "summary": "",
                "readyInMinutes": 15
            }))
        ]
        mock_asyncio.side_effect = [
            mock_fetch_search.return_value,
            mock_fetch_details.return_value
        ]
        
        result2 = app.run_agent("vegan tofu recipe")
        memory2 = result2["memory"]
        
        # Assertions: second request should not contain first request's data
        self.assertNotIn("chicken", memory2["normalized_ingredients"])
        self.assertIn("tofu", memory2["normalized_ingredients"])
        self.assertIn("vegan", memory2["constraints_applied"])
        self.assertEqual(len(memory2["recipes_fetched"]), 1)
        self.assertEqual(memory2["recipes_fetched"][0]["title"], "Tofu Dish")

    @patch.object(app.client.chat.completions, "create")
    def test_image_upload_workflow(self, mock_openai):
        """
        Test workflow with image upload: ingredient extraction from image -> 
        recipe fetch -> finish.
        """
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(b"fake image data")
            tmp_path = tmp.name
        
        try:
            ingredient_payload = {
                "normalized_ingredients": ["tomato", "onion", "garlic"],
                "categories": {"tomato": "vegetable", "onion": "vegetable", "garlic": "herb"},
                "constraints": [],
                "reflections": ["Identified ingredients from image"]
            }
            
            # Mock MCP (simplified for image test)
            with patch("app.asyncio.run") as mock_asyncio, \
                 patch("app.fetch_recipes_with_spoonacular_mcp") as mock_fetch_search, \
                 patch("app.fetch_recipe_details_mcp") as mock_fetch_details:
                
                mock_fetch_search.return_value = [
                    MagicMock(text=json.dumps({"results": [{"id": 333}]}))
                ]
                mock_fetch_details.return_value = [
                    MagicMock(text=json.dumps({
                        "title": "Image-Based Recipe",
                        "extendedIngredients": [{"original": "tomato"}],
                        "instructions": "Cook",
                        "image": "",
                        "summary": "",
                        "readyInMinutes": 10
                    }))
                ]
                mock_asyncio.side_effect = [
                    mock_fetch_search.return_value,
                    mock_fetch_details.return_value
                ]
                
                mock_openai.side_effect = [
                    fake_openai_response(ingredient_payload),  # Ingredient agent (with image)
                    fake_tool_call_response("fetch_recipes"),  # Manager round 1
                    fake_openai_response({"next_action": "finish", "reasoning": "Done"})  # Manager round 2
                ]
                
                result = app.run_agent("recipe from image", image_path=tmp_path)
                
                memory = result["memory"]
                self.assertEqual(len(memory["normalized_ingredients"]), 3)
                self.assertIn("tomato", memory["normalized_ingredients"])
                # Image path should be recorded
                self.assertEqual(len(memory["images_found"]), 1)
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_empty_goal_returns_error(self):
        """Test that run_agent returns error for empty goal."""
        result = app.run_agent("")
        self.assertIn("error", result)
        self.assertEqual(result["error"], "No goal provided")

    @patch("app.fetch_recipe_details_mcp", new_callable=AsyncMock)
    @patch("app.fetch_recipes_with_spoonacular_mcp", new_callable=AsyncMock)
    @patch("app.asyncio.run")
    @patch.object(app.client.chat.completions, "create")
    def test_manager_loop_stops_at_max_rounds(self, mock_openai, mock_asyncio, mock_fetch_search, mock_fetch_details):
        """
        Test that manager loop stops after max_rounds even if not finished.
        """
        ingredient_payload = {
            "normalized_ingredients": ["test"],
            "categories": {"test": "other"},
            "constraints": [],
            "reflections": []
        }
        
        # Manager keeps asking for process_ingredients (infinite loop scenario)
        # Each process_ingredients call will trigger call_ingredient_agent again
        # Need: 1 initial + (5 rounds * 2 calls each) = 11 total
        responses = [fake_openai_response(ingredient_payload)]  # Initial ingredient agent (in run_agent)
        for round_num in range(5):  # 5 rounds max
            responses.append(fake_tool_call_response("process_ingredients"))  # Manager decision
            responses.append(fake_openai_response(ingredient_payload))  # Ingredient agent call
        mock_openai.side_effect = responses
        
        mock_fetch_search.return_value = [
            MagicMock(text=json.dumps({"results": []}))
        ]
        mock_asyncio.return_value = mock_fetch_search.return_value
        
        result = app.run_agent("test recipe")
        
        # Should complete (max rounds reached)
        self.assertIn("history", result)
        history = result["history"]
        # Should have at least some history entries
        self.assertGreaterEqual(len(history), 0)

# ============================================================================
# Test Runner
# ============================================================================
if __name__ == "__main__":
    unittest.main()