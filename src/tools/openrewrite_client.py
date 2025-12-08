"""
LangChain tools for OpenRewrite operations in migration agents
"""
from pathlib import Path
import yaml
from langchain_core.tools import tool

@tool
def create_rewrite_config(project_path: str, recipes: str) -> str:
    """DEPRECATED: YAML configuration doesn't work. Use configure_openrewrite_recipes in maven_api instead."""
    return "WARNING: YAML configuration is deprecated and doesn't work reliably. Use configure_openrewrite_recipes from maven_api.py to configure recipes directly in pom.xml instead."

@tool
def get_available_recipes() -> str:
    """Get list of commonly used OpenRewrite recipes."""
    recipes = {
        "Java Version Migration": [
            "org.openrewrite.java.migrate.UpgradeToJava21"
        ],
        "Spring Boot Migration": [
            "org.openrewrite.java.spring.boot3.UpgradeSpringBoot_3_0",
            "org.openrewrite.java.spring.boot3.UpgradeSpringBoot_3_1",
            "org.openrewrite.java.spring.boot3.UpgradeSpringBoot_3_2"
        ],
        "Jakarta Migration": [
            "org.openrewrite.java.migrate.javax.MigrateJavaxToJakarta"
        ],
        "JUnit Migration": [
            "org.openrewrite.java.testing.junit5.JUnit4toJUnit5Migration"
        ],
        "Common Modernization": [
            "org.openrewrite.java.migrate.apache.commons.lang.ApacheCommonsLang3",
            "org.openrewrite.java.logging.slf4j.Slf4jBestPractices"
        ]
    }

    result = "OpenRewrite Recipes (Use for definitive list):\n\n"
    for category, recipe_list in recipes.items():
        result += f"{category}:\n"
        for recipe in recipe_list:
            result += f"  - {recipe}\n"
        result += "\n"

    prepend = "For a definitive list, call the openrewriteagent with a clear description of your current\n" \
              "repository configuration, including java, springboot and plugin versions (passed as strings not dict).\n" \
              "This function gives a list of commonly used recipes"

    result += "Recipe names may vary between OpenRewrite versions.\n"

    return prepend + result

# @tool
# def check_rewrite_config(project_path: str) -> str:
#     """Check if rewrite.yml configuration exists and show its contents."""
#     try:
#         config_path = Path(project_path) / "rewrite.yml"
#         if not config_path.exists():
#             return "No rewrite.yml configuration found"
#
#         with open(config_path, 'r') as f:
#             config = yaml.safe_load(f)
#
#         recipes = config.get("recipeList", [])
#         return f"Found rewrite.yml with {len(recipes)} recipes:\n" + "\n".join(f"- {recipe}" for recipe in recipes)
#
#     except Exception as e:
#         return f"Error reading rewrite config: {str(e)}"

@tool
def remove_rewrite_config(project_path: str) -> str:
    """Remove rewrite.yml configuration file."""
    try:
        config_path = Path(project_path) / "rewrite.yml"
        if config_path.exists():
            config_path.unlink()
            return "Removed rewrite.yml configuration file"
        else:
            return "No rewrite.yml file to remove"
    except Exception as e:
        return f"Error removing rewrite config: {str(e)}"

@tool
def suggest_recipes_for_java_version(current_version: str, target_version: str = "21") -> str:
    """Suggest OpenRewrite recipes based on current and target Java versions."""
    version_mapping = {
        "8": ["org.openrewrite.java.migrate.UpgradeToJava21"],
        "11": ["org.openrewrite.java.migrate.UpgradeToJava21"],
        "17": ["org.openrewrite.java.migrate.UpgradeToJava21"]
    }

    try:
        current = current_version.replace("1.8", "8")  # Handle 1.8 format
        recipes = []

        if current == "8" and target_version == "21":
            recipes.extend([
                "org.openrewrite.java.migrate.UpgradeToJava21",
            ])
        elif current == "11" and target_version == "21":
            recipes.extend([
                "org.openrewrite.java.migrate.UpgradeToJava21",
            ])
        elif current == "17" and target_version == "21":
            recipes.append("org.openrewrite.java.migrate.UpgradeToJava21")
        else:
            return f"No specific recipes for Java {current} to {target_version} migration"

        return f"Recommended recipes for Java {current} to {target_version}:\n" + "\n".join(f"- {recipe}" for recipe in recipes)

    except Exception as e:
        return f"Error suggesting recipes: {str(e)}"

# Collect all OpenRewrite tools
openrewrite_tools = [
    create_rewrite_config,
    get_available_recipes,
    # check_rewrite_config,
    remove_rewrite_config,
    suggest_recipes_for_java_version
]