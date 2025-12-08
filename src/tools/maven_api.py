"""
LangChain tools for Maven operations in migration agents
"""
import xml.etree.ElementTree as ET
from pathlib import Path
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List
import requests
import json
import re
import subprocess
import json
import os
from pathlib import Path

from src.utils.logging_config import log_summary
from dotenv import load_dotenv

load_dotenv()

@tool
def find_all_poms(project_path: str) -> str:
    """Find all pom.xml files in the project (including submodules). Use this first to discover multi-module projects."""
    try:
        project_dir = Path(project_path)
        if not project_dir.exists():
            log_summary(f"MAVEN FIND POMS ERROR: Project path not found {project_path}")
            return f"Error: Project path not found: {project_path}"

        pom_files = sorted(project_dir.rglob("pom.xml"))

        if not pom_files:
            log_summary(f"MAVEN FIND POMS: No pom.xml files found in {project_path}")
            return "No pom.xml files found in project"

        result = f"Found {len(pom_files)} pom.xml file(s):\n"
        for pom in pom_files:
            relative_path = pom.relative_to(project_dir)
            # Check if it's a parent POM
            try:
                content = pom.read_text(encoding='utf-8')
                is_parent = "<modules>" in content or "<packaging>pom</packaging>" in content
                pom_type = " [PARENT POM]" if is_parent else ""
            except:
                pom_type = ""
            result += f"  - {relative_path}{pom_type}\n"

        log_summary(f"MAVEN FIND POMS SUCCESS: Found {len(pom_files)} POMs in {project_path}")
        return result
    except Exception as e:
        log_summary(f"MAVEN FIND POMS ERROR: {str(e)}")
        return f"Error finding pom files: {str(e)}"

@tool
def read_pom(project_path: str) -> str:
    """Read and parse pom.xml to extract basic project information. For multi-module projects, this reads the root pom.xml."""
    pom_path = Path(project_path) / "pom.xml"
    try:
        if not pom_path.exists():
            log_summary(f"MAVEN READ POM ERROR: pom.xml not found at {pom_path}")
            return "Error: pom.xml not found"

        tree = ET.parse(pom_path)
        root = tree.getroot()

        # Handle namespace
        namespace = ""
        if root.tag.startswith("{"):
            namespace = root.tag.split("}")[0] + "}"

        # Extract basic info
        groupId = _get_text(root, f"{namespace}groupId")
        artifactId = _get_text(root, f"{namespace}artifactId")
        version = _get_text(root, f"{namespace}version")

        # Extract Java version
        java_version = "Unknown"
        props = root.find(f"{namespace}properties")
        if props is not None:
            java_version = (_get_text(props, f"{namespace}java.version") or
                          _get_text(props, f"{namespace}maven.compiler.source") or
                          _get_text(props, f"{namespace}maven.compiler.target") or
                          "8")

        # Count dependencies
        deps = root.find(f"{namespace}dependencies")
        dep_count = len(deps.findall(f"{namespace}dependency")) if deps is not None else 0

        result = f"""Project Info:
- GroupId: {groupId}
- ArtifactId: {artifactId}
- Version: {version}
- Java Version: {java_version}
- Dependencies: {dep_count}"""

        log_summary(f"MAVEN READ POM SUCCESS: {artifactId} (Java {java_version}, {dep_count} dependencies)")
        return result

    except Exception as e:
        log_summary(f"MAVEN READ POM ERROR: {str(e)} for {project_path}")
        return f"Error reading pom.xml: {str(e)}"

@tool
def read_pom_by_path(pom_file_path: str) -> str:
    """Read and parse a specific pom.xml file by its path (relative to project root or absolute)."""
    try:
        pom_path = Path(pom_file_path)
        if not pom_path.exists():
            log_summary(f"MAVEN READ POM BY PATH ERROR: pom.xml not found at {pom_path}")
            return f"Error: pom.xml not found at {pom_file_path}"

        tree = ET.parse(pom_path)
        root = tree.getroot()

        # Handle namespace
        namespace = ""
        if root.tag.startswith("{"):
            namespace = root.tag.split("}")[0] + "}"

        # Extract basic info
        groupId = _get_text(root, f"{namespace}groupId")
        artifactId = _get_text(root, f"{namespace}artifactId")
        version = _get_text(root, f"{namespace}version")
        packaging = _get_text(root, f"{namespace}packaging", "jar")

        # Extract Java version
        java_version = "Unknown"
        props = root.find(f"{namespace}properties")
        if props is not None:
            java_version = (_get_text(props, f"{namespace}java.version") or
                          _get_text(props, f"{namespace}maven.compiler.source") or
                          _get_text(props, f"{namespace}maven.compiler.target") or
                          "8")

        # Count dependencies
        deps = root.find(f"{namespace}dependencies")
        dep_count = len(deps.findall(f"{namespace}dependency")) if deps is not None else 0

        # Check for modules
        modules = root.find(f"{namespace}modules")
        module_list = []
        if modules is not None:
            for module in modules.findall(f"{namespace}module"):
                if module.text:
                    module_list.append(module.text)

        result = f"""POM: {pom_file_path}
- GroupId: {groupId}
- ArtifactId: {artifactId}
- Version: {version}
- Packaging: {packaging}
- Java Version: {java_version}
- Dependencies: {dep_count}"""

        if module_list:
            result += f"\n- Modules: {', '.join(module_list)}"

        log_summary(f"MAVEN READ POM BY PATH SUCCESS: {artifactId} at {pom_file_path}")
        return result

    except Exception as e:
        log_summary(f"MAVEN READ POM BY PATH ERROR: {str(e)} for {pom_file_path}")
        return f"Error reading pom.xml: {str(e)}"

@tool
def get_java_version(project_path: str) -> str:
    """Get the current Java version from root pom.xml."""
    pom_path = Path(project_path) / "pom.xml"
    try:
        tree = ET.parse(pom_path)
        root = tree.getroot()

        namespace = ""
        if root.tag.startswith("{"):
            namespace = root.tag.split("}")[0] + "}"

        props = root.find(f"{namespace}properties")
        if props is not None:
            java_version = (_get_text(props, f"{namespace}java.version") or
                          _get_text(props, f"{namespace}maven.compiler.source") or
                          _get_text(props, f"{namespace}maven.compiler.target"))
            if java_version:
                version = java_version if java_version != "1.8" else "8"
                log_summary(f"MAVEN GET JAVA VERSION SUCCESS: Found version {version} in {pom_path}")
                return version

        log_summary(f"MAVEN GET JAVA VERSION DEFAULT: Using default version 8 for {pom_path}")
        return "8"  # Default
    except Exception as e:
        log_summary(f"MAVEN GET JAVA VERSION ERROR: {str(e)} for {pom_path}")
        return f"Error: {str(e)}"

@tool
def update_java_version_in_pom(pom_file_path: str, java_version: str) -> str:
    """Update Java version in a specific pom.xml file."""
    try:
        pom_path = Path(pom_file_path)
        if not pom_path.exists():
            log_summary(f"MAVEN UPDATE JAVA VERSION IN POM ERROR: File not found {pom_file_path}")
            return f"Error: pom.xml not found at {pom_file_path}"

        content = pom_path.read_text(encoding='utf-8')

        # Simple replacements for common Java version properties
        replacements = [
            (f'<java.version>1.8</java.version>', f'<java.version>{java_version}</java.version>'),
            (f'<maven.compiler.source>1.8</maven.compiler.source>', f'<maven.compiler.source>{java_version}</maven.compiler.source>'),
            (f'<maven.compiler.target>1.8</maven.compiler.target>', f'<maven.compiler.target>{java_version}</maven.compiler.target>'),
            (f'<java.version>8</java.version>', f'<java.version>{java_version}</java.version>'),
            (f'<maven.compiler.source>8</maven.compiler.source>', f'<maven.compiler.source>{java_version}</maven.compiler.source>'),
            (f'<maven.compiler.target>8</maven.compiler.target>', f'<maven.compiler.target>{java_version}</maven.compiler.target>'),
            (f'<java.version>11</java.version>', f'<java.version>{java_version}</java.version>'),
            (f'<maven.compiler.source>11</maven.compiler.source>', f'<maven.compiler.source>{java_version}</maven.compiler.source>'),
            (f'<maven.compiler.target>11</maven.compiler.target>', f'<maven.compiler.target>{java_version}</maven.compiler.target>'),
            (f'<java.version>17</java.version>', f'<java.version>{java_version}</java.version>'),
            (f'<maven.compiler.source>17</maven.compiler.source>', f'<maven.compiler.source>{java_version}</maven.compiler.source>'),
            (f'<maven.compiler.target>17</maven.compiler.target>', f'<maven.compiler.target>{java_version}</maven.compiler.target>'),
        ]

        changes_made = 0
        for old, new in replacements:
            if old in content:
                content = content.replace(old, new)
                changes_made += 1

        if changes_made > 0:
            pom_path.write_text(content, encoding='utf-8')
            log_summary(f"MAVEN UPDATE JAVA VERSION IN POM SUCCESS: Updated {pom_file_path} to {java_version} ({changes_made} changes)")
            return f"Updated Java version to {java_version} in {pom_file_path} ({changes_made} changes made)"
        else:
            log_summary(f"MAVEN UPDATE JAVA VERSION IN POM NO CHANGES: No version properties found in {pom_file_path}")
            return f"No Java version properties found to update in pom.xml"

    except Exception as e:
        log_summary(f"MAVEN UPDATE JAVA VERSION IN POM ERROR: {str(e)} for {pom_file_path}")
        return f"Error updating Java version: {str(e)}"

@tool
def update_all_poms_java_version(project_path: str, java_version: str) -> str:
    """Update Java version in ALL pom.xml files in the project (for multi-module projects)."""
    try:
        project_dir = Path(project_path)
        if not project_dir.exists():
            log_summary(f"MAVEN UPDATE ALL POMS ERROR: Project path not found {project_path}")
            return f"Error: Project path not found: {project_path}"

        pom_files = sorted(project_dir.rglob("pom.xml"))

        if not pom_files:
            log_summary(f"MAVEN UPDATE ALL POMS ERROR: No pom.xml files found in {project_path}")
            return "Error: No pom.xml files found in project"

        results = []
        total_changes = 0

        for pom in pom_files:
            relative_path = pom.relative_to(project_dir)
            result = update_java_version_in_pom(str(pom), java_version)

            # Count changes
            if "changes made)" in result:
                import re
                match = re.search(r'\((\d+) changes made\)', result)
                if match:
                    total_changes += int(match.group(1))

            results.append(f"  {relative_path}: {result}")

        summary = f"Updated Java version to {java_version} in {len(pom_files)} pom.xml file(s) ({total_changes} total changes):\n"
        log_summary(f"MAVEN UPDATE ALL POMS SUCCESS: Updated {len(pom_files)} POMs to Java {java_version} in {project_path}")
        return summary + "\n".join(results)

    except Exception as e:
        log_summary(f"MAVEN UPDATE ALL POMS ERROR: {str(e)}")
        return f"Error updating pom files: {str(e)}"

@tool
def update_java_version(project_path: str, java_version: str) -> str:
    """Update Java version in root pom.xml properties. For multi-module projects, use update_all_poms_java_version instead."""
    pom_path = Path(project_path) / "pom.xml"
    try:
        content = pom_path.read_text(encoding='utf-8')

        # Simple replacements for common Java version properties
        replacements = [
            (f'<java.version>1.8</java.version>', f'<java.version>{java_version}</java.version>'),
            (f'<maven.compiler.source>1.8</maven.compiler.source>', f'<maven.compiler.source>{java_version}</maven.compiler.source>'),
            (f'<maven.compiler.target>1.8</maven.compiler.target>', f'<maven.compiler.target>{java_version}</maven.compiler.target>'),
            (f'<java.version>8</java.version>', f'<java.version>{java_version}</java.version>'),
            (f'<maven.compiler.source>8</maven.compiler.source>', f'<maven.compiler.source>{java_version}</maven.compiler.source>'),
            (f'<maven.compiler.target>8</maven.compiler.target>', f'<maven.compiler.target>{java_version}</maven.compiler.target>'),
        ]

        changes_made = 0
        for old, new in replacements:
            if old in content:
                content = content.replace(old, new)
                changes_made += 1

        if changes_made > 0:
            pom_path.write_text(content, encoding='utf-8')
            log_summary(f"MAVEN UPDATE JAVA VERSION SUCCESS: Updated to {java_version} ({changes_made} changes) in {pom_path}")
            return f"Updated Java version to {java_version} in pom.xml ({changes_made} changes made)"
        else:
            log_summary(f"MAVEN UPDATE JAVA VERSION NO CHANGES: No Java version properties found in {pom_path}")
            return f"No Java version properties found to update in pom.xml"

    except Exception as e:
        log_summary(f"MAVEN UPDATE JAVA VERSION ERROR: {str(e)} for version {java_version} in {pom_path}")
        return f"Error updating Java version: {str(e)}"

@tool
def list_dependencies(project_path: str) -> str:
    """List all dependencies from pom.xml."""
    pom_path = Path(project_path) / "pom.xml"
    try:
        tree = ET.parse(pom_path)
        root = tree.getroot()

        namespace = ""
        if root.tag.startswith("{"):
            namespace = root.tag.split("}")[0] + "}"

        deps = root.find(f"{namespace}dependencies")
        if deps is None:
            log_summary(f"MAVEN LIST DEPENDENCIES NO DEPS: No dependencies found in {pom_path}")
            return "No dependencies found"

        dependencies = []
        for dep in deps.findall(f"{namespace}dependency"):
            groupId = _get_text(dep, f"{namespace}groupId")
            artifactId = _get_text(dep, f"{namespace}artifactId")
            version = _get_text(dep, f"{namespace}version", "managed")
            scope = _get_text(dep, f"{namespace}scope", "compile")
            dependencies.append(f"- {groupId}:{artifactId}:{version} ({scope})")

        log_summary(f"MAVEN LIST DEPENDENCIES SUCCESS: Found {len(dependencies)} dependencies in {pom_path}")
        return f"Dependencies ({len(dependencies)}):\n" + "\n".join(dependencies)

    except Exception as e:
        log_summary(f"MAVEN LIST DEPENDENCIES ERROR: {str(e)} for {pom_path}")
        return f"Error listing dependencies: {str(e)}"

@tool
def add_openrewrite_plugin(project_path: str) -> str:
    """Add OpenRewrite Maven plugin with migrate-java dependency and sample config to pom.xml if not present."""
    try:
        pom_path = Path(project_path) / "pom.xml"
        content = pom_path.read_text(encoding='utf-8')

        if "rewrite-maven-plugin" in content:
            log_summary(f"MAVEN ADD OPENREWRITE PLUGIN ALREADY EXISTS: Plugin already present in {pom_path}")
            return "OpenRewrite plugin already present in pom.xml"

        # Version 5.42.0 is stable and has all modern migration recipes
        plugin_xml = '''        <plugin>
                <groupId>org.openrewrite.maven</groupId>
                <artifactId>rewrite-maven-plugin</artifactId>
                <version>5.42.0</version>
                <configuration>
                    <exportDatatables>true</exportDatatables>
                    <activeRecipes>
                        <recipe>org.openrewrite.java.migrate.UpgradeToJava21</recipe>
                    </activeRecipes>
                </configuration>
                <dependencies>
                    <dependency>
                        <groupId>org.openrewrite.recipe</groupId>
                        <artifactId>rewrite-migrate-java</artifactId>
                        <version>2.26.0</version>
                    </dependency>
                    <dependency>
                        <groupId>org.openrewrite.recipe</groupId>
                        <artifactId>rewrite-spring</artifactId>
                        <version>5.21.0</version>
                    </dependency>
                    <dependency>
                        <groupId>org.openrewrite.recipe</groupId>
                        <artifactId>rewrite-testing-frameworks</artifactId>
                        <version>2.19.0</version>
                    </dependency>
                </dependencies>
            </plugin>'''

        # Insert plugin before closing </plugins> tag
        if "</plugins>" in content:
            content = content.replace("</plugins>", f"{plugin_xml}\n            </plugins>")
        elif "<build>" in content and "</build>" in content:
            # Add plugins section to build
            plugins_section = f'''        <plugins>\n{plugin_xml}\n            </plugins>'''
            content = content.replace("</build>", f"{plugins_section}\n        </build>")
        else:
            # Add entire build section
            build_section = f'''    <build>\n        <plugins>\n{plugin_xml}\n            </plugins>\n        </build>'''
            content = content.replace("</project>", f"{build_section}\n</project>")

        pom_path.write_text(content, encoding='utf-8')
        log_summary(f"MAVEN ADD OPENREWRITE PLUGIN SUCCESS: Added plugin to {pom_path}")
        return "Successfully added OpenRewrite plugin to pom.xml"

    except Exception as e:
        log_summary(f"MAVEN ADD OPENREWRITE PLUGIN ERROR: {str(e)} for {pom_path}")
        return f"Error adding OpenRewrite plugin: {str(e)}"

class ConfigureOpenRewriteRecipesInput(BaseModel):
    """Input for configure_openrewrite_recipes tool."""
    project_path: str = Field(description="Path to the project directory containing pom.xml")
    recipes: List[str] = Field(description="List of OpenRewrite recipe names to configure")

@tool(args_schema=ConfigureOpenRewriteRecipesInput)
def configure_openrewrite_recipes(project_path: str, recipes: List[str]) -> str:
    """Configure OpenRewrite plugin with active recipes and dependencies in pom.xml."""
    try:
        pom_path = Path(project_path) / "pom.xml"
        content = pom_path.read_text(encoding='utf-8')

        if "rewrite-maven-plugin" not in content:
            log_summary(f"MAVEN CONFIGURE OPENREWRITE RECIPES ERROR: Plugin not found in {pom_path}")
            return "Error: OpenRewrite plugin not found. Add plugin first using add_openrewrite_plugin."

        # Build active recipes configuration
        active_recipes = ""
        for recipe in recipes:
            active_recipes += f"                <recipe>{recipe}</recipe>\n"

        # Complete plugin configuration with recipes and dependencies
        # Version 5.42.0+ required for modern migration recipes
        new_plugin_config = f'''        <plugin>
                <groupId>org.openrewrite.maven</groupId>
                <artifactId>rewrite-maven-plugin</artifactId>
                <version>5.42.0</version>
                <configuration>
                    <activeRecipes>
{active_recipes.rstrip()}
                    </activeRecipes>
                </configuration>
                <dependencies>
                    <dependency>
                        <groupId>org.openrewrite.recipe</groupId>
                        <artifactId>rewrite-migrate-java</artifactId>
                        <version>2.26.0</version>
                    </dependency>
                    <dependency>
                        <groupId>org.openrewrite.recipe</groupId>
                        <artifactId>rewrite-spring</artifactId>
                        <version>5.21.0</version>
                    </dependency>
                    <dependency>
                        <groupId>org.openrewrite.recipe</groupId>
                        <artifactId>rewrite-testing-frameworks</artifactId>
                        <version>2.19.0</version>
                    </dependency>
                </dependencies>
            </plugin>'''

        # Replace the existing plugin configuration
        import re
        plugin_pattern = r'<plugin>\s*<groupId>org\.openrewrite\.maven</groupId>.*?</plugin>'
        if re.search(plugin_pattern, content, re.DOTALL):
            content = re.sub(plugin_pattern, new_plugin_config, content, flags=re.DOTALL)
        else:
            log_summary(f"MAVEN CONFIGURE OPENREWRITE RECIPES ERROR: Could not find plugin to replace in {pom_path}")
            return "Error: Could not find OpenRewrite plugin to replace"

        pom_path.write_text(content, encoding='utf-8')
        log_summary(f"MAVEN CONFIGURE OPENREWRITE RECIPES SUCCESS: Configured {len(recipes)} recipes in {pom_path}")
        return f"Successfully configured OpenRewrite plugin with {len(recipes)} active recipes"

    except Exception as e:
        log_summary(f"MAVEN CONFIGURE OPENREWRITE RECIPES ERROR: {str(e)} for {pom_path}")
        return f"Error configuring OpenRewrite recipes: {str(e)}"

@tool
def add_rewrite_dependency(project_path: str, dependency_artifact: str, version: str = "2.0.7") -> str:
    """Add a specific OpenRewrite recipe dependency to the plugin."""
    try:
        pom_path = Path(project_path) / "pom.xml"
        content = pom_path.read_text(encoding='utf-8')

        if "rewrite-maven-plugin" not in content:
            log_summary(f"MAVEN ADD REWRITE DEPENDENCY ERROR: Plugin not found in {pom_path}")
            return "Error: OpenRewrite plugin not found in pom.xml"

        new_dependency = f'''            <dependency>
                <groupId>org.openrewrite.recipe</groupId>
                <artifactId>{dependency_artifact}</artifactId>
                <version>{version}</version>
            </dependency>'''

        # Check if dependencies section exists in the plugin
        if "<dependencies>" in content and "rewrite-maven-plugin" in content:
            # Add to existing dependencies
            content = content.replace("</dependencies>", f"{new_dependency}\n            </dependencies>")
        else:
            # Add dependencies section to plugin
            dependencies_section = f'''        <dependencies>
{new_dependency}
            </dependencies>'''
            # Insert before closing plugin tag
            content = content.replace("</plugin>", f"{dependencies_section}\n            </plugin>")

        pom_path.write_text(content, encoding='utf-8')
        log_summary(f"MAVEN ADD REWRITE DEPENDENCY SUCCESS: Added {dependency_artifact}:{version} to {pom_path}")
        return f"Successfully added dependency {dependency_artifact}:{version} to OpenRewrite plugin"

    except Exception as e:
        log_summary(f"MAVEN ADD REWRITE DEPENDENCY ERROR: {str(e)} for {dependency_artifact}:{version} in {pom_path}")
        return f"Error adding OpenRewrite dependency: {str(e)}"

@tool("update_dependencies", return_direct=True)
def update_dependencies(project_path: str) -> str:
    """
    Use this tool to get the latest versions of any dependency in the pom.xml
    Runs `mvn versions:display-dependency-updates` in the given project_path,
    parses out the lines showing newer versions, then calls an LLM to convert
    them into a JSON mapping dependency -> latest version. Returns the JSON string.
    """
    project_dir = Path(project_path)
    if not project_dir.exists():
        log_summary(f"MAVEN UPDATE DEPENDENCIES ERROR: Path not found {project_path}")
        return json.dumps({"error": f"Path not found: {project_path}"})

    # 1) run the mvn command
    proc = subprocess.run(
        ["mvn", "versions:display-dependency-updates"],
        cwd=project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    raw = proc.stdout + "\n" + proc.stderr

    # 2) extract only lines like "[INFO]   \s+group:artifact ... -> newVersion"
    matches = re.findall(
        r"\[INFO\]\s+([^\s]+)\s+\.+\s*->\s*([^\s]+)",
        raw,
    )

    if not matches:
        log_summary(f"MAVEN UPDATE DEPENDENCIES NO UPDATES: No updatable dependencies found in {project_path}")
        return json.dumps({"warning": "No updatable dependencies found."})

    # Build JSON directly from the parsed matches (no LLM needed)
    updates_dict = {dep: new for dep, new in matches}
    log_summary(f"MAVEN UPDATE DEPENDENCIES FOUND: {len(matches)} updatable dependencies in {project_path}")

    log_summary(f"MAVEN UPDATE DEPENDENCIES SUCCESS: Retrieved updates for {project_path}")
    return json.dumps(updates_dict, indent=2)

@tool
def get_latest_version_from_maven_central(group_id: str, artifact_id: str) -> str:
    """Query Maven Central to find the latest version of a dependency or plugin."""
    try:
        # Maven Central search API endpoint
        url = f"https://search.maven.org/solrsearch/select"
        params = {
            "q": f"g:{group_id} AND a:{artifact_id}",
            "core": "gav",
            "rows": 1,
            "wt": "json"
        }

        response = requests.get(url, params=params, timeout=10, verify=False)
        response.raise_for_status()

        data = response.json()

        if data.get("response", {}).get("numFound", 0) == 0:
            log_summary(f"MAVEN CENTRAL QUERY NOT FOUND: {group_id}:{artifact_id}")
            return f"No artifact found for {group_id}:{artifact_id}"

        docs = data.get("response", {}).get("docs", [])
        if not docs:
            log_summary(f"MAVEN CENTRAL QUERY NO VERSION: {group_id}:{artifact_id}")
            return f"No version information found for {group_id}:{artifact_id}"

        latest_version = docs[0].get("v", "unknown")
        timestamp = docs[0].get("timestamp", 0)

        # Convert timestamp to readable date
        import datetime
        date_str = datetime.datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d") if timestamp else "unknown"

        log_summary(f"MAVEN CENTRAL QUERY SUCCESS: {group_id}:{artifact_id} latest version {latest_version}")
        return f"Latest version of {group_id}:{artifact_id} is {latest_version} (published: {date_str})"

    except requests.RequestException as e:
        log_summary(f"MAVEN CENTRAL QUERY ERROR: Request failed for {group_id}:{artifact_id} - {str(e)}")
        return f"Error querying Maven Central: {str(e)}"
    except Exception as e:
        log_summary(f"MAVEN CENTRAL QUERY ERROR: Processing failed for {group_id}:{artifact_id} - {str(e)}")
        return f"Error processing Maven Central response: {str(e)}"

@tool
def get_spring_boot_latest_version() -> str:
    """Get the latest Spring Boot 3.x version from Maven Central."""
    try:
        # Query for Spring Boot starter parent
        url = "https://search.maven.org/solrsearch/select"
        params = {
            "q": "g:org.springframework.boot AND a:spring-boot-starter-parent AND v:3.*",
            "core": "gav",
            "rows": 10,
            "wt": "json"
        }

        response = requests.get(url, params=params, timeout=10, verify=False)
        response.raise_for_status()

        data = response.json()
        docs = data.get("response", {}).get("docs", [])

        if not docs:
            log_summary(f"MAVEN SPRING BOOT VERSION QUERY NOT FOUND: No Spring Boot 3.x versions found")
            return "No Spring Boot 3.x versions found"

        # Get the latest version (first in results)
        latest = docs[0]
        version = latest.get("v", "unknown")
        timestamp = latest.get("timestamp", 0)

        import datetime
        date_str = datetime.datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d") if timestamp else "unknown"

        log_summary(f"MAVEN SPRING BOOT VERSION QUERY SUCCESS: Latest version {version}")
        return f"Latest Spring Boot 3.x version: {version} (published: {date_str})"

    except Exception as e:
        log_summary(f"MAVEN SPRING BOOT VERSION QUERY ERROR: {str(e)}")
        return f"Error getting Spring Boot version: {str(e)}"

@tool
def get_spring_framework_latest_version() -> str:
    """Get the latest Spring Framework 6.x version from Maven Central."""
    try:
        url = "https://search.maven.org/solrsearch/select"
        params = {
            "q": "g:org.springframework AND a:spring-core AND v:6.*",
            "core": "gav",
            "rows": 10,
            "wt": "json"
        }

        response = requests.get(url, params=params, timeout=10, verify=False)
        response.raise_for_status()

        data = response.json()
        docs = data.get("response", {}).get("docs", [])

        if not docs:
            log_summary(f"MAVEN SPRING FRAMEWORK VERSION QUERY NOT FOUND: No Spring Framework 6.x versions found")
            return "No Spring Framework 6.x versions found"

        latest = docs[0]
        version = latest.get("v", "unknown")
        timestamp = latest.get("timestamp", 0)

        import datetime
        date_str = datetime.datetime.fromtimestamp(timestamp / 1000).strftime("%Y-%m-%d") if timestamp else "unknown"

        log_summary(f"MAVEN SPRING FRAMEWORK VERSION QUERY SUCCESS: Latest version {version}")
        return f"Latest Spring Framework 6.x version: {version} (published: {date_str})"

    except Exception as e:
        log_summary(f"MAVEN SPRING FRAMEWORK VERSION QUERY ERROR: {str(e)}")
        return f"Error getting Spring Framework version: {str(e)}"

def _get_text(element, tag: str, default: str = "") -> str:
    """Get text from XML element safely."""
    child = element.find(tag)
    return child.text if child is not None and child.text else default

# Collect all Maven tools
maven_tools = [
    find_all_poms, read_pom, read_pom_by_path, get_java_version,
    update_java_version, update_java_version_in_pom, update_all_poms_java_version,
    list_dependencies, add_openrewrite_plugin, configure_openrewrite_recipes,
    add_rewrite_dependency, get_latest_version_from_maven_central,
    get_spring_boot_latest_version, get_spring_framework_latest_version, update_dependencies
]