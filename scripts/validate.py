"""
FinAgent Code Validator - Checks for common cross-platform and Pydantic v2 issues
Run this before committing code to catch errors early.
"""

import re
from pathlib import Path
from typing import List, Tuple
import sys

class CodeValidator:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_file(self, file_path: Path) -> None:
        """Validate a single Python file for common issues"""
        if not file_path.suffix == ".py":
            return
        
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            self.warnings.append(f"Could not read {file_path}: {e}")
            return
        
        # Check for && in subprocess calls
        self._check_command_chaining(file_path, content)
        
        # Check for old Pydantic Config style
        self._check_pydantic_config(file_path, content)
        
        # Check for hardcoded path separators
        self._check_path_separators(file_path, content)
        
        # Check for proper Settings imports
        self._check_settings_imports(file_path, content)
    
    def _check_command_chaining(self, file_path: Path, content: str) -> None:
        """Check for && operator in subprocess calls"""
        # Pattern: subprocess.run with && in the command
        pattern = r'subprocess\.run\(["\'].*&&.*["\']'
        matches = re.finditer(pattern, content)
        
        for match in matches:
            line_num = content[:match.start()].count('\n') + 1
            self.errors.append(
                f"❌ {file_path}:{line_num} - Using '&&' in subprocess.run() "
                f"(Windows incompatible). Use list args + cwd instead."
            )
    
    def _check_pydantic_config(self, file_path: Path, content: str) -> None:
        """Check for old-style Pydantic Config class"""
        # Check if file has BaseSettings
        if "BaseSettings" not in content:
            return
        
        # Pattern: class Config with env_file (old style)
        pattern = r'class\s+Config:\s*\n\s*env_file\s*='
        matches = re.finditer(pattern, content)
        
        for match in matches:
            line_num = content[:match.start()].count('\n') + 1
            self.errors.append(
                f"❌ {file_path}:{line_num} - Using old Pydantic v1 'class Config' style. "
                f"Use 'model_config = SettingsConfigDict(...)' instead."
            )
        
        # Check if using SettingsConfigDict properly
        if "SettingsConfigDict" in content:
            # Good! But check if env_file uses Path
            if 'env_file="' in content or "env_file='" in content:
                if "Path(__file__)" not in content:
                    line_num = content.find("env_file=") 
                    line_num = content[:line_num].count('\n') + 1
                    self.warnings.append(
                        f"⚠️  {file_path}:{line_num} - env_file should use Path(__file__).parent.parent "
                        f"for explicit path resolution."
                    )
    
    def _check_path_separators(self, file_path: Path, content: str) -> None:
        """Check for hardcoded path separators"""
        # Pattern: hardcoded paths with / or \\ (excluding URLs)
        patterns = [
            (r'["\'](?!http)[^"\']*[\\/][^"\']*[\\/][^"\']*["\']', "hardcoded path separators"),
            (r'os\.path\.join', "os.path.join (use pathlib.Path instead)")
        ]
        
        for pattern, description in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                # Skip if it's in a comment or docstring
                line_start = content.rfind('\n', 0, match.start()) + 1
                line = content[line_start:content.find('\n', match.start())]
                if line.strip().startswith('#'):
                    continue
                
                line_num = content[:match.start()].count('\n') + 1
                self.warnings.append(
                    f"⚠️  {file_path}:{line_num} - Found {description}. "
                    f"Consider using pathlib.Path for cross-platform compatibility."
                )
    
    def _check_settings_imports(self, file_path: Path, content: str) -> None:
        """Check for proper pydantic-settings imports"""
        if "BaseSettings" not in content:
            return
        
        # Check if importing from wrong module
        if "from pydantic import BaseSettings" in content:
            line_num = content.find("from pydantic import BaseSettings")
            line_num = content[:line_num].count('\n') + 1
            self.errors.append(
                f"❌ {file_path}:{line_num} - Importing BaseSettings from 'pydantic'. "
                f"Should be 'from pydantic_settings import BaseSettings'."
            )
    
    def validate_project(self) -> bool:
        """Validate all Python files in the project"""
        print("🔍 FinAgent Code Validator")
        print("=" * 60)
        
        # Find all Python files
        python_files = list(self.project_root.rglob("*.py"))
        print(f"Checking {len(python_files)} Python files...\n")
        
        for file_path in python_files:
            # Skip virtual environments and build directories
            if any(part in file_path.parts for part in ['.venv', 'venv', '__pycache__', 'build', 'dist']):
                continue
            
            self.validate_file(file_path)
        
        # Print results
        if self.errors:
            print(f"\n❌ ERRORS FOUND ({len(self.errors)}):")
            print("-" * 60)
            for error in self.errors:
                print(error)
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            print("-" * 60)
            for warning in self.warnings:
                print(warning)
        
        if not self.errors and not self.warnings:
            print("✅ No issues found! Code follows FinAgent standards.")
            return True
        
        if self.errors:
            print("\n" + "=" * 60)
            print("❌ Validation FAILED - Please fix errors before committing")
            return False
        
        print("\n" + "=" * 60)
        print("⚠️  Validation PASSED with warnings - Review recommended")
        return True

def main():
    """Run validator from command line"""
    # Get project root (assuming script is in scripts/ directory)
    project_root = Path(__file__).parent.parent
    
    validator = CodeValidator(project_root)
    success = validator.validate_project()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()