import ast
from pathlib import Path
import sys
import tomllib

REPOSITORY_HOME = Path(__file__).resolve().parents[1]


def package_configuration():
    return tomllib.loads((REPOSITORY_HOME / "pyproject.toml").read_text())


def test_pytest_confined_to_tests_dir():
    assert package_configuration()["tool"]["pytest"]["ini_options"]["testpaths"] == [
        "tests"
    ]


def test_hardware_markers_declared():
    declared = package_configuration()["tool"]["pytest"]["ini_options"]["markers"]
    assert {"hardware", "integration", "gpu"} <= {
        item.split(":")[0].strip() for item in declared
    }


def test_wheel_declares_module_level_runtime_deps():
    required = " ".join(package_configuration()["project"]["dependencies"])
    distributions = {"numpy": "numpy", "torch": "torch", "cryptography": "cryptography"}
    observed = set()
    for path in (REPOSITORY_HOME / "wanferenz" / "protocol").glob("*.py"):
        for item in ast.parse(path.read_text()).body:
            if isinstance(item, ast.Import):
                observed.update(alias.name.split(".")[0] for alias in item.names)
            elif isinstance(item, ast.ImportFrom) and item.level == 0 and item.module:
                observed.add(item.module.split(".")[0])
    dependencies = observed - sys.stdlib_module_names - {"wanferenz"}
    assert dependencies
    for name in dependencies:
        assert name in distributions
        assert distributions[name] in required


def test_distribution_includes_model_assets():
    config = package_configuration()["tool"]["setuptools"]
    assert config["packages"]["find"]["include"] == ["wanferenz*"]
    assert "data/*.json" in config["package-data"]["wanferenz.model"]
