from setuptools import setup, find_packages

setup(
    name="quant-finance-pricing-lab",
    version="0.1.0",
    description="A quantitative finance pricing lab for option pricing, Greeks, numerical methods, and dashboards.",
    author="Xiaodong Yang",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "plotly",
        "streamlit",
    ],
)