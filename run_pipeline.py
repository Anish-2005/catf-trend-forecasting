"""Convenience entry point:  python run_pipeline.py [--input data.csv] [--seed 7]"""
import runpy

runpy.run_module("catf", run_name="__main__")
