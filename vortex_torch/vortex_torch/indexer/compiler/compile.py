from .graph import contruct_graph
from .interface import generate_interface
from ..context import Context
import os
import importlib.util
import sys

def compile(ctx: Context, cache_pool=None) -> None:
    print("------------------------\n*** Using vortex attention ***\n------------------------\n", flush = True) 

    full_graph, sub_graphs = contruct_graph(ctx)
    file_path, cls_name = generate_interface(full_graph, sub_graphs, ctx=ctx)
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    cls = getattr(module, cls_name)
    return cls
    
