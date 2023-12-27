
import tensorrt as trt

# for explicit batch
def run_trt_engine(context, engine, tensors):
    bindings = [None]*engine.num_bindings
    for name,tensor in tensors['inputs'].items():
        idx = engine.get_binding_index(name)
        bindings[idx] = tensor.data_ptr()

    for name,tensor in tensors['outputs'].items():
        idx = engine.get_binding_index(name)
        bindings[idx] = tensor.data_ptr()

    context.execute_v2(bindings=bindings)
    return

# for dynamic batch
# def run_trt_engine(context, engine, tensors):
#     bindings = [None]*engine.num_bindings
#     for name,tensor in tensors['inputs'].items():
#         idx = engine.get_binding_index(name)
#         bindings[idx] = tensor.data_ptr()
#         if engine.is_shape_binding(idx) and is_shape_dynamic(context.get_shape(idx)):
#             context.set_shape_input(idx, tensor)
#         elif is_shape_dynamic(engine.get_binding_shape(idx)):
#             context.set_binding_shape(idx, tensor.shape)

#     for name,tensor in tensors['outputs'].items():
#         idx = engine.get_binding_index(name)
#         bindings[idx] = tensor.data_ptr()

#     context.execute_v2(bindings=bindings)
#     return

def is_shape_dynamic(shape):
    return any([is_dimension_dynamic(dim) for dim in shape])


def is_dimension_dynamic(dim):
    return dim is None or dim <= 0


def load_engine(engine_filepath, trt_logger):
    with open(engine_filepath, "rb") as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def save_engine(serialized_engine, engine_filepath):
    with open(engine_filepath, "wb") as f:
        f.write(serialized_engine)
    return


def engine_info(engine_filepath):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    engine = load_engine(engine_filepath, TRT_LOGGER)

    binding_template = r"""
                        {btype} {{
                        name: "{bname}"
                        data_type: {dtype}
                        dims: {dims}
                        }}"""
    type_mapping = {"DataType.HALF": "TYPE_FP16",
                    "DataType.FLOAT": "TYPE_FP32",
                    "DataType.INT32": "TYPE_INT32",
                    "DataType.BOOL" : "TYPE_BOOL"}

    print("engine name", engine.name)
    print("has_implicit_batch_dimension", engine.has_implicit_batch_dimension)
    start_dim = 0 if engine.has_implicit_batch_dimension else 1
    print("num_optimization_profiles", engine.num_optimization_profiles)
    print("max_batch_size:", engine.max_batch_size)
    print("device_memory_size:", engine.device_memory_size)
    print("max_workspace_size:", engine.max_workspace_size)
    print("num_layers:", engine.num_layers)

    for i in range(engine.num_bindings):
        btype = "input" if engine.binding_is_input(i) else "output"
        bname = engine.get_binding_name(i)
        dtype = engine.get_binding_dtype(i)
        bdims = engine.get_binding_shape(i)
        config_values = {
            "btype": btype,
            "bname": bname,
            "dtype": type_mapping[str(dtype)],
            "dims": list(bdims[start_dim:])
        }
        final_binding_str = binding_template.format_map(config_values)
        print(final_binding_str)
    return


def build_engine_onnx(model_file, max_ws=2*1024*1024*1024, fp16=False, verbose=False):
    if verbose:
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    else:
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    # network = builder.create_network(common.EXPLICIT_BATCH)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_ws)
    # Load the Onnx model and parse it in order to populate the TensorRT network.
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if verbose:
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        
    with open(model_file, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    return builder.build_serialized_network(network, config)