import tensorrt as trt
    
def run_trt_engine(context, engine, tensors):
    bindings = [None]*engine.num_bindings
    for idx, binding in enumerate(engine):
        tensor_name = engine.get_tensor_name(idx)
        if engine.get_tensor_mode(binding)==trt.TensorIOMode.INPUT:
            bindings[idx] = tensors['inputs'][tensor_name].data_ptr()
            if context.get_tensor_shape(tensor_name):
                context.set_input_shape(tensor_name, tensors['inputs'][tensor_name].shape)
        else:
            bindings[idx] = tensors['outputs'][tensor_name].data_ptr()
    context.execute_v2(bindings=bindings)
    
def load_engine(engine_filepath):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_filepath, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine