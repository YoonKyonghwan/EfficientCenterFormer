# WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# configure the environment
# . tool/environment.sh

# if [ "$ConfigurationStatus" != "Success" ]; then
#     echo "Exit due to configure failure."
#     exit
# fi

# tensorrt version
# version=`trtexec | grep -m 1 TensorRT | sed -n "s/.*\[TensorRT v\([0-9]*\)\].*/\1/p"`

# resnet50/resnet50-int8/swint-tiny

# fp16/int8
# precision=$DEBUG_PRECISION

# precision flags
trtexec_fp16_flags="--fp16"
trtexec_dynamic_flags="--fp16"
if [ "$precision" == "int8" ]; then
    trtexec_dynamic_flags="--fp16 --int8"
fi

if [ "$DEBUG_DLA" == "0" ]; then
    trtexec_dynamic_flags_cam="${trtexec_dynamic_flags} --useDLACore=0"
    echo Using DLA...
else
    trtexec_dynamic_flags_cam=$trtexec_dynamic_flags
fi

function get_onnx_number_io(){
    model=$1

    if [ ! -f "$model" ]; then
        echo The model [$model] not exists.
        return
    fi

    number_of_input=`python -c "import onnx;m=onnx.load('$model');print(len(m.graph.input), end='')"`
    number_of_output=`python -c "import onnx;m=onnx.load('$model');print(len(m.graph.output), end='')"`
    # echo The model [$model] has $number_of_input inputs and $number_of_output outputs.
}

function compile_trt_model(){
    # $1: name
    # $2: precision_flags
    # $3: number_of_input
    # $4: number_of_output
    name=$1
    precision_flags=$2
    # number_of_input=$3
    # number_of_output=$4
    onnx_path=../onnx/$name.onnx
    engine_path=./$name.trt

    if [ -f $engine_path ]; then
        echo Model $name.trt already build 🙋🙋🙋.
        return
    fi
    
    get_onnx_number_io $onnx_path
    echo The model has $number_of_input inputs and $number_of_output outputs.

    input_flags="--inputIOFormats="
    output_flags="--outputIOFormats="
    for i in $(seq 1 $number_of_input); do
        input_flags+=fp16:chw,
    done

    for i in $(seq 1 $number_of_output); do
        output_flags+=fp16:chw,
    done

    cmd="--onnx=$onnx_path ${precision_flags} ${input_flags} ${output_flags} \
        --saveEngine=$engine_path \
        --memPoolSize=workspace:2048 --verbose --dumpLayerInfo \
        --dumpProfile --separateProfileRun \
        --profilingVerbosity=detailed --exportLayerInfo=layers/$name.json"

    # mkdir -p $result_save_directory
    echo Building the model: $name.trt, this will take several minutes. Wait a moment.
    trtexec $cmd > logs/$name.log 2>&1
    if [ $? != 0 ]; then
        echo 😥 Failed to build model $name.trt.
        echo You can check the error message by logs/$name.log 
        exit 1
    fi
}

# maybe int8 / fp16
# compile_trt_model "camera.backbone" "$trtexec_dynamic_flags_cam" 2 2 # old
# compile_trt_model "camera.backbone" "$trtexec_dynamic_flags_cam" 1 1
# compile_trt_model "camera.depth" "$trtexec_dynamic_flags_cam" 2 2
# compile_trt_model "fuser" "$trtexec_dynamic_flags" 2 1

# # fp16 only
compile_trt_model "poolformer_folded" "$trtexec_fp16_flags"
# compile_trt_model "backbone" "$trtexec_fp16_flags"
# compile_trt_model "camera.vtransform" "$trtexec_fp16_flags" 1 1
# compile_trt_model "head.bbox" "$trtexec_fp16_flags" 1 6