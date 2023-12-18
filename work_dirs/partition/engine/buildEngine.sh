name=$1

trtexec --onnx=../onnx/$name.onnx \
        --saveEngine=$name.trt \
        --memPoolSize=workspace:4096 \
        --fp16 \
        --explicitBatch \
        --verbose \
        --dumpLayerInfo \
        --dumpProfile \
        --separateProfileRun \
        --profilingVerbosity=detailed \
        --exportLayerInfo=layers/$name.json \
        > logs/$name.log 2>&1

if [ $? != 0 ]; then
        echo 😥 Failed to build model $name.trt.
        echo You can check the error message by logs/$name.log 
        exit 1
fi
