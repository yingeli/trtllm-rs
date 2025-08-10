use anyhow::anyhow;
use std::ops::Deref;
use std::path::Path;

use cxx::UniquePtr;

pub use self::ffi::{DataType, ModelType, Response, Result, VecTokens};

#[cxx::bridge]
mod ffi {
    #[namespace = "tensorrt_llm::executor"]
    #[derive(Copy, Clone, Debug)]
    #[repr(i32)]
    enum DataType {
        #[cxx_name = "kBOOL"]
        Bool,

        #[cxx_name = "kUINT8"]
        Uint8,

        #[cxx_name = "kINT8"]
        Int8,

        #[cxx_name = "kINT32"]
        Int32,

        #[cxx_name = "kINT64"]
        Int64,

        #[cxx_name = "kBF16"]
        Bf16,

        #[cxx_name = "kFP8"]
        Fp8,

        #[cxx_name = "kFP16"]
        Fp16,

        #[cxx_name = "kFP32"]
        Fp32,

        #[cxx_name = "kUNKNOWN"]
        Unknown,
    }

    #[namespace = "tensorrt_llm::executor"]
    #[derive(Copy, Clone, Debug)]
    #[repr(i32)]
    enum ModelType {
        #[cxx_name = "kDECODER_ONLY"]
        DecoderOnly,

        #[cxx_name = "kENCODER_ONLY"]
        EncoderOnly,

        #[cxx_name = "kENCODER_DECODER"]
        EncoderDecoder,
    }

    #[derive(PartialEq, Clone)]
    struct VecTokens {
        v: Vec<u32>,
    }

    struct Result {
        is_final: bool,

        output_token_ids: Vec<VecTokens>,
    }

    /*
    #[namespace = "tensorrt_llm::executor"]
    struct Result {
        #[cxx_name = "isFinal"]
        is_final: bool,

        #[cxx_name = "outputTokenIds"]
        output_token_ids: Vec<Vec<u32>>,
    }
    */

    #[namespace = "tensorrt_llm::executor"]
    extern "C++" {
        include!("tensorrt_llm/executor/types.h");

        type DataType;
    }

    #[namespace = "tensorrt_llm::executor"]
    extern "C++" {
        include!("tensorrt_llm/executor/tensor.h");

        type Shape;

        type Tensor;
    }

    #[namespace = "tensorrt_llm::executor"]
    unsafe extern "C++" {
        include!("tensorrt_llm/executor/executor.h");

        type ModelType;

        type LogitsPostProcessorConfig;

        #[cxx_name = "setProcessorMap"]
        fn set_processor_map(
            self: Pin<&mut LogitsPostProcessorConfig>, 
            processorMap: bool
        ) -> Result<()>;        

        type ExecutorConfig;

        type Request;

        #[cxx_name = "setStreaming"]
        fn set_streaming(self: Pin<&mut Request>, streaming: bool) -> Result<()>;

        #[cxx_name = "setEndId"]
        fn set_end_id(self: Pin<&mut Request>, end_id: i32) -> Result<()>;

        #[cxx_name = "setPadId"]
        fn set_pad_id(self: Pin<&mut Request>, pad_id: i32) -> Result<()>;

        //#[cxx_name = "setEncoderInputFeatures"]
        //fn set_encoder_input_features(self: Pin<&mut Request>, features: UniquePtr<Tensor>) -> Result<()>;

        type Response;

        type Executor;

        #[cxx_name = "enqueueRequest"]
        fn enqueue_request(self: Pin<&mut Executor>, request: &Request) -> Result<u64>;
    }

    unsafe extern "C++" {
        include!("trtllm/src/sys/executor.h");

        fn shape(dims: &[usize]) -> UniquePtr<Shape>;

        type TensorData;

        unsafe fn tensor(
            data_type: DataType,
            data: *mut TensorData,
            shape: UniquePtr<Shape>,
        ) -> Result<UniquePtr<Tensor>>;

        fn executor_config() -> UniquePtr<ExecutorConfig>;

        fn executor(
            model_path: &str,
            model_type: ModelType,
            executor_config: &ExecutorConfig,
        ) -> Result<UniquePtr<Executor>>;

        fn request(input_token_ids: &[u32], max_tokens: u32) -> UniquePtr<Request>;

        fn set_encoder_input_features(
            request: Pin<&mut Request>,
            features: UniquePtr<Tensor>,
        ) -> Result<()>;

        fn await_responses(
            executor: Pin<&mut Executor>,
            request_id: u64,
        ) -> Result<UniquePtr<CxxVector<Response>>>;

        fn get_num_responses_ready(executor: &Executor, request_id: u64) -> Result<u32>;

        fn get_result(response: &Response) -> Result<Result>;
    }
}

pub struct Shape {
    ptr: UniquePtr<ffi::Shape>,
}

impl Shape {
    pub fn new(dims: &[usize]) -> Self {
        let ptr = ffi::shape(dims);
        Shape { ptr }
    }
}

pub struct Tensor {
    ptr: UniquePtr<ffi::Tensor>,
}

impl Tensor {
    pub fn of<T>(data_type: DataType, data: &[T], shape: Shape) -> anyhow::Result<Self> {
        let data = data.as_ptr() as *mut ffi::TensorData;
        let ptr = unsafe { ffi::tensor(data_type, data, shape.ptr) }
            .map_err(|e| anyhow!("Failed to create tensor: {}", e))?;
        Ok(Tensor { ptr })
    }
}

pub struct ExecutorConfig {
    ptr: UniquePtr<ffi::ExecutorConfig>,
}

impl ExecutorConfig {
    pub fn new() -> Self {
        let ptr = ffi::executor_config();
        ExecutorConfig { ptr }
    }
}

pub struct Request {
    ptr: UniquePtr<ffi::Request>,
}

impl Request {
    pub fn new(input_token_ids: &[u32], max_tokens: u32) -> Self {
        let ptr = ffi::request(input_token_ids, max_tokens);
        Request { ptr }
    }

    pub fn set_streaming(&mut self, streaming: bool) -> anyhow::Result<()> {
        self.ptr
            .pin_mut()
            .set_streaming(streaming)
            .map_err(|e| anyhow!("Failed to set streaming: {}", e))
    }

    pub fn set_end_id(&mut self, end_id: u32) -> anyhow::Result<()> {
        self.ptr
            .pin_mut()
            .set_end_id(end_id as i32)
            .map_err(|e| anyhow!("Failed to set end id: {}", e))
    }

    pub fn set_pad_id(&mut self, pad_id: u32) -> anyhow::Result<()> {
        self.ptr
            .pin_mut()
            .set_pad_id(pad_id as i32)
            .map_err(|e| anyhow!("Failed to set pad id: {}", e))
    }

    pub fn set_encoder_input_features(&mut self, features: Tensor) -> anyhow::Result<()> {
        ffi::set_encoder_input_features(self.ptr.pin_mut(), features.ptr)
            .map_err(|e| anyhow!("Failed to set encoder input features: {}", e))
    }
}

impl Response {
    pub fn get_result(&self) -> anyhow::Result<Result> {
        ffi::get_result(self).map_err(|e| anyhow!("Failed to get result from response: {}", e))
    }
}

pub struct ResponseIterator<'a> {
    responses: UniquePtr<cxx::CxxVector<Response>>,
    pos: usize,
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> ResponseIterator<'a> {
    fn new(responses: UniquePtr<cxx::CxxVector<Response>>) -> Self {
        ResponseIterator {
            responses,
            pos: 0,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a> Iterator for ResponseIterator<'a> {
    type Item = &'a Response;

    fn next(&mut self) -> Option<Self::Item> {
        let next = unsafe { (*self.responses.as_ptr()).get(self.pos) }?;
        self.pos += 1;
        Some(next)
    }
}

impl From<VecTokens> for Vec<u32> {
    fn from(value: VecTokens) -> Self {
        value.v
    }
}

impl Deref for VecTokens {
    type Target = Vec<u32>;

    fn deref(&self) -> &Self::Target {
        &self.v
    }
}

impl Result {
    pub fn is_final(&self) -> bool {
        self.is_final
    }

    pub fn output_token_ids(&self) -> &[VecTokens] {
        self.output_token_ids.as_slice()
    }
}

pub struct Executor {
    ptr: UniquePtr<ffi::Executor>,
}

impl Executor {
    pub fn open<P: AsRef<Path>>(
        model_path: P,
        model_type: ModelType,
        config: &ExecutorConfig,
    ) -> anyhow::Result<Self> {
        let model_path = model_path.as_ref();
        let path = model_path
            .to_str()
            .ok_or_else(|| anyhow!("Invalid model path: {}", model_path.display()))?;
        let ptr = ffi::executor(path, model_type, &config.ptr)?;
        Ok(Executor { ptr })
    }

    pub fn enqueue_request(&mut self, request: &Request) -> anyhow::Result<u64> {
        self.ptr
            .pin_mut()
            .enqueue_request(&request.ptr)
            .map_err(|e| anyhow!("Failed to enqueue request: {}", e))
    }

    pub fn await_responses(
        &mut self,
        request_id: u64,
    ) -> anyhow::Result<impl Iterator<Item = &Response>> {
        let responses = ffi::await_responses(self.ptr.pin_mut(), request_id)?;
        Ok(ResponseIterator::new(responses))
    }

    pub fn get_num_responses_ready(&self, request_id: u64) -> anyhow::Result<usize> {
        let num = ffi::get_num_responses_ready(&self.ptr, request_id)
            .map_err(|e| anyhow!("Failed to get number of responses ready: {}", e))?;
        Ok(num as usize)
    }
}
