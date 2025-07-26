use std::path::Path;
use anyhow::{anyhow, Result};

use cxx::UniquePtr;

pub use self::ffi::ModelType;

#[cxx::bridge]
mod ffi {
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

    #[namespace = "tensorrt_llm::executor"]
    unsafe extern "C++" {
        include!("tensorrt_llm/executor/executor.h");
        
        type ModelType;

        type ExecutorConfig;

        type Request;

        type Response;

        type Executor;

        #[cxx_name = "enqueueRequest"]
        fn enqueue_request(
            self: Pin<&mut Executor>,
            request: &Request
        ) -> Result<u64>;
    }

    unsafe extern "C++" {       
        include!("trtllm/src/sys/executor.h");

        fn executor_config() -> Result<UniquePtr<ExecutorConfig>>;

        fn request(
            input_token_ids: &[u32],
            max_tokens: u32
        ) -> Result<UniquePtr<Request>>;

        fn await_response(
            executor: Pin<&mut Executor>,
            request_id: u64
        ) -> Result<UniquePtr<Response>>;

        fn get_num_responses_ready(
            executor: &Executor,
            request_id: u64
        ) -> Result<u32>;

        fn executor(
            model_path: &str,
            model_type: ModelType,
            executor_config: &ExecutorConfig
        ) -> Result<UniquePtr<Executor>>;
    }
}

pub struct ExecutorConfig {
    ptr: UniquePtr<ffi::ExecutorConfig>,
}

impl ExecutorConfig {
    pub fn new() -> Result<Self> {
        let ptr = ffi::executor_config()?;
        Ok(ExecutorConfig { ptr })
    }

    fn as_ptr(&self) -> &ffi::ExecutorConfig {
        &self.ptr
    }
}

pub struct Request {
    ptr: UniquePtr<ffi::Request>,
}

impl Request {
    pub fn new(input_token_ids: &[u32], max_tokens: u32) -> Result<Self> {
        let ptr = ffi::request(input_token_ids, max_tokens)?;
        Ok(Request { ptr })
    }

    fn as_ptr(&self) -> &ffi::Request {
        &self.ptr
    }
}

pub struct Executor {
    ptr: UniquePtr<ffi::Executor>,
}

impl Executor {
    pub fn new<P: AsRef<Path>>(model_path: P, model_type: ModelType, config: &ExecutorConfig) -> Result<Self> {
        let model_path = model_path.as_ref();
        let path = model_path.to_str().ok_or_else(|| anyhow!("Invalid model path: {}", model_path.display()))?;
        let ptr = ffi::executor(path, model_type, config.as_ptr())?;
        Ok(Executor { ptr })
    }

    pub fn enqueue_request(&mut self, request: &Request) -> Result<u64> {
        self.ptr.pin_mut().enqueue_request(request.as_ptr())
            .map_err(|e| anyhow!("Failed to enqueue request: {}", e))
    }

    fn await_response(&mut self, request_id: u64) -> Result<UniquePtr<ffi::Response>> {
        ffi::await_response(self.ptr.pin_mut(), request_id)
            .map_err(|e| anyhow!("Failed to await response: {}", e))
    }

    fn get_num_responses_ready(&self, request_id: u64) -> Result<usize> {
        let num = ffi::get_num_responses_ready(self.as_ptr(), request_id)
            .map_err(|e| anyhow!("Failed to get number of responses ready: {}", e))?;
        Ok(num as usize)
    }

    fn as_ptr(&self) -> &ffi::Executor {
        &self.ptr
    }
}