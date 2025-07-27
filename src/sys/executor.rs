use anyhow::anyhow;
use std::ops::Deref;
use std::path::Path;

use cxx::UniquePtr;

pub use self::ffi::{ModelType, Response, Result, VecTokens};

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
    unsafe extern "C++" {
        include!("tensorrt_llm/executor/executor.h");

        type ModelType;
        type ExecutorConfig;

        type Request;

        #[cxx_name = "setStreaming"]
        fn set_streaming(self: Pin<&mut Request>, streaming: bool) -> Result<()>;

        #[cxx_name = "setEndId"]
        fn set_end_id(self: Pin<&mut Request>, end_id: i32) -> Result<()>;

        #[cxx_name = "setPadId"]
        fn set_pad_id(self: Pin<&mut Request>, pad_id: i32) -> Result<()>;

        // type Result;
        type Response;

        //#[cxx_name = "getResult"]
        //pub fn get_result(self: &Response) -> &Result;

        type Executor;

        #[cxx_name = "enqueueRequest"]
        fn enqueue_request(self: Pin<&mut Executor>, request: &Request) -> Result<u64>;
    }

    unsafe extern "C++" {
        include!("trtllm/src/sys/executor.h");

        fn executor_config() -> UniquePtr<ExecutorConfig>;

        fn executor(
            model_path: &str,
            model_type: ModelType,
            executor_config: &ExecutorConfig,
        ) -> Result<UniquePtr<Executor>>;

        fn request(input_token_ids: &[u32], max_tokens: u32) -> UniquePtr<Request>;

        fn await_responses(
            executor: Pin<&mut Executor>,
            request_id: u64,
        ) -> Result<UniquePtr<CxxVector<Response>>>;

        /*
        fn await_response(
            executor: Pin<&mut Executor>,
            request_id: u64,
        ) -> Result<UniquePtr<Response>>;
         */

        fn get_num_responses_ready(executor: &Executor, request_id: u64) -> Result<u32>;

        fn get_result(response: &Response) -> Result<Result>;
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

impl Response {
    pub fn get_result(&self) -> anyhow::Result<Result> {
        ffi::get_result(self).map_err(|e| anyhow!("Failed to get result from response: {}", e))
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

pub struct ExecutorConfig {
    ptr: UniquePtr<ffi::ExecutorConfig>,
}

impl ExecutorConfig {
    pub fn new() -> Self {
        let ptr = ffi::executor_config();
        ExecutorConfig { ptr }
    }

    fn as_ptr(&self) -> &ffi::ExecutorConfig {
        &self.ptr
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

    fn as_ptr(&self) -> &ffi::Request {
        &self.ptr
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
        let ptr = ffi::executor(path, model_type, config.as_ptr())?;
        Ok(Executor { ptr })
    }

    pub fn enqueue_request(&mut self, request: &Request) -> anyhow::Result<u64> {
        self.ptr
            .pin_mut()
            .enqueue_request(request.as_ptr())
            .map_err(|e| anyhow!("Failed to enqueue request: {}", e))
    }

    pub fn await_responses(
        &mut self,
        request_id: u64,
    ) -> anyhow::Result<impl Iterator<Item = &Response>> {
        let responses = ffi::await_responses(self.ptr.pin_mut(), request_id)?;
        Ok(ResponseIterator::new(responses))
    }

    /*
    pub fn await_response(&mut self, request_id: u64) -> Result<UniquePtr<Response>> {
        let response = ffi::await_response(self.ptr.pin_mut(), request_id)
            .map_err(|e| anyhow!("Failed to await response: {}", e))?;
        Ok(response)
    }
    */

    pub fn get_num_responses_ready(&self, request_id: u64) -> anyhow::Result<usize> {
        let num = ffi::get_num_responses_ready(self.as_ptr(), request_id)
            .map_err(|e| anyhow!("Failed to get number of responses ready: {}", e))?;
        Ok(num as usize)
    }

    fn as_ptr(&self) -> &ffi::Executor {
        &self.ptr
    }
}
