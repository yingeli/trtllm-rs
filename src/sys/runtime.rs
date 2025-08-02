use anyhow::{Result, anyhow};

use cxx::{SharedPtr, UniquePtr};

// pub use self::ffi::{};

#[cxx::bridge]
mod ffi {
    #[namespace = "tensorrt_llm::runtime"]
    extern "C++" {
        include!("tensorrt_llm/runtime/bufferManager.h");

        type CudaStream;

        type BufferManager;

    }

    extern "C++" {
        include!("trtllm/src/sys/runtime.h");

        type CUstream_st;

        unsafe fn cudaStream(stream: *mut CUstream_st) -> SharedPtr<CudaStream>;

        unsafe fn bufferManager(stream: SharedPtr<CudaStream>) -> UniquePtr<BufferManager>;
    }
}

pub struct CudaStream {
    ptr: SharedPtr<ffi::CudaStream>,
}

impl Default for CudaStream {
    fn default() -> Self {
        CudaStream {
            ptr: unsafe { ffi::cudaStream(std::ptr::null_mut()) },
        }
    }
}

impl CudaStream {
    pub fn new<T>(stream: *mut T) -> Self {
        let stream = stream as *mut ffi::CUstream_st;
        let ptr = unsafe { ffi::cudaStream(stream) };
        CudaStream { ptr }
    }
}

pub struct BufferManager {
    ptr: UniquePtr<ffi::BufferManager>,
}

impl BufferManager {
    pub fn new(stream: CudaStream) -> Self {
        let ptr = unsafe { ffi::bufferManager(stream.ptr) };
        BufferManager { ptr }
    }
}
