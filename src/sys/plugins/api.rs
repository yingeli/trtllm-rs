use anyhow::{Result, anyhow};

use cxx::UniquePtr;

#[cxx::bridge]
mod ffi {
    extern "C++" {
        include!("trtllm/src/sys/plugins/api.h");

        unsafe fn init_trtllm_plugins() -> Result<bool>;
    }
}

pub fn init_trtllm_plugins() -> Result<bool> {
    unsafe {
        ffi::init_trtllm_plugins().map_err(|e| anyhow!("Failed to initialize plugins: {}", e))
    }
}
