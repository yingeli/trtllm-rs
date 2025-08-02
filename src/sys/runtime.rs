use anyhow::anyhow;
use std::ops::Deref;
use std::path::Path;

use cxx::UniquePtr;

pub use self::ffi::{ModelType, Response, Result, VecTokens};

#[cxx::bridge]
mod ffi {
    #[namespace = "tensorrt_llm::executor"]
    unsafe extern "C++" {
        include!("tensorrt_llm/executor/executor.h");
    }    
}