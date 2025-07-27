use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/sys");
    println!("cargo:rerun-if-changed=build.rs");

    println!("cargo:rustc-link-search=/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs");
    println!("cargo:rustc-link-lib=nvinfer_plugin_tensorrt_llm");
    println!("cargo:rustc-link-lib=tensorrt_llm");

    cxx_build::bridges(["src/sys/executor.rs", "src/sys/plugins/api.rs"])
        .file("src/sys/executor.cpp")
        .include("/app/tensorrt_llm/include")
        .std("c++20")
        .cuda(true)
        .static_crt(cfg!(target_os = "windows"))
        .flag_if_supported("/EHsc")
        .compile("trtllm");
}
