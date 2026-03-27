use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let src_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("vendor");

    // --- Detect CUDA toolkit ---
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| {
            // Try common paths
            for p in &[
                "/usr/local/cuda",
                "/usr/local/cuda-13.2",
                "/usr/local/cuda-12.6",
                "/usr/local/cuda-12",
                "/opt/cuda",
            ] {
                if std::path::Path::new(p).exists() {
                    return p.to_string();
                }
            }
            panic!("CUDA toolkit not found. Set CUDA_PATH or CUDA_HOME environment variable.");
        });
    let nvcc = format!("{}/bin/nvcc", cuda_path);
    let cuda_include = format!("{}/include", cuda_path);
    let cuda_lib = format!("{}/lib64", cuda_path);

    // --- Detect compute capability ---
    let ccap = env::var("CUDA_CCAP").unwrap_or_else(|_| "120".to_string());

    // --- GPU tuning defines ---
    let blocks_per_grid = env::var("NOSTR_BLOCKS_PER_GRID").unwrap_or_else(|_| "3072".to_string());
    let threads_per_block =
        env::var("NOSTR_THREADS_PER_BLOCK").unwrap_or_else(|_| "256".to_string());
    let keys_per_batch = env::var("KEYS_PER_THREAD_BATCH").unwrap_or_else(|_| "256".to_string());

    let gpu_defines = [
        format!("-DNOSTR_BLOCKS_PER_GRID={}", blocks_per_grid),
        format!("-DNOSTR_THREADS_PER_BLOCK={}", threads_per_block),
        format!("-DKEYS_PER_THREAD_BATCH={}", keys_per_batch),
    ];

    // --- Compile .cu files with nvcc ---
    let cu_files = &[
        ("GPU/GPURummage.cu", true),    // needs gpu_defines
        ("GPU/CudaPowMiner.cu", false), // no gpu_defines needed
        ("GPU/rummage_ffi.cu", true),   // needs gpu_defines (wraps GPURummage)
    ];

    let mut obj_files: Vec<PathBuf> = Vec::new();

    for (cu_file, needs_gpu_defines) in cu_files {
        let src_path = src_dir.join(cu_file);
        let obj_name = cu_file.replace('/', "_").replace(".cu", ".o");
        let obj_path = out_dir.join(&obj_name);

        let mut cmd = Command::new(&nvcc);
        cmd.arg("-allow-unsupported-compiler")
            .arg("--compile")
            .args(["--compiler-options", "-fPIC"])
            .arg("-O2")
            .arg(format!("-I{}", src_dir.display()))
            .arg(format!("-I{}", cuda_include))
            .arg(format!("-gencode=arch=compute_{},code=sm_{}", ccap, ccap))
            .arg("-o")
            .arg(&obj_path)
            .arg("-c")
            .arg(&src_path);

        if *needs_gpu_defines {
            for d in &gpu_defines {
                cmd.arg(d);
            }
        }

        let status = cmd
            .status()
            .unwrap_or_else(|e| panic!("Failed to run nvcc on {}: {}", cu_file, e));
        assert!(status.success(), "nvcc failed on {}", cu_file);
        obj_files.push(obj_path);
    }

    // --- Compile .cpp files with cc crate ---
    let cpp_files = &[
        "CPU/Point.cpp",
        "CPU/Int.cpp",
        "CPU/IntMod.cpp",
        "CPU/SECP256K1.cpp",
    ];

    let mut cc_build = cc::Build::new();
    cc_build
        .cpp(true)
        .flag("-O2")
        .flag("-march=native")
        .flag("-Wno-write-strings")
        .flag("-fPIC")
        .define("WITHGPU", None)
        .define("NOSTR_BLOCKS_PER_GRID", blocks_per_grid.as_str())
        .define("NOSTR_THREADS_PER_BLOCK", threads_per_block.as_str())
        .define("KEYS_PER_THREAD_BATCH", keys_per_batch.as_str())
        .include(&src_dir)
        .include(&cuda_include);

    for f in cpp_files {
        cc_build.file(src_dir.join(f));
    }

    cc_build.compile("rummage_cpp");

    // --- Create static archive from CUDA objects ---
    let cuda_lib_path = out_dir.join("librummage_cuda.a");
    let mut ar_cmd = Command::new("ar");
    ar_cmd.arg("rcs").arg(&cuda_lib_path);
    for obj in &obj_files {
        ar_cmd.arg(obj);
    }
    let status = ar_cmd.status().expect("Failed to run ar");
    assert!(status.success(), "ar failed");

    // --- Link instructions ---
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=rummage_cuda");
    println!("cargo:rustc-link-search=native={}", cuda_lib);
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=curand");
    println!("cargo:rustc-link-lib=dylib=gmp");
    println!("cargo:rustc-link-lib=dylib=ssl");
    println!("cargo:rustc-link-lib=dylib=crypto");
    println!("cargo:rustc-link-lib=dylib=stdc++");

    // --- Rerun triggers ---
    println!(
        "cargo:rerun-if-changed={}",
        src_dir.join("GPU/rummage_ffi.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        src_dir.join("GPU/rummage_ffi.cu").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        src_dir.join("GPU/CudaPowMiner.cu").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        src_dir.join("GPU/CudaPowMiner.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        src_dir.join("GPU/GPURummage.cu").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        src_dir.join("GPU/GPURummage.h").display()
    );
    for f in cpp_files {
        println!("cargo:rerun-if-changed={}", src_dir.join(f).display());
    }

    // --- bindgen ---
    let header = src_dir.join("GPU/rummage_ffi.h");
    let bindings = bindgen::Builder::default()
        .header(header.to_str().unwrap())
        .allowlist_function("rummage_.*")
        .allowlist_type("Rummage.*")
        .allowlist_var("RUMMAGE_.*")
        .generate_comments(true)
        .derive_debug(true)
        .derive_default(true)
        .generate()
        .expect("bindgen failed on rummage_ffi.h");

    let bindings_path = out_dir.join("bindings.rs");
    bindings
        .write_to_file(&bindings_path)
        .expect("Failed to write bindings.rs");
}
