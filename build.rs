use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const OFFSET: u64 = 0xcbf29ce484222325;
const PRIME: u64 = 0x100000001b3;

fn collect_rust_sources(directory: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            collect_rust_sources(&path, files)?;
        } else if path.extension().is_some_and(|ext| ext == "rs") && path.is_file() {
            files.push(path);
        }
    }
    Ok(())
}

fn update(hash: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *hash = (*hash ^ u64::from(byte)).wrapping_mul(PRIME);
    }
}

fn main() -> std::io::Result<()> {
    let root =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("Cargo sets the manifest path"));
    let mut paths = vec![
        root.join("Cargo.toml"),
        root.join("Cargo.lock"),
        root.join("build.rs"),
    ];
    collect_rust_sources(&root.join("src"), &mut paths)?;
    let mut files: Vec<_> = paths
        .into_iter()
        .map(|path| {
            let relative = path
                .strip_prefix(&root)
                .expect("sources are below the manifest");
            let name = relative
                .components()
                .map(|part| part.as_os_str().to_str().expect("source paths are UTF-8"))
                .collect::<Vec<_>>()
                .join("/");
            (name, path)
        })
        .collect();
    files.sort_by(|left, right| left.0.cmp(&right.0));

    // A content checksum for accidental stale-build detection, not a security
    // signature. Keep the framing and FNV-1a calculation in sync with tools/native_build.py.
    let mut hash = OFFSET;
    update(&mut hash, b"mixedlm-native-v1\0");
    for (name, path) in files {
        update(&mut hash, name.as_bytes());
        update(&mut hash, &[0]);
        update(&mut hash, &fs::read(path)?);
        update(&mut hash, &[0]);
    }
    println!("cargo:rustc-env=MIXEDLM_SOURCE_FINGERPRINT=fnv1a64:{hash:016x}");
    for input in ["src", "Cargo.toml", "Cargo.lock", "build.rs"] {
        println!("cargo:rerun-if-changed={input}");
    }
    Ok(())
}
