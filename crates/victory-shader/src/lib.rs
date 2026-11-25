//! Xbox Series shaders to host shaders.
//!
//! An Xbox Series shader is a standard DXBC container ([`dxbc`]). Its `DXIL`
//! part ([`dxil`]) may carry Xbox additions around the LLVM bitcode: an
//! `XDAT` table before it indexing blobs after it, among them `xbop`
//! (compiler options) and the zlib-compressed GFX10 code. The bitcode may use
//! console-only intrinsics, so the GFX10 code is translated in general.

pub mod bvh;
pub mod dxbc;
pub mod dxil;
pub mod emit;
pub mod gfx10;
pub mod io;
pub mod ir;
pub mod native;
pub mod resource;
pub mod spill;
pub mod structure;

pub use dxbc::{Container, FourCC, Part};
pub use dxil::{DxilProgram, ShaderKind, XboxExtras};
pub use native::{
    HwStage, NativeProgram, ResourceType, RootParam, RootType, UsedResource, native_programs,
};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ShaderError {
    #[error("not a DXBC container")]
    NotDxbc,
    #[error("malformed shader: {0}")]
    Malformed(&'static str),
    #[error("GFX10 code at {offset:#x}: {reason}")]
    Isa { offset: usize, reason: &'static str },
    /// Valid code the translator cannot handle yet: an opcode name or a construct.
    #[error("GFX10 code at {offset:#x}: {what} is not supported")]
    Unsupported { offset: usize, what: &'static str },
}

/// The translator's own sources, by which translations kept on disk tell
/// whether this translator made them.
pub fn sources() -> &'static [&'static str] {
    &[
        include_str!("bvh.rs"),
        include_str!("dxbc.rs"),
        include_str!("dxil.rs"),
        include_str!("emit/bitcode.rs"),
        include_str!("emit/bary_fallback.rs"),
        include_str!("emit/bitcode/census.rs"),
        include_str!("emit/bitcode/opt.rs"),
        include_str!("emit/container.rs"),
        include_str!("emit/diff_tests.rs"),
        include_str!("emit/host.rs"),
        include_str!("emit/intersect.rs"),
        include_str!("emit/intersect_tests.rs"),
        include_str!("emit/lower.rs"),
        include_str!("emit/mesh_fallback.rs"),
        include_str!("emit/mod.rs"),
        include_str!("emit/own_bits.rs"),
        include_str!("emit/rect.rs"),
        include_str!("emit/sampler.rs"),
        include_str!("emit/shader.rs"),
        include_str!("emit/threading.rs"),
        include_str!("emit/validator.rs"),
        include_str!("emit/valu_tests.rs"),
        include_str!("emit/waterfall_tests.rs"),
        include_str!("emit/wave.rs"),
        include_str!("emit/wave64_tests.rs"),
        include_str!("gfx10/fields.rs"),
        include_str!("gfx10/mod.rs"),
        include_str!("gfx10/opcodes.rs"),
        include_str!("gfx10/sp3.rs"),
        include_str!("io.rs"),
        include_str!("ir.rs"),
        include_str!("lib.rs"),
        include_str!("native.rs"),
        include_str!("resource.rs"),
        include_str!("spill.rs"),
        include_str!("structure.rs"),
    ]
}

#[cfg(test)]
mod source_tests {
    /// Every source file is in the list.
    #[test]
    fn every_source_is_listed() {
        fn count(dir: &std::path::Path) -> usize {
            std::fs::read_dir(dir)
                .unwrap()
                .map(|e| e.unwrap().path())
                .map(|p| if p.is_dir() { count(&p) } else { usize::from(p.extension().is_some_and(|x| x == "rs")) })
                .sum()
        }
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        assert_eq!(super::sources().len(), count(&dir));
    }
}
