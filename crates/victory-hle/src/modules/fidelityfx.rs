//! Keep the title's FidelityFX context and queries, with an optional spatial
//! dispatch on GPUs that cannot run its upscaler. This is a partial override:
//! the real title DLL must still be mapped and initialized.
use std::ffi::c_void;

const DLL: &str = "amd_fidelityfx_loader_xbox.dll";

pub(crate) fn resolve(dll: &str, symbol: &str) -> Option<usize> {
    if !dll.eq_ignore_ascii_case(DLL) {
        return None;
    }
    match symbol {
        "ffxDispatch" => Some(dispatch as *const () as usize),
        _ => None,
    }
}

extern "system" fn dispatch(context: *mut c_void, desc: *const c_void) -> u32 {
    if victory_gpu::try_spatial_upscale(desc) {
        return 0;
    }
    original(c"ffxDispatch", context, desc.cast_mut())
}

fn original(name: &std::ffi::CStr, context: *mut c_void, desc: *mut c_void) -> u32 {
    let Some(module) = crate::runtime::image_named(DLL) else {
        return 6;
    };
    // SAFETY: this module is a mapped title image, kept loaded by the runtime.
    let original = unsafe {
        windows_sys::Win32::System::LibraryLoader::GetProcAddress(
            module as *mut c_void,
            name.as_ptr().cast(),
        )
    };
    let Some(original) = original else { return 6 };
    // SAFETY: the original export has the public FidelityFX dispatch ABI.
    let original: extern "system" fn(*mut c_void, *mut c_void) -> u32 =
        unsafe { std::mem::transmute(original) };
    original(context, desc)
}
