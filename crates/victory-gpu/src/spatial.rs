//! Spatial replacement for FidelityFX upscale dispatches. No temporal history,
//! motion reconstruction or sharpening; ordinary filtered scaling only.
//! Descriptor prefix follows the public ffxDispatchDescUpscale ABI:
//! https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK
use crate::{host, object::This};
use std::ffi::c_void;
use std::sync::OnceLock;

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

const SOURCE: &str = r#"
#define RS "DescriptorTable(SRV(t0), UAV(u0)), RootConstants(num32BitConstants=4,b0), StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP, addressW=TEXTURE_ADDRESS_CLAMP)"
Texture2D<float4> source : register(t0);
RWTexture2D<float4> target : register(u0);
SamplerState linearClamp : register(s0);
cbuffer Dimensions : register(b0) { uint2 renderSize; uint2 outputSize; }
[RootSignature(RS)]
[numthreads(8,8,1)]
void main(uint2 id : SV_DispatchThreadID) {
    if (any(id >= outputSize)) return;
    uint w,h; source.GetDimensions(w,h);
    float2 pixel = clamp((id + 0.5) * renderSize / outputSize, 0.5, renderSize - 0.5);
    target[id] = source.SampleLevel(linearClamp, pixel / float2(w,h), 0);
}
"#;

struct Pass {
    root: ID3D12RootSignature,
    pipeline: ID3D12PipelineState,
}
// SAFETY: immutable, free-threaded D3D12 objects; recording uses each caller's list.
unsafe impl Send for Pass {}
unsafe impl Sync for Pass {}

impl Pass {
    fn new() -> Option<Self> {
        let h = host::host()?;
        let code = host::compile(SOURCE, s!("main"), s!("cs_5_1")).ok()?;
        // SAFETY: compiled bytes and the borrowed root remain live throughout creation.
        unsafe {
            let root: ID3D12RootSignature = h.device.CreateRootSignature(0, &code).ok()?;
            let pipeline = h
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .ok()?;
            Some(Self { root, pipeline })
        }
    }
    fn record(
        &self,
        list: &host::HostList,
        source: &ID3D12Resource,
        target: &ID3D12Resource,
        render: [u32; 2],
    ) -> Option<()> {
        let (sd, dd) = unsafe { (source.GetDesc(), target.GetDesc()) };
        let Some(h) = host::host() else { return None };
        // SAFETY: each dispatch owns its descriptors until list reset; no heap slot reuse.
        let views: ID3D12DescriptorHeap = match unsafe {
            h.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 2,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
        } {
            Ok(v) => v,
            Err(_) => return None,
        };
        unsafe {
            let cpu = views.GetCPUDescriptorHandleForHeapStart();
            let step = h
                .device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: sd.Format,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV {
                        MipLevels: 1,
                        ..Default::default()
                    },
                },
            };
            let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: dd.Format,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_UAV::default(),
                },
            };
            h.device.CreateShaderResourceView(source, Some(&srv), cpu);
            h.device.CreateUnorderedAccessView(
                target,
                None,
                Some(&uav),
                D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: cpu.ptr + step as usize,
                },
            );
        }
        let l = &list.list;
        unsafe {
            l.SetDescriptorHeaps(&[Some(views.clone())]);
            l.SetComputeRootSignature(&self.root);
            l.SetPipelineState(&self.pipeline);
            l.SetComputeRootDescriptorTable(0, views.GetGPUDescriptorHandleForHeapStart());
            let dims = [render[0], render[1], dd.Width as u32, dd.Height];
            l.SetComputeRoot32BitConstants(1, 4, dims.as_ptr().cast(), 0);
            l.Dispatch((dd.Width as u32).div_ceil(8), dd.Height.div_ceil(8), 1);
        }
        crate::mesh_fallback::barrier(list, target);
        list.keep(&views);
        list.keep(&self.root);
        list.keep(&self.pipeline);
        Some(())
    }
}

/// Tries only the known public upscale descriptor and our own GPU objects.
/// Unknown descriptors and unsupported output formats stay with the title DLL.
pub fn try_spatial_upscale(desc: *const c_void) -> bool {
    let enabled = match std::env::var("VICTORY_FSR_FALLBACK").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => host::host().is_some_and(|h| !h.mesh_shaders()),
    };
    if !enabled {
        return false;
    }
    let mut bytes = [0u8; 384];
    if !host::read_into(desc.cast(), &mut bytes) {
        return false;
    }
    let word = |offset| u32::from_ne_bytes(bytes[offset..offset + 4].try_into().unwrap());
    let pointer = |offset| usize::from_ne_bytes(bytes[offset..offset + 8].try_into().unwrap());
    if pointer(0) != 0x10001 {
        return false;
    }
    let list = pointer(16) as This;
    let source = pointer(24) as This;
    let output = pointer(312) as This;
    let render = [word(376), word(380)];
    if !crate::list::owns(list)
        || !crate::resource::owns(source)
        || !crate::resource::owns(output)
        || source == output
    {
        return false;
    }
    let Some(src) = crate::resource::host_copy(source) else {
        return false;
    };
    let Some(dst) = crate::resource::host_copy(output) else {
        return false;
    };
    // SAFETY: these resources are owned by our runtime.
    let (sd, dd) = unsafe { (src.texture.GetDesc(), dst.texture.GetDesc()) };
    if sd.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
        || dd.Dimension != sd.Dimension
        || sd.DepthOrArraySize != 1
        || dd.DepthOrArraySize != 1
        || sd.SampleDesc.Count != 1
        || dd.SampleDesc.Count != 1
        || !dd
            .Flags
            .contains(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)
        || render.contains(&0)
        || render[0] as u64 > sd.Width
        || render[1] > sd.Height
        || dd.Width == 0
        || dd.Height == 0
        || dd.Width > 16384
        || dd.Height > 16384
    {
        return false;
    }
    // Only typed float-compatible color formats. Avoid typeless, integer and depth views.
    if ![2, 10, 11, 24, 26, 28, 87].contains(&sd.Format.0)
        || ![2, 10, 11, 24, 26, 28].contains(&dd.Format.0)
    {
        return false;
    }
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    let Some(pass) = PASS.get_or_init(Pass::new).as_ref() else {
        return false;
    };
    let mut recorded = false;
    crate::list::with_state(list, |list, state| {
        let l = &list.list;
        unsafe {
            for (resource, wanted) in [
                (source, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
                (output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
            ] {
                if let Some((texture, before)) = list.transition(resource, wanted.0 as u32) {
                    l.ResourceBarrier(&[host::transition(
                        texture,
                        D3D12_RESOURCE_STATES(before as i32),
                        wanted,
                    )]);
                }
            }
        }
        recorded = pass
            .record(list, &src.texture, &dst.texture, render)
            .is_some();
        state.replay(list);
    });
    if recorded {
        static LOG: std::sync::Once = std::sync::Once::new();
        LOG.call_once(|| {
            tracing::warn!(
                ?render,
                output_width = dd.Width,
                output_height = dd.Height,
                "using spatial FidelityFX fallback (temporal reconstruction disabled)"
            )
        });
    }
    recorded
}

#[cfg(all(test, feature = "diagnostics"))]
mod tests {
    use super::*;
    use windows::Win32::Foundation::RECT;
    use windows::Win32::Graphics::Dxgi::Common::*;

    #[test]
    #[ignore = "requires a D3D12 device"]
    fn spatial_scale_clamps_to_render_extent_and_handles_partial_groups() {
        let h = host::host().expect("D3D12 device");
        unsafe {
            let texture = |w, hgt, flags, state| {
                let mut result: Option<ID3D12Resource> = None;
                h.device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES {
                            Type: D3D12_HEAP_TYPE_DEFAULT,
                            ..Default::default()
                        },
                        D3D12_HEAP_FLAG_NONE,
                        &D3D12_RESOURCE_DESC {
                            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                            Width: w,
                            Height: hgt,
                            DepthOrArraySize: 1,
                            MipLevels: 1,
                            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                            SampleDesc: DXGI_SAMPLE_DESC {
                                Count: 1,
                                Quality: 0,
                            },
                            Flags: flags,
                            ..Default::default()
                        },
                        state,
                        None,
                        &mut result,
                    )
                    .unwrap();
                result.unwrap()
            };
            let src = texture(
                4,
                2,
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
            );
            let dst = texture(
                5,
                3,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            );
            let list = h.new_list().unwrap();
            let rtv = h.rtv(Some(&src), None);
            list.list
                .ClearRenderTargetView(rtv, &[0., 1., 0., 1.], None);
            list.list.ClearRenderTargetView(
                rtv,
                &[1., 0., 0., 1.],
                Some(&[RECT {
                    left: 0,
                    top: 0,
                    right: 2,
                    bottom: 2,
                }]),
            );
            list.list.ResourceBarrier(&[host::transition(
                &src,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            )]);
            Pass::new()
                .unwrap()
                .record(&list, &src, &dst, [2, 2])
                .unwrap();
            list.list.ResourceBarrier(&[host::transition(
                &dst,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            let (buffer, footprint) =
                crate::capture::read_back(&h.device, &list.list, &dst).unwrap();
            list.close();
            h.execute(&[&list]);
            let mut ptr = std::ptr::null_mut();
            buffer.Map(0, None, Some(&mut ptr)).unwrap();
            let data = std::slice::from_raw_parts(
                ptr.cast::<u8>(),
                footprint.Footprint.RowPitch as usize * 3,
            );
            for row in 0..3 {
                for col in 0..5 {
                    let at = row * footprint.Footprint.RowPitch as usize + col * 4;
                    assert_eq!(&data[at..at + 4], &[255, 0, 0, 255]);
                }
            }
            buffer.Unmap(0, None);
            assert!(h.device.GetDeviceRemovedReason().is_ok());
        }
    }
}
