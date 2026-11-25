//! SDR presentation of separate scene and UI planes. Copying only the
//! scene plane loses Series S menus: their pixels live in the UI plane.
//! The display compositor scales each source rectangle into its destination
//! and blends subsequent planes using premultiplied alpha.

use std::sync::{Mutex, OnceLock};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::s;

use crate::host;

pub(crate) struct Layer {
    pub texture: ID3D12Resource,
    pub state: u32,
    pub source: Option<[i32; 4]>,
    pub destination: Option<[f32; 4]>,
}

const SHADER: &str = r#"
#define RS "DescriptorTable(SRV(t0), visibility=SHADER_VISIBILITY_PIXEL), RootConstants(num32BitConstants=4, b0, visibility=SHADER_VISIBILITY_VERTEX), StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_LINEAR, addressU=TEXTURE_ADDRESS_CLAMP, addressV=TEXTURE_ADDRESS_CLAMP, addressW=TEXTURE_ADDRESS_CLAMP, visibility=SHADER_VISIBILITY_PIXEL)"
Texture2D<float4> image : register(t0);
SamplerState linearClamp : register(s0);
cbuffer View : register(b0) { float4 source; }
struct Vertex { float4 position : SV_Position; float2 uv : TEXCOORD0; };
[RootSignature(RS)]
Vertex vertex(uint i : SV_VertexID) {
    float2 uv = float2((i << 1) & 2, i & 2);
    Vertex v;
    v.position = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
    v.uv = source.xy + uv * source.zw;
    return v;
}
[RootSignature(RS)]
float4 pixel(Vertex v) : SV_Target { return image.SampleLevel(linearClamp, v.uv, 0); }
"#;

struct Compositor {
    root: ID3D12RootSignature,
    pipelines: [ID3D12PipelineState; 2],
    views: ID3D12DescriptorHeap,
    target: Option<(ID3D12Resource, u32, u32)>,
    list: host::HostList,
}

// SAFETY: D3D12 objects are free-threaded; the mutex serializes recording
// and waits for execution before any resource or descriptor is reused.
unsafe impl Send for Compositor {}

impl Compositor {
    fn new(host: &host::Host) -> windows::core::Result<Self> {
        let device = &host.device;
        let vs = host::compile(SHADER, s!("vertex"), s!("vs_5_1"))?;
        let ps = host::compile(SHADER, s!("pixel"), s!("ps_5_1"))?;
        // SAFETY: shader bytes contain the root signature; all descriptions
        // point at live data, and the borrowed root is never released by them.
        unsafe {
            let root = device.CreateRootSignature(0, &vs)?;
            let make = |blend: bool| {
                let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    VS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: vs.as_ptr().cast(),
                        BytecodeLength: vs.len(),
                    },
                    PS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: ps.as_ptr().cast(),
                        BytecodeLength: ps.len(),
                    },
                    SampleMask: u32::MAX,
                    RasterizerState: D3D12_RASTERIZER_DESC {
                        FillMode: D3D12_FILL_MODE_SOLID,
                        CullMode: D3D12_CULL_MODE_NONE,
                        DepthClipEnable: true.into(),
                        ..Default::default()
                    },
                    DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                        DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                        FrontFace: D3D12_DEPTH_STENCILOP_DESC {
                            StencilFailOp: D3D12_STENCIL_OP_KEEP,
                            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
                            StencilPassOp: D3D12_STENCIL_OP_KEEP,
                            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                        },
                        BackFace: D3D12_DEPTH_STENCILOP_DESC {
                            StencilFailOp: D3D12_STENCIL_OP_KEEP,
                            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
                            StencilPassOp: D3D12_STENCIL_OP_KEEP,
                            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                        },
                        ..Default::default()
                    },
                    PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                    NumRenderTargets: 1,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    ..Default::default()
                };
                desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
                desc.BlendState.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
                    BlendEnable: blend.into(),
                    SrcBlend: D3D12_BLEND_ONE,
                    DestBlend: D3D12_BLEND_INV_SRC_ALPHA,
                    BlendOp: D3D12_BLEND_OP_ADD,
                    SrcBlendAlpha: D3D12_BLEND_ONE,
                    DestBlendAlpha: D3D12_BLEND_INV_SRC_ALPHA,
                    BlendOpAlpha: D3D12_BLEND_OP_ADD,
                    LogicOp: D3D12_LOGIC_OP_NOOP,
                    RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                    ..Default::default()
                };
                device.CreateGraphicsPipelineState(&desc)
            };
            let pipelines = [make(false)?, make(true)?];
            let views = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 8,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })?;
            let list = host.new_list()?;
            list.close();
            Ok(Self {
                root,
                pipelines,
                views,
                target: None,
                list,
            })
        }
    }

    fn show(
        &mut self,
        host: &host::Host,
        layers: &[Layer],
        frame: u64,
    ) -> windows::core::Result<()> {
        if layers.is_empty() {
            return Ok(());
        }
        #[cfg(feature = "diagnostics")]
        if crate::capture::listed(frame) && std::env::var_os("VICTORY_DUMP_PLANES").is_some() {
            for (i, layer) in layers.iter().enumerate() {
                let desc = unsafe { layer.texture.GetDesc() };
                crate::capture::dump_one(host, &layer.texture, layer.state, frame, 100 + i,
                    std::ptr::null(), &desc);
            }
        }
        let extent = layers.iter().fold((0, 0), |(w, h), layer| {
            let d = unsafe { layer.texture.GetDesc() };
            // Destination placement is normalized to the output. Preserve
            // the sharpest plane (the UI can be larger than the scene).
            (w.max(d.Width as u32), h.max(d.Height))
        });
        let (width, height) = extent;
        if width == 0 || height == 0 || width > 16384 || height > 16384 || layers.len() > 8 {
            return Err(windows::core::Error::from_hresult(windows::core::HRESULT(
                0x80070057u32 as i32,
            )));
        }
        // SAFETY: our resources, descriptors and command list remain live
        // until the synchronous execute completes. Inputs are restored to
        // their tracked state without changing the title's state tracking.
        unsafe {
            if self
                .target
                .as_ref()
                .is_none_or(|(_, w, h)| (*w, *h) != extent)
            {
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width: width as u64,
                    Height: height,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                    ..Default::default()
                };
                let mut target = None;
                let clear = D3D12_CLEAR_VALUE {
                    Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                    Anonymous: D3D12_CLEAR_VALUE_0 {
                        Color: [0.0, 0.0, 0.0, 1.0],
                    },
                };
                host.device.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    Some(&clear),
                    &mut target,
                )?;
                self.target = Some((
                    target.ok_or_else(windows::core::Error::empty)?,
                    width,
                    height,
                ));
            }
            let target = &self.target.as_ref().unwrap().0;
            self.list.reset();
            let list = &self.list.list;
            let rtv = host.rtv(Some(target), None);
            list.ClearRenderTargetView(rtv, &[0.0, 0.0, 0.0, 1.0], None);
            list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.SetDescriptorHeaps(&[Some(self.views.clone())]);
            list.SetGraphicsRootSignature(&self.root);
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            let step = host
                .device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            for (i, layer) in layers.iter().enumerate() {
                let texture = &layer.texture;
                let d = texture.GetDesc();
                let source = layer
                    .source
                    .unwrap_or([0, 0, d.Width as i32, d.Height as i32]);
                let destination = layer.destination.unwrap_or([0.0, 0.0, 1.0, 1.0]);
                let uv = [
                    source[0] as f32 / d.Width as f32,
                    source[1] as f32 / d.Height as f32,
                    (source[2] - source[0]) as f32 / d.Width as f32,
                    (source[3] - source[1]) as f32 / d.Height as f32,
                ];
                let mut cpu = self.views.GetCPUDescriptorHandleForHeapStart();
                cpu.ptr += i * step as usize;
                let mut gpu = self.views.GetGPUDescriptorHandleForHeapStart();
                gpu.ptr += i as u64 * step as u64;
                let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT(crate::resource::target_format(d.Format.0 as u32) as i32),
                    ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2D: D3D12_TEX2D_SRV {
                            MipLevels: 1,
                            ..Default::default()
                        },
                    },
                };
                host.device
                    .CreateShaderResourceView(texture, Some(&view), cpu);
                let before = D3D12_RESOURCE_STATES(host::legal_state(texture, layer.state) as i32);
                let sampled = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
                list.ResourceBarrier(&[host::transition(texture, before, sampled)]);
                list.SetPipelineState(&self.pipelines[usize::from(i != 0)]);
                list.SetGraphicsRootDescriptorTable(0, gpu);
                list.SetGraphicsRoot32BitConstants(1, 4, uv.as_ptr().cast(), 0);
                list.RSSetViewports(&[placement(destination, width, height)]);
                list.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: width as i32,
                    bottom: height as i32,
                }]);
                list.DrawInstanced(3, 1, 0, 0);
                list.ResourceBarrier(&[host::transition(texture, sampled, before)]);
            }
            self.list.close();
            host.execute(&[&self.list]);
            host.present(target, D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32, frame);
        }
        Ok(())
    }
}

fn placement(rect: [f32; 4], width: u32, height: u32) -> D3D12_VIEWPORT {
    D3D12_VIEWPORT {
        TopLeftX: rect[0] * width as f32,
        TopLeftY: rect[1] * height as f32,
        Width: (rect[2] - rect[0]) * width as f32,
        Height: (rect[3] - rect[1]) * height as f32,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    }
}

pub(crate) fn show(layers: &[Layer], frame: u64) -> windows::core::Result<()> {
    static COMPOSITOR: OnceLock<Mutex<Option<Compositor>>> = OnceLock::new();
    let Some(host) = host::host() else {
        return Ok(());
    };
    let mut compositor = COMPOSITOR
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let result = (|| {
        if compositor.is_none() {
            *compositor = Some(Compositor::new(host)?);
        }
        compositor.as_mut().unwrap().show(host, layers, frame)
    })();
    if result.is_err() {
        host.log_messages();
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalized_placement_scales_to_the_output() {
        let full = placement([0.0, 0.0, 1.0, 1.0], 3840, 2160);
        assert_eq!(
            (full.TopLeftX, full.TopLeftY, full.Width, full.Height),
            (0.0, 0.0, 3840.0, 2160.0)
        );
        let inset = placement([0.25, 0.5, 0.75, 1.0], 3840, 2160);
        assert_eq!(
            (inset.TopLeftX, inset.TopLeftY, inset.Width, inset.Height),
            (960.0, 1080.0, 1920.0, 1080.0)
        );
    }

    /// An actual GPU readback: a smaller translucent UI plane covers only
    /// the right half of an opaque scene. No title files or snapshots.
    #[cfg(feature = "diagnostics")]
    #[test]
    #[ignore = "requires a D3D12 device and creates a presentation window"]
    fn two_planes_scale_place_and_blend_on_gpu() {
        let host = host::host().expect("D3D12 device");
        let solid = |size: u32, color: [f32; 4]| unsafe {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: size as u64,
                Height: size,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                ..Default::default()
            };
            let mut texture: Option<ID3D12Resource> = None;
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    None,
                    &mut texture,
                )
                .unwrap();
            let texture = texture.unwrap();
            let list = host.new_list().unwrap();
            list.list
                .ClearRenderTargetView(host.rtv(Some(&texture), None), &color, None);
            list.close();
            host.execute(&[&list]);
            Layer {
                texture,
                state: D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32,
                source: None,
                destination: None,
            }
        };
        let background = solid(2, [1.0, 0.0, 0.0, 1.0]);
        let mut overlay = solid(1, [0.0, 0.5, 0.0, 0.5]);
        overlay.destination = Some([0.5, 0.0, 1.0, 1.0]);
        let mut compositor = Compositor::new(host).unwrap();
        compositor.show(host, &[background, overlay], 3).unwrap();
        let texture = &compositor.target.as_ref().unwrap().0;
        let list = host.new_list().unwrap();
        let readback = unsafe {
            list.list.ResourceBarrier(&[host::transition(
                texture,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            crate::capture::read_back(&host.device, &list.list, texture).unwrap()
        };
        list.close();
        host.execute(&[&list]);
        let (buffer, footprint) = readback;
        let mut data = std::ptr::null_mut();
        unsafe {
            buffer.Map(0, None, Some(&mut data)).unwrap();
            let bytes = std::slice::from_raw_parts(
                data.cast::<u8>(),
                footprint.Footprint.RowPitch as usize * 2,
            );
            for row in 0..2 {
                let at = row * footprint.Footprint.RowPitch as usize;
                assert_eq!(&bytes[at..at + 4], &[255, 0, 0, 255]);
                let blended = &bytes[at + 4..at + 8];
                assert!(
                    (126..=129).contains(&blended[0]) && (126..=129).contains(&blended[1]),
                    "{blended:?}"
                );
                assert_eq!(&blended[2..], &[0, 255]);
            }
            buffer.Unmap(0, None);
        }
    }
}
