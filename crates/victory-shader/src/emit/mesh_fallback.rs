//! Mesh export emulation: run the console program as compute, then pull
//! its vertices and per-primitive attributes through a regular vertex stage.
use super::{
    HostShader,
    bitcode::{Attr, Bin, Term, V},
    container::SigElement,
    shader::Shader,
};
use crate::{ShaderError, native::HwStage};
use std::cell::{Cell, RefCell};

/// A private raw UAV, shared by the compute producer and vertex consumer.
pub const SLOT_MESH_OUTPUT: u8 = 252;
pub const SLOT_MESH_TRIANGLES: u8 = 251;

#[derive(Debug, Clone)]
pub struct MeshLayout {
    pub max_vertices: u32,
    pub max_primitives: u32,
    pub vertex_stride: u32,
    pub primitive_stride: u32,
    pub group_stride: u32,
    vertices: Vec<SigElement>,
    primitives: Vec<SigElement>,
}
pub struct MeshFallback {
    pub compute: HostShader,
    pub vertex: HostShader,
    pub layout: MeshLayout,
}
thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(false) };
    static LAYOUT: RefCell<Option<MeshLayout>> = const { RefCell::new(None) };
}
pub(super) fn enabled() -> bool {
    ENABLED.with(Cell::get)
}

pub fn translate_mesh_fallback(container: &[u8]) -> Result<MeshFallback, ShaderError> {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            ENABLED.with(|v| v.set(self.0));
        }
    }
    let _reset = Reset(ENABLED.with(|v| v.replace(true)));
    LAYOUT.with(|v| v.borrow_mut().take());
    let mut compute = super::translate(container)?;
    let layout = LAYOUT
        .with(|v| v.borrow_mut().take())
        .ok_or(ShaderError::Malformed("not a mesh shader"))?;
    compute.stage = HwStage::Cs;
    let vertex = mesh_fallback_vertex(&layout, None)?;
    Ok(MeshFallback {
        compute,
        vertex,
        layout,
    })
}

fn layout(s: &Shader) -> MeshLayout {
    let mesh = s.mesh.expect("mesh export");
    let vertex_stride = (s.outputs.len() as u32 * 16).max(16);
    let primitive_stride = 16 + s.prims.len() as u32 * 16;
    MeshLayout {
        max_vertices: mesh.max_vertices,
        max_primitives: mesh.max_primitives,
        vertex_stride,
        primitive_stride,
        group_stride: mesh.max_vertices * vertex_stride + mesh.max_primitives * primitive_stride,
        vertices: s.outputs.clone(),
        primitives: s.prims.clone(),
    }
}

pub(super) fn export(s: &mut Shader, b: usize, opcode: u32, args: &[V]) -> V {
    if opcode == 168 {
        return s.m.i32(0);
    }
    let layout = layout(s);
    let zero = s.m.i32(0);
    let group = s.op(b, "groupId.i32", Attr::ReadNone, s.t.i32, 94, &[zero]);
    let stride = s.m.i32(layout.group_stride);
    let base = s.m.bin(b, Bin::Mul, group, stride);
    let vertices_bytes = layout.max_vertices * layout.vertex_stride;
    let (index, stride, extra, component) = match opcode {
        169 => (args[0], layout.primitive_stride, vertices_bytes, None),
        171 => (
            *args.last().unwrap(),
            layout.vertex_stride,
            0,
            Some((args[0], args[2])),
        ),
        172 => (
            *args.last().unwrap(),
            layout.primitive_stride,
            vertices_bytes + 16,
            Some((args[0], args[2])),
        ),
        _ => unreachable!(),
    };
    let stride = s.m.i32(stride);
    let index = s.m.bin(b, Bin::Mul, index, stride);
    let extra = s.m.i32(extra);
    let mut at = s.m.bin(b, Bin::Add, base, index);
    at = s.m.bin(b, Bin::Add, at, extra);
    if let Some((id, col)) = component {
        let sixteen = s.m.i32(16);
        let id = s.m.bin(b, Bin::Mul, id, sixteen);
        let i32t = s.t.i32;
        let col = s.m.cast(b, super::bitcode::Cast::ZExt, col, i32t);
        let four = s.m.i32(4);
        let col = s.m.bin(b, Bin::Mul, col, four);
        at = s.m.bin(b, Bin::Add, at, id);
        at = s.m.bin(b, Bin::Add, at, col);
    }
    let h = s.root_uav_handle(SLOT_MESH_OUTPUT);
    let undef = s.m.undef(s.t.i32);
    let mut values = [undef; 4];
    let mask = if opcode == 169 {
        values[..3].copy_from_slice(&args[1..4]);
        7
    } else {
        let x = args[3];
        values[0] = if s.m.type_of(x) == s.t.f32 {
            s.m.cast(b, super::bitcode::Cast::Bitcast, x, s.t.i32)
        } else {
            x
        };
        1
    };
    let mask = s.m.i8(mask);
    let align = s.m.i32(4);
    s.op(
        b,
        "rawBufferStore.i32",
        Attr::NoUnwind,
        s.t.void,
        140,
        &[
            h, at, undef, values[0], values[1], values[2], values[3], mask, align,
        ],
    )
}

pub(super) fn finish(s: &mut Shader) {
    if !enabled() || s.mesh.is_none() {
        return;
    }
    LAYOUT.with(|v| *v.borrow_mut() = Some(layout(s)));
    s.inputs.clear();
    s.outputs.clear();
    s.prims.clear();
    s.mesh = None;
}

fn load(s: &mut Shader, at: V) -> V {
    load_slot(s, at, SLOT_MESH_OUTPUT)
}
fn load_slot(s: &mut Shader, at: V, slot: u8) -> V {
    let b = s.entry;
    let h = s.root_srv_handle(slot);
    let ty = s.m.named_struct("dx.types.ResRet.i32", vec![s.t.i32; 5]);
    let undef = s.m.undef(s.t.i32);
    let mask = s.m.i8(1);
    let align = s.m.i32(4);
    let r = s.op(
        b,
        "rawBufferLoad.i32",
        Attr::ReadOnly,
        ty,
        139,
        &[h, at, undef, mask, align],
    );
    s.m.extract(b, r, 0)
}

pub fn mesh_fallback_vertex(
    l: &MeshLayout,
    pixel: Option<&[u8]>,
) -> Result<HostShader, ShaderError> {
    vertex(l, pixel, false)
}

pub fn mesh_fallback_vertex_compact(
    l: &MeshLayout,
    pixel: Option<&[u8]>,
) -> Result<HostShader, ShaderError> {
    vertex(l, pixel, true)
}

fn vertex(l: &MeshLayout, pixel: Option<&[u8]>, compact: bool) -> Result<HostShader, ShaderError> {
    let inputs = if let Some(pixel) = pixel {
        let c = crate::dxbc::Container::parse(pixel)?;
        c.part(crate::dxbc::FourCC::ISG1)
            .map(|p| super::signature(p.data))
            .transpose()?
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    let mut s = Shader::new(false);
    s.inputs = vec![SigElement {
        name: "SV_VertexID".into(),
        index: 0,
        kind: 1,
        comp: 5,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 1,
        usage: 1,
        half: false,
    }];
    let mut sources = Vec::new();
    let mut free_row = if pixel.is_some() {
        &inputs
    } else {
        &l.vertices
    }
    .iter()
    .filter_map(|e| e.row)
    .max()
    .map_or(0, |r| r + 1);
    for (primitive, elements) in [(false, &l.vertices), (true, &l.primitives)] {
        for (id, element) in elements.iter().enumerate() {
            let mut o = element.clone();
            if pixel.is_some() {
                let matched = inputs.iter().find(|i| {
                    i.kind == o.kind
                        && i.index == o.index
                        && (o.kind != 0 || i.name.eq_ignore_ascii_case(&o.name))
                });
                if let Some(input) = matched {
                    if let Some(row) = input.row {
                        o.row = Some(row);
                        o.col = input.col;
                    } else {
                        o.row = Some(free_row);
                        o.col = 0;
                        free_row += 1;
                    }
                    o.cols = o.cols.min(input.cols);
                } else if matches!(o.kind, 3..=7) {
                    o.row = Some(free_row);
                    o.col = 0;
                    free_row += 1;
                } else {
                    continue;
                }
            } else if primitive {
                o.row = Some(free_row);
                o.col = 0;
                free_row += 1;
            }
            o.half = false;
            o.usage = (1 << o.cols) - 1;
            s.outputs.push(o);
            sources.push((primitive, id));
        }
    }
    let b = s.entry;
    let (zero, col, undef) = (s.m.i32(0), s.m.i8(0), s.m.undef(s.t.i32));
    let vid = s.op(
        b,
        "loadInput.i32",
        Attr::ReadNone,
        s.t.i32,
        4,
        &[zero, zero, col, undef],
    );
    let three = s.m.i32(3);
    let mut primitive = s.m.bin(b, Bin::UDiv, vid, three);
    if compact {
        let four = s.m.i32(4);
        let at = s.m.bin(b, Bin::Mul, primitive, four);
        primitive = load_slot(&mut s, at, SLOT_MESH_TRIANGLES);
    }
    let corner = s.m.bin(b, Bin::URem, vid, three);
    let n = s.m.i32(l.max_primitives);
    let group = s.m.bin(b, Bin::UDiv, primitive, n);
    let primitive = s.m.bin(b, Bin::URem, primitive, n);
    let stride = s.m.i32(l.group_stride);
    let group_at = s.m.bin(b, Bin::Mul, group, stride);
    let stride = s.m.i32(l.primitive_stride);
    let primitive_at = s.m.bin(b, Bin::Mul, primitive, stride);
    let primitive_at = s.m.bin(b, Bin::Add, group_at, primitive_at);
    let k = s.m.i32(l.max_vertices * l.vertex_stride);
    let primitive_at = s.m.bin(b, Bin::Add, primitive_at, k);
    let four = s.m.i32(4);
    let corner = s.m.bin(b, Bin::Mul, corner, four);
    let index_at = s.m.bin(b, Bin::Add, primitive_at, corner);
    let vertex = load(&mut s, index_at);
    let stride = s.m.i32(l.vertex_stride);
    let vertex_at = s.m.bin(b, Bin::Mul, vertex, stride);
    let vertex_at = s.m.bin(b, Bin::Add, group_at, vertex_at);
    for i in 0..s.outputs.len() {
        let (primitive, element) = sources[i];
        for c in 0..s.outputs[i].cols {
            let offset =
                s.m.i32(element as u32 * 16 + u32::from(c) * 4 + if primitive { 16 } else { 0 });
            let at = s.m.bin(
                b,
                Bin::Add,
                if primitive { primitive_at } else { vertex_at },
                offset,
            );
            let x = load(&mut s, at);
            let float = s.outputs[i].comp == 9;
            let x = if float {
                s.m.cast(b, super::bitcode::Cast::Bitcast, x, s.t.f32)
            } else {
                x
            };
            let args = [s.m.i32(i as u32), zero, s.m.i8(c), x];
            s.op(
                b,
                if float {
                    "storeOutput.f32"
                } else {
                    "storeOutput.i32"
                },
                Attr::NoUnwind,
                s.t.void,
                5,
                &args,
            );
        }
    }
    s.m.terminate(b, Term::Ret);
    Ok(HostShader {
        stage: HwStage::Vs,
        dxil: s.finish(),
        user_data: vec![],
        samplers: vec![],
        static_samplers: vec![],
        root_cbvs: vec![],
        root_srvs: if compact {
            vec![SLOT_MESH_TRIANGLES, SLOT_MESH_OUTPUT]
        } else {
            vec![SLOT_MESH_OUTPUT]
        },
        root_uavs: vec![],
        root_indexed_views: vec![],
    })
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    #[test]
    fn emulated_exports_and_vertex_pulling_validate() {
        ENABLED.with(|e| e.set(true));
        let mut s = Shader::new(false);
        s.mesh = Some(super::super::shader::Mesh {
            max_vertices: 3,
            max_primitives: 1,
            topology: 2,
        });
        s.threads = Some([1, 1, 1]);
        s.outputs = vec![SigElement {
            name: "SV_Position".into(),
            index: 0,
            kind: 3,
            comp: 9,
            interp: 4,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        }];
        s.prims = vec![SigElement {
            name: "COLOR".into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp: 1,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        }];
        let b = s.entry;
        for vertex in 0..3u32 {
            for col in 0..4u8 {
                let values = [
                    [-1.0f32, -1.0, 0.5, 1.0],
                    [-1.0, 3.0, 0.5, 1.0],
                    [3.0, -1.0, 0.5, 1.0],
                ];
                let args = [
                    s.m.i32(0),
                    s.m.i32(0),
                    s.m.i8(col),
                    s.m.f32(values[vertex as usize][col as usize].to_bits()),
                    s.m.i32(vertex),
                ];
                s.op(
                    b,
                    "storeVertexOutput.f32",
                    Attr::NoUnwind,
                    s.t.void,
                    171,
                    &args,
                );
            }
        }
        let args = [s.m.i32(0), s.m.i32(0), s.m.i32(1), s.m.i32(2)];
        s.op(b, "emitIndices", Attr::NoUnwind, s.t.void, 169, &args);
        for col in 0..4u8 {
            let args = [
                s.m.i32(0),
                s.m.i32(0),
                s.m.i8(col),
                s.m.f32(1.0f32.to_bits()),
                s.m.i32(0),
            ];
            s.op(
                b,
                "storePrimitiveOutput.f32",
                Attr::NoUnwind,
                s.t.void,
                172,
                &args,
            );
        }
        s.m.terminate(b, Term::Ret);
        let mut compute = s.finish();
        ENABLED.with(|e| e.set(false));
        let layout = LAYOUT.with(|l| l.borrow_mut().take()).unwrap();
        let mut vertex = mesh_fallback_vertex(&layout, None).unwrap().dxil;
        let path = super::super::validator::Validator::default_path();
        if let Ok(mut validator) = super::super::validator::Validator::load(path.as_deref()) {
            validator.validate(&mut compute).expect("compute exports");
            validator.validate(&mut vertex).expect("vertex pulling");
        }
        // A pixel stage can pack a two-component primitive attribute into
        // a different row and column than the mesh signatures. The vertex
        // replacement must link to those exact registers.
        let mut ps = Shader::new(true);
        ps.inputs = vec![SigElement {
            name: "COLOR".into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp: 1,
            row: Some(3),
            col: 2,
            cols: 2,
            usage: 3,
            half: false,
        }];
        ps.outputs = vec![SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        }];
        let b = ps.entry;
        for col in 0..4u8 {
            let z = ps.m.i32(0);
            let c = ps.m.i8(col % 2);
            let undef = ps.m.undef(ps.t.i32);
            let x = ps.op(
                b,
                "loadInput.f32",
                Attr::ReadNone,
                ps.t.f32,
                4,
                &[z, z, c, undef],
            );
            let c = ps.m.i8(col);
            ps.op(
                b,
                "storeOutput.f32",
                Attr::NoUnwind,
                ps.t.void,
                5,
                &[z, z, c, x],
            );
        }
        ps.m.terminate(b, Term::Ret);
        let pixel = HostShader {
            stage: HwStage::Ps,
            dxil: ps.finish(),
            user_data: vec![],
            samplers: vec![],
            static_samplers: vec![],
            root_cbvs: vec![],
            root_srvs: vec![],
            root_uavs: vec![],
            root_indexed_views: vec![],
        };
        let mut linked_vertex = mesh_fallback_vertex(&layout, Some(&pixel.dxil)).unwrap();
        let mut compact_vertex = mesh_fallback_vertex_compact(&layout, Some(&pixel.dxil)).unwrap();
        if let Ok(mut validator) = super::super::validator::Validator::load(path.as_deref()) {
            validator
                .validate(&mut linked_vertex.dxil)
                .expect("packed vertex outputs");
            validator
                .validate(&mut compact_vertex.dxil)
                .expect("compacted vertex pulling");
        }
        if let Ok(host) = super::super::host::Host::new(false) {
            host.pipeline(&compact_vertex, &pixel)
                .expect("compacted ordinary graphics pipeline");
            host.pipeline(&linked_vertex, &pixel)
                .expect("ordinary graphics pipeline on a non-mesh device");
            let cs = HostShader {
                stage: HwStage::Cs,
                dxil: compute,
                root_srvs: vec![],
                root_uavs: vec![SLOT_MESH_OUTPUT],
                ..pixel.clone()
            };
            let (root, pipeline) = host.compute_pipeline(&cs).unwrap();
            use windows::Win32::Graphics::Direct3D12::*;
            use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
            use windows::core::Interface;
            unsafe {
                let size = (u64::from(layout.group_stride) * 3).max(256);
                let make = |heap, flags, state| {
                    let mut buffer: Option<ID3D12Resource> = None;
                    host.device
                        .CreateCommittedResource(
                            &D3D12_HEAP_PROPERTIES {
                                Type: heap,
                                ..Default::default()
                            },
                            D3D12_HEAP_FLAG_NONE,
                            &D3D12_RESOURCE_DESC {
                                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                                Width: size,
                                Height: 1,
                                DepthOrArraySize: 1,
                                MipLevels: 1,
                                SampleDesc: DXGI_SAMPLE_DESC {
                                    Count: 1,
                                    Quality: 0,
                                },
                                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                                Flags: flags,
                                ..Default::default()
                            },
                            state,
                            None,
                            &mut buffer,
                        )
                        .unwrap();
                    buffer.unwrap()
                };
                let output = make(
                    D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
                let readback = make(
                    D3D12_HEAP_TYPE_READBACK,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                );
                let allocator: ID3D12CommandAllocator = host
                    .device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .unwrap();
                let list: ID3D12GraphicsCommandList = host
                    .device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, &pipeline)
                    .unwrap();
                list.SetComputeRootSignature(&root);
                list.SetComputeRootUnorderedAccessView(1, output.GetGPUVirtualAddress());
                list.Dispatch(3, 1, 1);
                list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        Transition: std::mem::ManuallyDrop::new(
                            D3D12_RESOURCE_TRANSITION_BARRIER {
                                pResource: std::mem::transmute_copy(&output),
                                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                            },
                        ),
                    },
                    ..Default::default()
                }]);
                list.CopyBufferRegion(&readback, 0, &output, 0, size);
                list.Close().unwrap();
                let queue: ID3D12CommandQueue = host
                    .device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .unwrap();
                queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                let fence: ID3D12Fence = host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
                queue.Signal(&fence, 1).unwrap();
                let start = std::time::Instant::now();
                while fence.GetCompletedValue() < 1 {
                    assert!(start.elapsed().as_secs() < 10);
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                let mut data = std::ptr::null_mut();
                readback.Map(0, None, Some(&mut data)).unwrap();
                let words = std::slice::from_raw_parts(data.cast::<u32>(), size as usize / 4);
                for group in 0..3 {
                    let at = group * layout.group_stride as usize / 4;
                    assert_eq!(
                        &words[at..at + 4],
                        &[-1.0f32, -1.0, 0.5, 1.0].map(f32::to_bits)
                    );
                    assert_eq!(&words[at + 12..at + 15], &[0, 1, 2]);
                    assert_eq!(&words[at + 16..at + 20], &[1.0f32.to_bits(); 4]);
                }
                readback.Unmap(0, None);
                // Draw the generated vertices through the ordinary vertex
                // pipeline and inspect an actual rasterized pixel.
                let (graphics_root, graphics) = host.pipeline(&linked_vertex, &pixel).unwrap();
                let mut target: Option<ID3D12Resource> = None;
                host.device.CreateCommittedResource(&D3D12_HEAP_PROPERTIES {Type:D3D12_HEAP_TYPE_DEFAULT,..Default::default()},
                    D3D12_HEAP_FLAG_NONE,&D3D12_RESOURCE_DESC {Dimension:D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width:1,Height:1,DepthOrArraySize:1,MipLevels:1,Format:windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT,
                    SampleDesc:DXGI_SAMPLE_DESC{Count:1,Quality:0},Flags:D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,..Default::default()},
                    D3D12_RESOURCE_STATE_RENDER_TARGET,None,&mut target).unwrap();
                let target = target.unwrap();
                let rtv: ID3D12DescriptorHeap = host
                    .device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                        NumDescriptors: 1,
                        ..Default::default()
                    })
                    .unwrap();
                let view = rtv.GetCPUDescriptorHandleForHeapStart();
                host.device.CreateRenderTargetView(&target, None, view);
                let draw: ID3D12GraphicsCommandList = host
                    .device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, &graphics)
                    .unwrap();
                let moved = |resource: &ID3D12Resource, before, after| D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        Transition: std::mem::ManuallyDrop::new(
                            D3D12_RESOURCE_TRANSITION_BARRIER {
                                pResource: std::mem::transmute_copy(resource),
                                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                StateBefore: before,
                                StateAfter: after,
                            },
                        ),
                    },
                    ..Default::default()
                };
                draw.ResourceBarrier(&[moved(
                    &output,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                )]);
                draw.ClearRenderTargetView(view, &[0.0; 4], None);
                draw.OMSetRenderTargets(1, Some(&view), false, None);
                draw.SetGraphicsRootSignature(&graphics_root);
                draw.SetGraphicsRootShaderResourceView(2, output.GetGPUVirtualAddress());
                draw.RSSetViewports(&[D3D12_VIEWPORT {
                    Width: 1.0,
                    Height: 1.0,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                    ..Default::default()
                }]);
                draw.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                    left: 0,
                    top: 0,
                    right: 1,
                    bottom: 1,
                }]);
                draw.IASetPrimitiveTopology(
                    windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                );
                draw.DrawInstanced(9, 1, 0, 0);
                draw.ResourceBarrier(&[moved(
                    &target,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                let to = D3D12_TEXTURE_COPY_LOCATION{pResource:std::mem::transmute_copy(&readback),Type:D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous:D3D12_TEXTURE_COPY_LOCATION_0{PlacedFootprint:D3D12_PLACED_SUBRESOURCE_FOOTPRINT{Offset:0,Footprint:D3D12_SUBRESOURCE_FOOTPRINT {
                    Format:windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32G32B32A32_FLOAT,Width:1,Height:1,Depth:1,RowPitch:256}}}};
                let from = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::transmute_copy(&target),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        SubresourceIndex: 0,
                    },
                };
                draw.CopyTextureRegion(&to, 0, 0, 0, &from, None);
                draw.Close().unwrap();
                queue.ExecuteCommandLists(&[Some(draw.cast().unwrap())]);
                queue.Signal(&fence, 2).unwrap();
                let start = std::time::Instant::now();
                while fence.GetCompletedValue() < 2 {
                    assert!(start.elapsed().as_secs() < 10);
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                readback.Map(0, None, Some(&mut data)).unwrap();
                assert_eq!(std::slice::from_raw_parts(data.cast::<f32>(), 4), &[1.0; 4]);
                readback.Unmap(0, None);
            }
        }
    }
}
