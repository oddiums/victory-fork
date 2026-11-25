//! Shared, bounded GPU storage for compute-generated mesh exports. All
//! host work uses one queue; UAV barriers order reuse between mesh draws.
use crate::host::{HostList, host, transition};
use std::sync::{Mutex, OnceLock};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

pub(crate) const OUTPUT_BYTES: u64 = 128 << 20;

/// Argument generation leaves this buffer in INDIRECT_ARGUMENT, including
/// when a later binding cannot be made. Always restore it for the next use.
pub(crate) struct ArgumentUse<'a>(pub &'a HostList, pub &'a ID3D12Resource);
impl Drop for ArgumentUse<'_> {
    fn drop(&mut self) {
        unsafe {
            self.0.list.ResourceBarrier(&[transition(
                self.1,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
    }
}

struct Buffers {
    output: ID3D12Resource,
    arguments: ID3D12Resource,
}
unsafe impl Send for Buffers {}
unsafe impl Sync for Buffers {}

pub(crate) fn buffers(list: &HostList) -> Option<(ID3D12Resource, ID3D12Resource)> {
    static BUFFERS: OnceLock<Mutex<Option<Buffers>>> = OnceLock::new();
    let mut buffers = BUFFERS
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if buffers.is_none() {
        *buffers = Some(Buffers {
            output: host()?.uav_buffer(OUTPUT_BYTES)?,
            arguments: host()?.uav_buffer(64 << 10)?,
        });
    }
    let b = buffers.as_ref()?;
    list.keep(&b.output);
    list.keep(&b.arguments);
    Some((b.output.clone(), b.arguments.clone()))
}

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=7, b0), SRV(t0), SRV(t1), UAV(u0)"
cbuffer K : register(b0) { uint offset; uint capacity; uint primitives; uint record; uint hasCount; uint countOffset; uint chunks; }
ByteAddressBuffer source : register(t0);
ByteAddressBuffer counter : register(t1);
RWByteAddressBuffer output : register(u0);
[RootSignature(RS)]
[numthreads(64,1,1)]
void main(uint id : SV_DispatchThreadID) {
    if (id >= chunks) return;
    uint n = 0;
    if (hasCount == 0 || record < counter.Load(countOffset)) {
        uint3 g = source.Load3(offset);
        if (g.y != 0 && g.z != 0) n = min(g.x, 65535u);
    }
    uint base = id * capacity;
    n = min(n > base ? n - base : 0, capacity);
    output.Store4(id * 32, uint4(n, 1, 1, 0));
    output.Store4(id * 32 + 16, uint4(n * primitives * 3, 1, 0, 0));
}
"#;
struct Arguments {
    root: ID3D12RootSignature,
    pipeline: ID3D12PipelineState,
}
unsafe impl Send for Arguments {}
unsafe impl Sync for Arguments {}

/// Converts one bounded mesh record to paired compute and draw records,
/// one pair per page of output storage. Zero counts remain zero.
pub(crate) fn arguments(
    list: &HostList,
    source: (&ID3D12Resource, u64),
    counter: Option<(&ID3D12Resource, u64)>,
    destination: &ID3D12Resource,
    capacity: u32,
    primitives: u32,
    record: u32,
) -> Option<u32> {
    static MADE: OnceLock<Option<Arguments>> = OnceLock::new();
    let a = MADE
        .get_or_init(|| {
            let device = &host()?.device;
            let code = crate::host::compile(SOURCE, s!("main"), s!("cs_5_1")).ok()?;
            unsafe {
                let root: ID3D12RootSignature = device.CreateRootSignature(0, &code).ok()?;
                let pipeline = device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: code.as_ptr().cast(),
                            BytecodeLength: code.len(),
                        },
                        ..Default::default()
                    })
                    .ok()?;
                Some(Arguments { root, pipeline })
            }
        })
        .as_ref()?;
    let chunks = 65535u32.div_ceil(capacity);
    let constants = [
        u32::try_from(source.1).ok()?,
        capacity,
        primitives,
        record,
        u32::from(counter.is_some()),
        counter.map_or(0, |(_, offset)| offset as u32),
        chunks,
    ];
    let count = counter.map_or(source.0, |(buffer, _)| buffer);
    unsafe {
        let l = &list.list;
        l.ResourceBarrier(&[
            transition(
                source.0,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            ),
            transition(
                destination,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ),
        ]);
        l.SetComputeRootSignature(&a.root);
        l.SetPipelineState(&a.pipeline);
        l.SetComputeRoot32BitConstants(0, 7, constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, source.0.GetGPUVirtualAddress());
        l.SetComputeRootShaderResourceView(2, count.GetGPUVirtualAddress());
        l.SetComputeRootUnorderedAccessView(3, destination.GetGPUVirtualAddress());
        l.Dispatch(chunks.div_ceil(64), 1, 1);
        l.ResourceBarrier(&[
            transition(
                destination,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
            transition(
                source.0,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
        ]);
    }
    list.keep(&a.root);
    list.keep(&a.pipeline);
    Some(chunks)
}

pub(crate) fn barrier(list: &HostList, output: &ID3D12Resource) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: unsafe { std::mem::transmute_copy(output) },
            }),
        },
        ..Default::default()
    };
    unsafe {
        list.list.ResourceBarrier(&[barrier]);
    }
}

// One group per mesh group, matching the producer's indirect dispatch.
// Reject zero-area and out-of-range indices before vertex pulling. This
// saves the VS and (where used) GS work for padded/culled primitives.
const COMPACT: &str = r#"
#define RS "RootConstants(num32BitConstants=5,b0), SRV(t0), UAV(u0), UAV(u1)"
cbuffer Layout : register(b0) { uint primitives; uint vertices; uint stride; uint vertexBytes; uint primitiveStride; }
ByteAddressBuffer exports : register(t0);
RWByteAddressBuffer triangles : register(u0);
RWByteAddressBuffer draw : register(u1);
groupshared uint flags[64];
groupshared uint first;
[RootSignature(RS)]
[numthreads(64,1,1)]
void main(uint3 g : SV_GroupID, uint t : SV_GroupIndex) {
    uint group = g.x;
    [loop] for (uint block = 0; block < primitives; block += 64) {
        uint p = block + t;
        bool valid = false;
        if (p < primitives) {
            uint3 ix = exports.Load3(group * stride + vertexBytes + p * primitiveStride);
            valid = all(ix < vertices) && ix.x != ix.y && ix.y != ix.z && ix.x != ix.z;
        }
        flags[t] = valid ? 1 : 0;
        GroupMemoryBarrierWithGroupSync();
        uint rank = 0;
        for (uint i = 0; i < t; i++) rank += flags[i];
        if (t == 63) {
            uint count = rank + flags[t];
            uint was = 0;
            if (count != 0) draw.InterlockedAdd(0,count * 3,was);
            first = was / 3;
        }
        GroupMemoryBarrierWithGroupSync();
        if (valid) triangles.Store((first + rank) * 4,group * primitives + p);
        GroupMemoryBarrierWithGroupSync();
    }
}
"#;

pub(crate) struct Compact {
    root: ID3D12RootSignature,
    pipeline: ID3D12PipelineState,
    pub triangles: ID3D12Resource,
    pub draw: ID3D12Resource,
    reset: ID3D12Resource,
}
// SAFETY: immutable D3D12 objects. One host queue orders shared scratch use.
unsafe impl Send for Compact {}
unsafe impl Sync for Compact {}

pub(crate) fn compactor() -> Option<&'static Compact> {
    static PASS: OnceLock<Option<Compact>> = OnceLock::new();
    PASS.get_or_init(|| {
        let h = host()?;
        let code = crate::host::compile(COMPACT, s!("main"), s!("cs_5_1")).ok()?;
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
            let (reset, data) = h.upload_buffer(16)?;
            std::ptr::copy_nonoverlapping([0u32, 1, 0, 0].as_ptr(), data.cast(), 4);
            Some(Compact {
                root,
                pipeline,
                triangles: h.uav_buffer(OUTPUT_BYTES / 4)?,
                draw: h.uav_buffer(16)?,
                reset,
            })
        }
    })
    .as_ref()
}

impl Compact {
    /// Exports are already in NON_PIXEL_SHADER_RESOURCE. Scratch begins in
    /// COMMON and is left as SRV/INDIRECT until `done` follows the draw.
    pub(crate) fn record(
        &self,
        list: &HostList,
        exports: &ID3D12Resource,
        layout: &victory_shader::emit::MeshLayout,
        groups: u32,
        indirect: Option<(&ID3D12CommandSignature, &ID3D12Resource, u64)>,
    ) {
        let k = [
            layout.max_primitives,
            layout.max_vertices,
            layout.group_stride,
            layout.max_vertices * layout.vertex_stride,
            layout.primitive_stride,
        ];
        self.run(list, exports, k, groups, indirect);
    }

    fn run(
        &self,
        list: &HostList,
        exports: &ID3D12Resource,
        k: [u32; 5],
        groups: u32,
        indirect: Option<(&ID3D12CommandSignature, &ID3D12Resource, u64)>,
    ) {
        unsafe {
            let l = &list.list;
            l.ResourceBarrier(&[
                transition(
                    &self.draw,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ),
                transition(
                    &self.triangles,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                ),
            ]);
            l.CopyBufferRegion(&self.draw, 0, &self.reset, 0, 16);
            l.ResourceBarrier(&[transition(
                &self.draw,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
            l.SetComputeRootSignature(&self.root);
            l.SetPipelineState(&self.pipeline);
            l.SetComputeRoot32BitConstants(0, 5, k.as_ptr().cast(), 0);
            l.SetComputeRootShaderResourceView(1, exports.GetGPUVirtualAddress());
            l.SetComputeRootUnorderedAccessView(2, self.triangles.GetGPUVirtualAddress());
            l.SetComputeRootUnorderedAccessView(3, self.draw.GetGPUVirtualAddress());
            if let Some((signature, args, offset)) = indirect {
                l.ExecuteIndirect(signature, 1, args, offset, None, 0);
            } else {
                l.Dispatch(groups, 1, 1);
            }
            l.ResourceBarrier(&[
                transition(
                    &self.draw,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                ),
                transition(
                    &self.triangles,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                ),
            ]);
        }
        for resource in [&self.draw, &self.triangles, &self.reset] {
            list.keep(resource);
        }
        list.keep(&self.root);
        list.keep(&self.pipeline);
    }

    pub(crate) fn done(&self, list: &HostList) {
        unsafe {
            list.list.ResourceBarrier(&[
                transition(
                    &self.draw,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COMMON,
                ),
                transition(
                    &self.triangles,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                ),
            ]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compact_triangles_keep_live_ids_and_reset_empty_draws_on_gpu() {
        let Some(h) = host() else { return };
        let pass = compactor().expect("triangle compaction pipeline");
        let exports = h.uav_buffer(224).unwrap();
        let readback = h.read_back_buffer(32).unwrap();
        let (upload, data) = h.upload_buffer(224).unwrap();
        let mut words = [0u32; 56];
        for (group, prim, indices) in [
            (0, 0, [0, 1, 2]),
            (0, 2, [0, 1, 3]),
            (0, 3, [2, 1, 0]),
            (1, 1, [0, 1, 2]),
            (1, 2, [1, 1, 2]),
        ] {
            let at = group * 28 + 12 + prim * 4;
            words[at..at + 3].copy_from_slice(&indices);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(words.as_ptr(), data.cast(), words.len());
        }
        let list = h.new_list().unwrap();
        unsafe {
            list.list.ResourceBarrier(&[transition(
                &exports,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
            list.list.CopyBufferRegion(&exports, 0, &upload, 0, 224);
            list.list.ResourceBarrier(&[transition(
                &exports,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            )]);
        }
        list.close();
        h.execute(&[&list]);
        for groups in [2, 0] {
            let list = h.new_list().unwrap();
            pass.run(&list, &exports, [4, 3, 112, 48, 16], groups, None);
            unsafe {
                list.list.ResourceBarrier(&[
                    transition(
                        &pass.draw,
                        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    ),
                    transition(
                        &pass.triangles,
                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    ),
                ]);
                list.list.CopyBufferRegion(&readback, 0, &pass.draw, 0, 16);
                list.list
                    .CopyBufferRegion(&readback, 16, &pass.triangles, 0, 12);
                list.list.ResourceBarrier(&[
                    transition(
                        &pass.draw,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_COMMON,
                    ),
                    transition(
                        &pass.triangles,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_COMMON,
                    ),
                ]);
            }
            list.close();
            h.execute(&[&list]);
            unsafe {
                let mut ptr = std::ptr::null_mut();
                readback.Map(0, None, Some(&mut ptr)).unwrap();
                let out = std::slice::from_raw_parts(ptr.cast::<u32>(), 7);
                assert_eq!(&out[..4], &[if groups == 0 { 0 } else { 9 }, 1, 0, 0]);
                if groups != 0 {
                    let mut ids = out[4..7].to_vec();
                    ids.sort_unstable();
                    assert_eq!(ids, [0, 3, 5]);
                }
                readback.Unmap(0, None);
            }
        }
        assert!(unsafe { h.device.GetDeviceRemovedReason() }.is_ok());
    }
    #[test]
    fn indirect_mesh_pages_preserve_all_groups_and_zero_inactive_records() {
        let Some(host) = host() else { return };
        let source = host.uav_buffer(12).unwrap();
        let destination = host.uav_buffer(128).unwrap();
        let readback = host.read_back_buffer(128).unwrap();
        let (upload, data) = host.upload_buffer(16).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping([65535u32, 1, 1, 0].as_ptr(), data.cast(), 4);
        }
        let list = host.new_list().unwrap();
        unsafe {
            list.list.ResourceBarrier(&[transition(
                &source,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
            list.list.CopyBufferRegion(&source, 0, &upload, 0, 12);
            list.list.ResourceBarrier(&[transition(
                &source,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            )]);
        }
        list.close();
        host.execute(&[&list]);
        for inactive in [false, true] {
            let list = host.new_list().unwrap();
            let count = inactive.then_some((&upload, 12));
            assert_eq!(
                arguments(&list, (&source, 0), count, &destination, 16384, 7, 0),
                Some(4)
            );
            unsafe {
                list.list.ResourceBarrier(&[transition(
                    &destination,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                list.list
                    .CopyBufferRegion(&readback, 0, &destination, 0, 128);
                list.list.ResourceBarrier(&[transition(
                    &destination,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
            list.close();
            host.execute(&[&list]);
            let mut data = std::ptr::null_mut();
            unsafe {
                readback.Map(0, None, Some(&mut data)).unwrap();
                let words = std::slice::from_raw_parts(data.cast::<u32>(), 32);
                for chunk in 0..4 {
                    let n = if inactive {
                        0
                    } else if chunk == 3 {
                        16383
                    } else {
                        16384
                    };
                    assert_eq!(&words[chunk * 8..chunk * 8 + 3], &[n, 1, 1]);
                    assert_eq!(&words[chunk * 8 + 4..chunk * 8 + 8], &[n * 21, 1, 0, 0]);
                }
                readback.Unmap(0, None);
            }
        }
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }
}
