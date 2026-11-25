//! Barycentric inputs on older GPUs, carried through a geometry stage as
//! ordinary interpolants and flat copies of each triangle's attributes.
use super::{
    HostShader,
    bitcode::{Attr, Term, V},
    container::SigElement,
    shader::{Geometry, Shader},
};
use crate::{
    ShaderError,
    dxbc::{Container, FourCC},
};
use std::cell::Cell;
thread_local! { static ACTIVE: Cell<bool> = const { Cell::new(false) }; }
pub(super) fn enabled() -> bool {
    ACTIVE.with(Cell::get)
}
pub fn translate_bary_fallback(bytes: &[u8]) -> Result<HostShader, ShaderError> {
    with_fallback(|| super::translate(bytes))
}
pub(super) fn with_fallback<T>(f: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            ACTIVE.with(|v| v.set(self.0));
        }
    }
    let _reset = Reset(ACTIVE.with(|v| v.replace(true)));
    f()
}

pub(super) fn input(s: &mut Shader, mut e: SigElement) -> usize {
    if let Some(i) = s
        .inputs
        .iter()
        .position(|x| x.name == e.name && x.index == e.index)
    {
        return i;
    }
    e.row = Some(
        s.inputs
            .iter()
            .filter_map(|i| i.row)
            .max()
            .map_or(0, |r| r + 1),
    );
    e.col = 0;
    e.usage = 0;
    e.half = false;
    s.inputs.push(e);
    s.inputs.len() - 1
}
fn virtual_input(e: &SigElement) -> bool {
    matches!(e.name.as_str(), "VICTORY_BARY" | "VICTORY_VERTEX")
}
pub(super) struct Extra {
    output: usize,
    source: Option<(usize, usize)>,
}

pub(super) fn augment(s: &mut Shader, ps: &[SigElement]) -> Result<Vec<Extra>, ShaderError> {
    let mut extras = Vec::new();
    let mut spare = ps.iter().filter_map(|e| e.row).max().map_or(0, |r| r + 1);
    let first = ps
        .iter()
        .filter(|e| virtual_input(e))
        .filter_map(|e| e.row)
        .min();
    for e in &mut s.outputs {
        if e.row.zip(first).is_some_and(|(r, first)| r >= first) {
            e.row = Some(spare);
            spare += 1;
        }
    }
    for e in ps.iter().filter(|e| virtual_input(e)) {
        let source = if e.name == "VICTORY_VERTEX" {
            let original = ps
                .get(e.index as usize / 3)
                .ok_or(ShaderError::Malformed("barycentric source"))?;
            let id = s
                .inputs
                .iter()
                .position(|i| {
                    i.kind == original.kind
                        && i.index == original.index
                        && i.name.eq_ignore_ascii_case(&original.name)
                })
                .ok_or(ShaderError::Malformed(
                    "barycentric attribute missing from vertex stage",
                ))?;
            Some((id, e.index as usize % 3))
        } else {
            None
        };
        extras.push(Extra {
            output: s.outputs.len(),
            source,
        });
        s.outputs.push(SigElement {
            usage: (1 << e.cols) - 1,
            ..e.clone()
        });
    }
    if s.outputs.iter().filter_map(|e| e.row).any(|r| r >= 32) {
        return Err(ShaderError::Unsupported {
            offset: 0,
            what: "barycentric emulation exceeds vertex attribute registers",
        });
    }
    Ok(extras)
}

pub(super) fn emit_extras(s: &mut Shader, extras: &[Extra], values: &[Vec<Vec<V>>], vertex: usize) {
    let b = s.entry;
    for extra in extras {
        for col in 0..s.outputs[extra.output].cols {
            let x = if let Some((input, corner)) = extra.source {
                values[input][col as usize][vertex / 3 * 3 + corner]
            } else {
                s.m.f32(if col as usize == vertex % 3 {
                    1.0f32.to_bits()
                } else {
                    0
                })
            };
            let args = [s.m.i32(extra.output as u32), s.m.i32(0), s.m.i8(col), x];
            s.op(
                b,
                if s.outputs[extra.output].comp == 9 {
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
}

pub fn barycentric_gs(vs: &[u8], ps: &[u8]) -> Result<Option<Vec<u8>>, ShaderError> {
    let sig = |bytes: &[u8], kind| -> Result<Vec<SigElement>, ShaderError> {
        let c = Container::parse(bytes)?;
        c.part(kind)
            .map(|p| super::signature(p.data))
            .transpose()
            .map(|v| v.unwrap_or_default())
    };
    let inputs = sig(ps, FourCC::ISG1)?;
    if !inputs.iter().any(virtual_input) {
        return Ok(None);
    }
    let mut outputs = sig(vs, FourCC::OSG1)?;
    for e in &mut outputs { e.usage = (1 << e.cols) - 1; }
    let mut s = Shader::new(false);
    s.geometry = Some(Geometry { max_vertices: 3 });
    s.inputs = outputs.clone();
    s.outputs = outputs;
    let extras = augment(&mut s, &inputs)?;
    let b = s.entry;
    let mut values = Vec::new();
    for i in 0..s.inputs.len() {
        let mut columns = Vec::new();
        for c in 0..s.inputs[i].cols {
            let mut vertices = Vec::new();
            for v in 0..3 {
                let args = [s.m.i32(i as u32), s.m.i32(0), s.m.i8(c), s.m.i32(v)];
                vertices.push(s.op(
                    b,
                    if s.inputs[i].comp == 9 {
                        "loadInput.f32"
                    } else {
                        "loadInput.i32"
                    },
                    Attr::ReadNone,
                    if s.inputs[i].comp == 9 {
                        s.t.f32
                    } else {
                        s.t.i32
                    },
                    4,
                    &args,
                ));
            }
            columns.push(vertices);
        }
        values.push(columns);
    }
    let stream = s.m.i8(0);
    for vertex in 0..3 {
        for i in 0..s.inputs.len() {
            for c in 0..s.inputs[i].cols {
                let args = [
                    s.m.i32(i as u32),
                    s.m.i32(0),
                    s.m.i8(c),
                    values[i][c as usize][vertex],
                ];
                s.op(
                    b,
                    if s.outputs[i].comp == 9 {
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
        emit_extras(&mut s, &extras, &values, vertex);
        s.op(b, "emitStream", Attr::NoUnwind, s.t.void, 97, &[stream]);
    }
    s.op(b, "cutStream", Attr::NoUnwind, s.t.void, 98, &[stream]);
    s.m.terminate(b, Term::Ret);
    Ok(Some(s.finish()))
}
