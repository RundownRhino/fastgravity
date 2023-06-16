use numpy::{
    ndarray::{Array, ArrayView, Dimension},
    PyArray2,
};
use pyo3::{exceptions::PyValueError, PyResult, Python};

use crate::{vec2::Vec2, F};

pub fn check_pos_array<'a, D: Dimension>(
    arr: &'a ArrayView<F, D>,
) -> PyResult<impl Iterator<Item = Vec2<F>> + 'a> {
    let shape = arr.shape();
    match shape {
        &[_n, 2] => Ok(arr.rows().into_iter().map(|row| Vec2 {
            x: row[0],
            y: row[1],
        })),
        [_n, _m] => Err(PyValueError::new_err(format!(
            "Array must be of shape (n,2) but was {:?}",
            shape
        ))),
        _ => Err(PyValueError::new_err(format!(
            "Array must be two-dimensional but was of shape {:?}",
            shape
        ))),
    }
}

pub fn to_pos_array(py: Python<'_>, positions: impl Iterator<Item = Vec2<F>>) -> &'_ PyArray2<F> {
    let arr = positions
        .flat_map(|v| [v.x, v.y].into_iter())
        .collect::<Vec<_>>();
    let n = arr.len() / 2;
    PyArray2::from_owned_array(py, Array::from_shape_vec([n, 2], arr).unwrap())
}
