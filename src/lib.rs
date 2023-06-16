mod mat2;
mod utils;
mod vec2;
use mat2::Mat2;
use numpy::{ndarray::Array1, PyArray1, PyArray2, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyValueError, prelude::*};
use utils::to_pos_array;
use vec2::Vec2;

use crate::utils::check_pos_array;

/// default value for θ. nodes with width/r<θ are considered far enough away to
/// use the approximate potential.
const DEFAULT_ACC: F = 0.3;
const G: F = -1.;

/// A Python module implemented in Rust.
#[pymodule]
fn fastgravity(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<GravitySystem>()?;
    Ok(())
}

#[pyclass]
struct GravitySystem {
    root: QuadNode,
}
#[pymethods]
impl GravitySystem {
    #[new]
    fn py_new(positions: PyReadonlyArrayDyn<F>, masses: PyReadonlyArrayDyn<F>) -> PyResult<Self> {
        let n = *positions.shape().first().unwrap_or(&0);
        if n != *masses.shape().first().unwrap_or(&0) {
            return Err(PyValueError::new_err(format!(
                "The sizes of the positions and masses arrays should be equal; were {} and {}",
                n,
                masses.len()
            )));
        }
        if n == 0 {
            return Err(PyValueError::new_err("The number of points can't be zero."));
        }
        if masses.shape() != [n] {
            return Err(PyValueError::new_err(format!(
                "The masses array should be 1d, got shape {:?}.",
                masses.shape(),
            )));
        }
        let positions = positions.as_array();
        let masses = masses.as_array();
        let vecs = check_pos_array(&positions)?;
        let pts = vecs
            .zip(masses.iter())
            .map(|(pos, m)| Body { pos, mass: *m })
            .collect();
        Ok(Self {
            root: tree_from_points(pts),
        })
    }

    #[pyo3(signature = (at_pos, use_quad=true, accuracy=DEFAULT_ACC))]
    fn evaluate_potential<'py>(
        &self,
        py: Python<'py>,
        at_pos: PyReadonlyArrayDyn<F>,
        use_quad: bool,
        accuracy: F,
    ) -> PyResult<&'py PyArray1<F>> {
        // TODO: rayon?
        let arr = at_pos.as_array();
        let vecs = check_pos_array(&arr)?;
        Ok(PyArray1::from_owned_array(
            py,
            Array1::from_vec(
                vecs.map(|v| self.root.potential_at(v, use_quad, accuracy))
                    .collect(),
            ),
        ))
    }

    #[pyo3(signature = (at_pos, use_quad=true, accuracy=DEFAULT_ACC))]
    fn evaluate_gravity<'py>(
        &self,
        py: Python<'py>,
        at_pos: PyReadonlyArrayDyn<F>,
        use_quad: bool,
        accuracy: F,
    ) -> PyResult<&'py PyArray2<F>> {
        // TODO: rayon?
        let arr = at_pos.as_array();
        let vecs = check_pos_array(&arr)?;
        Ok(to_pos_array(
            py,
            vecs.map(|v| self.root.gravity_at(v, use_quad, accuracy)),
        ))
    }
}

type F = f64;

#[derive(Clone, Copy)]
struct Body {
    mass: F,
    pos: Vec2<F>,
}
trait Quad: Sized {
    fn com(&self) -> (F, Vec2<F>);
    fn quadrupole(&self) -> Mat2<F>;
    fn potential_at(&self, pos: Vec2<F>, _use_quad: bool, accuracy: F) -> F;
    fn gravity_at(&self, pos: Vec2<F>, _use_quad: bool, accuracy: F) -> Vec2<F>;
}

enum QuadNode {
    Leaf(QuadLeaf),
    Interior(QuadInterior),
}

struct QuadLeaf {
    body: Body,
}
struct QuadInterior {
    /// yx: sw, se, nw, ne
    /// yx: --, -+, +-, ++
    children: [Option<Box<QuadNode>>; 4],
    com: Vec2<F>,
    total_mass: F,
    quadrupole: Mat2<F>,

    extent_x: (F, F),
    extent_y: (F, F),
}
impl QuadInterior {
    fn new(
        sw: Option<QuadNode>,
        se: Option<QuadNode>,
        nw: Option<QuadNode>,
        ne: Option<QuadNode>,
        extent_x: (F, F),
        extent_y: (F, F),
    ) -> Self {
        let (total_mass, com) = {
            let mut mass = 0.;
            let mut com = Vec2::zero();
            for child in [&sw, &se, &nw, &ne].into_iter().flatten() {
                let (child_m, child_com) = child.com();
                mass += child_m;
                com = com + child_com * child_m;
            }
            assert!(mass != 0.); // sanity check
            com = com / mass;
            (mass, com)
        };
        let mut quadrupole = Mat2::default();
        for child in [&sw, &se, &nw, &ne].into_iter().flatten() {
            let (child_m, child_com) = child.com();
            let child_q = child.quadrupole();
            quadrupole = quadrupole + child_q + to_quadrup_tensor(com - child_com) * child_m;
        }
        Self {
            children: [
                sw.map(Box::new),
                se.map(Box::new),
                nw.map(Box::new),
                ne.map(Box::new),
            ],
            total_mass,
            com,
            quadrupole,
            extent_x,
            extent_y,
        }
    }

    fn width(&self) -> F {
        F::hypot(
            self.extent_x.1 - self.extent_x.0,
            self.extent_y.1 - self.extent_y.0,
        )
    }

    fn some_children(&self) -> impl Iterator<Item = &QuadNode> {
        self.children
            .iter()
            .filter_map(|x| x.as_ref().map(|n| n.as_ref()))
    }
}

fn make_node(pts: Vec<Body>, extent_x: (F, F), extent_y: (F, F)) -> Option<QuadNode> {
    if pts.is_empty() {
        None
    } else if pts.len() == 1 {
        Some(QuadNode::Leaf(QuadLeaf {
            body: *pts.first().unwrap(),
        }))
    } else {
        let (l, r) = extent_x;
        let (b, t) = extent_y;
        let div_x = (l + r) / 2.;
        let div_y = (b + t) / 2.;
        let sw = make_node(
            pts.iter()
                .copied()
                .filter(|x| x.pos.x < div_x && x.pos.y < div_y)
                .collect(),
            (l, div_x),
            (b, div_y),
        );
        let se = make_node(
            pts.iter()
                .copied()
                .filter(|x| x.pos.x >= div_x && x.pos.y < div_y)
                .collect(),
            (div_x, r),
            (b, div_y),
        );
        let nw = make_node(
            pts.iter()
                .copied()
                .filter(|x| x.pos.x < div_x && x.pos.y >= div_y)
                .collect(),
            (l, div_x),
            (div_y, t),
        );
        let ne = make_node(
            pts.iter()
                .copied()
                .filter(|x| x.pos.x >= div_x && x.pos.y >= div_y)
                .collect(),
            (div_x, r),
            (div_y, t),
        );
        Some(QuadNode::Interior(QuadInterior::new(
            sw, se, nw, ne, extent_x, extent_y,
        )))
    }
}

fn tree_from_points(pts: Vec<Body>) -> QuadNode {
    assert!(!pts.is_empty());
    let extent_x = (
        pts.iter().map(|b| b.pos.x).min_by(F::total_cmp).unwrap(),
        pts.iter().map(|b| b.pos.x).max_by(F::total_cmp).unwrap(),
    );
    let extent_y = (
        pts.iter().map(|b| b.pos.y).min_by(F::total_cmp).unwrap(),
        pts.iter().map(|b| b.pos.y).max_by(F::total_cmp).unwrap(),
    );
    make_node(pts, extent_x, extent_y).unwrap()
}

/// Computes Q_{αβ} = 2 r_α r_β - δ_{αβ} r^2
/// It's traceless and symmetric, so has only 2 independent elements
fn to_quadrup_tensor(r: Vec2<F>) -> Mat2<F> {
    let diag = r.x * r.x - r.y * r.y;
    let cross = 2. * r.x * r.y;
    Mat2 {
        xx: diag,
        yy: -diag,
        xy: cross,
        yx: cross,
    }
}

impl Quad for QuadLeaf {
    fn com(&self) -> (F, Vec2<F>) {
        (self.body.mass, self.body.pos)
    }

    fn quadrupole(&self) -> Mat2<F> {
        Mat2::default()
    }

    fn potential_at(&self, pos: Vec2<F>, _use_quad: bool, _accuracy: F) -> F {
        let dist = (pos - self.body.pos).norm();
        if dist == 0. {
            0.
        } else {
            G * self.body.mass / dist
        }
    }

    fn gravity_at(&self, pos: Vec2<F>, _use_quad: bool, _accuracy: F) -> Vec2<F> {
        let r = pos - self.body.pos;
        let dist = r.norm();
        if dist == 0. {
            Default::default()
        } else {
            let e = r / dist;
            e * (G * self.body.mass / dist.powi(2))
        }
    }
}

impl Quad for QuadInterior {
    fn com(&self) -> (F, Vec2<F>) {
        (self.total_mass, self.com)
    }

    fn quadrupole(&self) -> Mat2<F> {
        self.quadrupole
    }

    fn potential_at(&self, pos: Vec2<F>, use_quad: bool, accuracy: F) -> F {
        let (mass, com) = self.com();
        let r = pos - com;
        let dist = r.norm();
        if dist > 0. && self.width() / dist < accuracy {
            let scalar_part = mass / dist;
            let mut total = scalar_part;
            if use_quad {
                let e = r / dist;
                let quadrupole_part = self.quadrupole().eval_quadratic(e) / (2. * dist.powi(3));
                total += quadrupole_part;
            }
            G * total
        } else {
            // exact calculation
            self.some_children()
                .map(|x| x.potential_at(pos, use_quad, accuracy))
                .sum::<F>()
        }
    }

    fn gravity_at(&self, pos: Vec2<F>, use_quad: bool, accuracy: F) -> Vec2<F> {
        let (mass, com) = self.com();
        let r = pos - com;
        let dist = r.norm();
        if dist > 0. && self.width() / dist < accuracy {
            let e = r / dist;
            let scalar_part = e * (mass / dist.powi(2));
            let mut total = scalar_part;
            if use_quad {
                let dist4 = dist.powi(4);
                let quadrupole_part_1 = e * (self.quadrupole().eval_quadratic(e) * 2.5 / dist4);
                let quadrupole_part_2 = -self.quadrupole().matmul(e) / dist4;
                total = total + quadrupole_part_1 + quadrupole_part_2;
            }
            total * G
        } else {
            // exact calculation
            self.some_children()
                .map(|x| x.gravity_at(pos, use_quad, accuracy))
                .sum()
        }
    }
}

impl Quad for QuadNode {
    fn com(&self) -> (F, Vec2<F>) {
        match self {
            QuadNode::Leaf(x) => x.com(),
            QuadNode::Interior(x) => x.com(),
        }
    }

    fn quadrupole(&self) -> Mat2<F> {
        match self {
            QuadNode::Leaf(x) => x.quadrupole(),
            QuadNode::Interior(x) => x.quadrupole(),
        }
    }

    fn potential_at(&self, pos: Vec2<F>, use_quad: bool, accuracy: F) -> F {
        match self {
            QuadNode::Leaf(x) => x.potential_at(pos, use_quad, accuracy),
            QuadNode::Interior(x) => x.potential_at(pos, use_quad, accuracy),
        }
    }

    fn gravity_at(&self, pos: Vec2<F>, use_quad: bool, accuracy: F) -> Vec2<F> {
        match self {
            QuadNode::Leaf(x) => x.gravity_at(pos, use_quad, accuracy),
            QuadNode::Interior(x) => x.gravity_at(pos, use_quad, accuracy),
        }
    }
}
