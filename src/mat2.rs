use std::ops::{Add, Mul};

use crate::vec2::Vec2;

#[derive(Clone, Copy, Debug, Default)]
pub struct Mat2<T> {
    pub xx: T,
    pub xy: T,
    pub yx: T,
    pub yy: T,
}

impl<T> Mat2<T>
where
    T: Copy + Mul<T, Output = T> + Add<T, Output = T>,
{
    /// Evaluates v^T@self@v
    pub fn eval_quadratic(&self, v: Vec2<T>) -> T {
        let Vec2 { x, y } = v;
        self.xx * x * x + self.yy * y * y + (self.xy + self.yx) * x * y
    }

    pub fn matmul(&self, v: Vec2<T>) -> Vec2<T> {
        let Vec2 { x, y } = v;
        Vec2 {
            x: self.xx * x + self.xy * y,
            y: self.yx * x + self.yy * y,
        }
    }
}

impl<T> Mul<T> for Mat2<T>
where
    T: Copy + Mul<T, Output = T>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self {
            xx: self.xx * rhs,
            xy: self.xy * rhs,
            yx: self.yx * rhs,
            yy: self.yy * rhs,
        }
    }
}

impl<T> Add<Self> for Mat2<T>
where
    T: Copy + Add<T, Output = T>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            xx: self.xx + rhs.xx,
            xy: self.xy + rhs.xy,
            yx: self.yx + rhs.yx,
            yy: self.yy + rhs.yy,
        }
    }
}
