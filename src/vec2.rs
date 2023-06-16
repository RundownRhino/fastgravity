#![allow(dead_code)]
use std::{
    iter::Sum,
    ops::{Add, Div, Mul, Neg, Sub},
};

use crate::F;

#[derive(Clone, Copy, Debug, Default)]
pub struct Vec2<T>
where
    T: Copy,
{
    pub x: T,
    pub y: T,
}
impl<T: Add<Output = T> + Copy> Add for Vec2<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Vec2 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}
impl<T: Sub<Output = T> + Copy> Sub for Vec2<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Vec2 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}
impl<T, S> Mul<S> for Vec2<T>
where
    T: Mul<S, Output = T> + Copy,
    S: Copy,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: S) -> Self::Output {
        Vec2 {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}
impl<T, S, R> Div<S> for Vec2<T>
where
    T: Div<S, Output = R> + Copy,
    S: Copy,
    R: Copy,
{
    type Output = Vec2<R>;

    #[inline]
    fn div(self, rhs: S) -> Self::Output {
        Vec2 {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}
impl<T: Neg<Output = T> + Copy> Neg for Vec2<T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Vec2 {
            x: -self.x,
            y: -self.y,
        }
    }
}
#[allow(dead_code)]
impl<T: Copy> Vec2<T> {
    #[inline]
    pub fn new(x: T, y: T) -> Self {
        Vec2 { x, y }
    }

    /// A vector of Default values for the type.
    #[inline]
    pub fn zero() -> Self
    where
        T: Default,
    {
        Self {
            x: Default::default(),
            y: Default::default(),
        }
    }

    #[inline]
    pub fn dot<H, R>(self, other: Vec2<H>) -> R
    where
        T: Mul<H, Output = R>,
        R: Add<R, Output = R>,
        H: Copy,
    {
        self.x * other.x + self.y * other.y
    }

    #[inline]
    pub fn cross<H, R>(self, other: Vec2<H>) -> R
    where
        T: Mul<H, Output = R>,
        R: Sub<R, Output = R>,
        H: Copy,
    {
        self.x * other.y - self.y * other.x
    }

    /// Squared length
    #[inline]
    pub fn sq_len<R, F>(self) -> F
    where
        T: Mul<T, Output = R>,
        R: Add<R, Output = F>,
    {
        self.x * self.x + self.y * self.y
    }

    /// Vector length (Euclidean norm).
    #[inline]
    pub fn norm<R, F>(self) -> F
    where
        T: Mul<T, Output = R>,
        R: Add<R, Output = F>,
        F: Sqrt,
    {
        self.sq_len().sqrt()
    }

    #[inline]
    pub fn as_tuple(&self) -> (T, T) {
        (self.x, self.y)
    }
}

impl<T> Vec2<T>
where
    T: Copy + std::ops::Div<Output = F>,
{
    /// Normalize vector by dividing by the norm.
    #[inline]
    pub fn normalized<R, F, D>(self) -> Vec2<D>
    where
        T: Mul<T, Output = R> + Div<F, Output = D>,
        R: Add<R, Output = F>,
        F: Copy + Sqrt,
        D: Copy,
    {
        self / self.norm()
    }
}

impl Vec2<f64> {
    /// Angle from the positive x axis, counterclockwise. From -pi to pi.
    #[inline]
    pub fn angle_of(&self) -> f64 {
        self.y.atan2(self.x)
    }

    /// Counterclockwise from self to other. From -pi to pi.
    #[inline]
    pub fn angle_to(&self, other: &Self) -> f64 {
        let mult = self.norm() * other.norm();
        let cos = self.dot(*other) / mult;
        let sin = self.cross(*other) / mult;
        f64::atan2(sin, cos)
    }
}
// Copy of previous one:
impl Vec2<f32> {
    /// Angle from the positive x axis, counterclockwise. From -pi to pi.
    #[inline]
    pub fn angle_of(&self) -> f32 {
        self.y.atan2(self.x)
    }

    /// Counterclockwise from self to other. From -pi to pi.
    #[inline]
    pub fn angle_to(&self, other: &Self) -> f32 {
        let mult = self.norm() * other.norm();
        let cos = self.dot(*other) / mult;
        let sin = self.cross(*other) / mult;
        f32::atan2(sin, cos)
    }
}
pub trait Rotate<T, G> {
    /// Rotate counterclockwise by an angle in radians.
    fn rotate(self, angle: G) -> Self
    where
        T: Mul<G, Output = T> + Add<T, Output = T> + Sub<T, Output = T>,
        G: Angle + Copy;
}
impl<T, G> Rotate<T, G> for Vec2<T>
where
    T: Copy + Mul<G, Output = T> + Add<T, Output = T> + Sub<T, Output = T>,
    G: Copy + Angle,
{
    #[inline]
    fn rotate(self, angle: G) -> Self {
        let cos = angle.cos();
        let sin = angle.sin();
        Vec2 {
            x: self.x * cos - self.y * sin,
            y: self.y * cos + self.x * sin,
        }
    }
}
pub trait CastTo<H>
where
    H: Copy,
{
    fn cast_to(self) -> Vec2<H>;
}
impl<T, H> CastTo<H> for Vec2<T>
where
    T: Copy,
    H: Copy + From<T>,
{
    fn cast_to(self) -> Vec2<H> {
        Vec2 {
            x: self.x.into(),
            y: self.y.into(),
        }
    }
}

pub trait Angle: Copy {
    fn cos(self) -> Self;
    fn sin(self) -> Self;
    fn sin_cos(self) -> (Self, Self);
}
impl Angle for f32 {
    #[inline]
    fn cos(self) -> Self {
        self.cos()
    }

    #[inline]
    fn sin(self) -> Self {
        self.sin()
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        self.sin_cos()
    }
}
impl Angle for f64 {
    #[inline]
    fn cos(self) -> Self {
        self.cos()
    }

    #[inline]
    fn sin(self) -> Self {
        self.sin()
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        self.sin_cos()
    }
}
pub trait Sqrt {
    fn sqrt(self) -> Self;
}
impl Sqrt for f32 {
    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
}
impl Sqrt for f64 {
    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
}

impl<T> Sum for Vec2<T>
where
    T: Copy + Add<T, Output = T> + Default,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(Self::add).unwrap_or_default()
    }
}
