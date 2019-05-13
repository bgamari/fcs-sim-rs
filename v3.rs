extern crate std;

use std::f64;
use std::ops;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct V3<T> {
    pub x: T,
    pub y: T,
    pub z: T
}

impl<T> V3<T> {
    pub fn origin() -> V3<T> where T: Scalar {
        V3 { x: Scalar::zero(), y: Scalar::zero(), z: Scalar::zero() }
    }
}

pub trait Scalar {
    fn zero() -> Self;
}

impl Scalar for f64 {
    fn zero() -> f64 {
        0.0
    }
}

impl<T: ops::Add<Output=T>> ops::Add for V3<T> {
    type Output = V3<T>;
    fn add(self, rhs: V3<T>) -> V3<T> {
        V3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z
        }
    }
}

impl<T: Scalar + ops::Add<Output=T>> std::iter::Sum for V3<T> {
    fn sum<I: Iterator<Item=V3<T>>>(iter: I) -> V3<T> {
        iter.fold(V3::origin(), std::ops::Add::add)
    }
}

impl<T: Copy + ops::Mul<Output=T>> ops::Mul<T> for V3<T> {
    type Output = V3<T>;
    fn mul(self, rhs: T) -> V3<T> {
        V3 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl<T: Copy + ops::Div<Output=T>> ops::Div<T> for V3<T> {
    type Output = V3<T>;
    fn div(self, rhs: T) -> V3<T> {
        V3 {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}
