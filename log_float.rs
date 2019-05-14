use std::ops;
use num_traits::real::Real;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct LogFloat<T>(T);

impl<T: Copy + num_traits::Float> LogFloat<T> {
    pub fn from_value(n: T) -> LogFloat<T> {
        LogFloat(Real::ln(n))
    }

    pub fn from_ln(n: T) -> LogFloat<T> {
        LogFloat(n)
    }

    pub fn zero() -> LogFloat<T> {
        LogFloat(num_traits::Float::neg_infinity())
    }

    pub fn exp(&self) -> T {
        let LogFloat(n) = self;
        *n
    }

    pub fn to_value(&self) -> T {
        let LogFloat(n) = self;
        n.exp()
    }

    pub fn sum(v: &Vec<LogFloat<T>>) -> LogFloat<T> where
        T: std::cmp::PartialOrd + Real + std::iter::Sum
    {
        use std::cmp::Ordering;
        let x0: T = *v.iter().fold(None, |acc, LogFloat(x)| {
            match acc {
                Some(y) =>
                    match x.partial_cmp(y) {
                        Some(Ordering::Greater) => Some(x),
                        _ => Some(y)
                    },
                None => Some(x)
            }
        }).expect("LogFloat::sum: No maximum");
        let sum: T = v.iter().map(|LogFloat(x)| Real::exp_m1(*x - x0)).sum();
        let n: T = T::from(v.len()).unwrap();
        LogFloat(x0 + Real::ln_1p(n + sum))
    }
}

impl<T: Copy + ops::Add<Output=T>> ops::Mul<LogFloat<T>> for LogFloat<T> {
    type Output = LogFloat<T>;
    fn mul(self, rhs: LogFloat<T>) -> LogFloat<T> {
        let LogFloat(a) = self;
        let LogFloat(b) = rhs;
        LogFloat(a + b)
    }
}

impl<T: Copy + ops::Sub<Output=T>> ops::Div<LogFloat<T>> for LogFloat<T> {
    type Output = LogFloat<T>;
    fn div(self, rhs: LogFloat<T>) -> LogFloat<T> {
        let LogFloat(a) = self;
        let LogFloat(b) = rhs;
        LogFloat(a - b)
    }
}
