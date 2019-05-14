extern crate rand;
extern crate nalgebra as na;
extern crate num_traits;
extern crate rayon;
extern crate csv;
#[macro_use]
extern crate serde_derive;

mod v3;
mod log_float;

use rand::distributions::{Distribution};
use std::f64;
use v3::V3;
use num_traits::Num;
use log_float::LogFloat;

struct RandomWalk<'a, Rng> {
    rng: &'a mut Rng,
    sigma: f64, /// MSD per step
    pos: V3<f64>
}

impl<'a, Rng: rand::Rng> std::iter::Iterator for RandomWalk<'a, Rng> {
    type Item = V3<f64>;
    fn next(&mut self) -> Option<V3<f64>> {
        use rand::distributions::Normal;
        let r = Normal::new(0.0, self.sigma).sample(&mut self.rng);
        self.pos = self.pos + OnSphere.sample(&mut self.rng) * r;
        Some(self.pos)
    }
}

struct OnSphere;

impl Distribution<V3<f64>> for OnSphere {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> V3<f64> {
        use std::f64::consts::PI;
        use rand::distributions::Uniform;
        let theta: f64 = rng.sample(Uniform::new(0.0, 2.0*PI));
        let phi: f64 = (1.0 as f64 - 2.0 * rng.sample(Uniform::new(0.0, 1.0))).acos();
        V3 {
            x: phi.sin() * theta.cos(),
            y: phi.sin() * theta.sin(),
            z: phi.cos()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Box {
    half_box_size: V3<f64>
}

impl Box {
    fn new(box_size: V3<f64>) -> Box {
        Box {
            half_box_size: box_size / 2.0
        }
    }

    fn contains(&self, p: &V3<f64>) -> bool {
        let s = self.half_box_size;
        p.x.abs() < s.x
            && p.y.abs() < s.y
            && p.z.abs() < s.z
    }

    fn sample_point<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> V3<f64> {
        use rand::distributions::Uniform;
        let s = self.half_box_size;
        V3 {
            x: rng.sample(Uniform::new(-s.x, s.x)),
            y: rng.sample(Uniform::new(-s.y, s.y)),
            z: rng.sample(Uniform::new(-s.y, s.z)),
        }
    }
}

struct WalkThroughBox {
    sim_box: Box,
    sigma: f64
}

impl WalkThroughBox {
    fn sample<R: rand::Rng + Sized>(&self, rng: &mut R) -> Vec<V3<f64>> {
        let b = &self.sim_box;
        let p0 = b.sample_point(rng);
        let walk0: Vec<V3<f64>> = {
            let walk = RandomWalk { rng: rng, sigma: self.sigma, pos: p0 };
            walk.take_while(|p| b.contains(p)).collect()
        };
        let walk1: Vec<V3<f64>> = {
            let walk = RandomWalk { rng: rng, sigma: self.sigma, pos: p0 };
            walk.take_while(|p| b.contains(p)).collect()
        };
        walk0.into_iter().rev().chain(walk1.into_iter()).collect()
    }
}

fn beam_intensity(beam_size: V3<f64>, p: V3<f64>) -> LogFloat<f64> {
    let alpha: f64 =
          p.x.powi(2) / beam_size.x.powi(2)
        + p.y.powi(2) / beam_size.y.powi(2)
        + p.z.powi(2) / beam_size.z.powi(2);
    //(-2.0 * alpha).exp()
    LogFloat::from_ln(-2.0 * alpha)
}

fn log_space(min: f64, max: f64, n: usize) -> Vec<f64> {
    use num_traits::real::Real;
    assert!(min > 0.0);
    assert!(max > 0.0);
    let a: f64 = Real::ln(min);
    let b: f64 = Real::ln(max);
    let nn: f64 = n as f64;
    let dx: f64 = (b - a) / nn;
    (0..n).map(|i| {
        let ii = i as f64;
        Real::exp(a + ii * dx)
    }).collect()
}

fn write_vec<T, I>(path: &std::path::Path, mut v: I) -> std::io::Result<()> where
    I: Iterator<Item=T>,
    T: serde::Serialize
{
    let mut wtr = csv::WriterBuilder::new()
        .delimiter(b'\t')
        .from_path(path)?;

    v.try_for_each(|x| wtr.serialize(x))?;
    Ok(())
}

fn main() {
    use std::vec::Vec;
    use rand::FromEntropy;
    use rayon::prelude::*;
    use num_traits::real::Real;

    let diffusivity = 0.122; // nm^2 / ns
    let beam_size = V3 {x: 200.0, y: 200.0, z: 1000.0}; // nm
    let sim_box = Box::new(beam_size * 50.0);
    let step_t: f64 = 10e-9; // seconds
    //let n_steps: u64 = (100e-3 / step_t) as u64;
    let sample_idxs: Vec<_> = (0..128).collect();
    let max_tau: u64 = (10e-3 / step_t) as u64;
    let n_taus: u64 = 1000;
    let sigma = Real::sqrt(6.0 * diffusivity * step_t / 1e-9);

    let tau_ts: Vec<f64> = log_space(step_t as f64, max_tau as f64, n_taus as usize);
    let taus: Vec<usize> =
        tau_ts
        .iter()
        .map(|x| Real::round(*x) as usize)
        .collect();
    //println!("taus: {:?}\n", tau_ts);

    let results: Vec<Vec<LogFloat<f64>>> = sample_idxs.par_iter().map(|i| {
          let mut rng = rand::rngs::SmallRng::from_entropy();
          let walk = WalkThroughBox {
              sim_box: sim_box,
              sigma: sigma
          };
          //write_vec(std::path::Path::new("traj.txt"), &walk.sample(&mut rng)).unwrap();
          let steps: Vec<LogFloat<f64>> =
              walk
              .sample(&mut rng)
              .into_iter()
              .map(|x| beam_intensity(beam_size, x))
              // .take(n_steps as usize)
              .collect();
          //write_vec(std::path::Path::new("out.txt"), &steps).unwrap();
          let padded_steps: Vec<LogFloat<f64>> = pad_to_length(2*max_tau as usize, LogFloat::from_value(0.0), steps);
          println!("{} trajectory done", i);
          let corrs: Vec<LogFloat<f64>> =
              taus
              .par_iter()
              .map(|tau| correlate_log(max_tau as usize, *tau, &padded_steps))
              .collect();

          println!("{} done", i);
          let fname = format!("corr-{:04}", i);
          let path = std::path::Path::new(&fname);
          write_vec(path, tau_ts.iter().zip(&corrs));
          corrs
    }).collect();
}

fn norm_corr(corr: &Vec<LogFloat<f64>>) -> Vec<f64>
{
    let g0 = corr[0];
    corr.iter().map(|g| (*g / g0).to_value()).collect()
}

fn pad_to_length<T: Clone>(length: usize, pad: T, mut vec: Vec<T>) -> Vec<T> {
    if vec.len() < length {
        let mut padding = Vec::new();
        padding.resize(length - vec.len(), pad);
        vec.append(&mut padding)
    }
    vec
}

fn mean<N, T>(iter: T) -> N where
    N: Num + std::convert::From<u32>,
    T: Iterator<Item=N>
{
    use std::convert::From;
    let (accum, count) = iter.fold((From::from(0 as u32), 0), |(accum, count): (N, u32), x| (accum+x, count+1));
    accum / From::from(count)
}

fn correlate(max_tau: usize, tau: usize, vec: &Vec<f64>) -> f64 {
    assert!(vec.len() >= max_tau);
    let dot =
        vec
        .iter()
        .take(vec.len() - max_tau)
        .zip(vec.iter().skip(tau))
        .map(|(x,y)| x * y);
    mean(dot)
}

fn mean_log<T>(vec: &Vec<LogFloat<T>>) -> LogFloat<T> where
    T: std::iter::Sum + num_traits::Float + std::cmp::PartialOrd
{
    let n: LogFloat<T> = LogFloat::from_value(T::from(vec.len()).expect("mean_log: n too large"));
    LogFloat::sum(vec) / n
}

fn correlate_log<T>(max_tau: usize, tau: usize, vec: &Vec<LogFloat<T>>) -> LogFloat<T> where
    T: num_traits::Float + std::iter::Sum
{
    assert!(vec.len() >= max_tau);
    let dot: Vec<LogFloat<T>> =
        vec
        .iter()
        .take(vec.len() - max_tau)
        .zip(vec.iter().skip(tau))
        .map(|(x,y)| *x * *y)
        .collect();
    mean_log(&dot)
}
