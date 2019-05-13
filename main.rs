extern crate rand;
extern crate nalgebra as na;
extern crate num_traits;
extern crate rayon;
mod v3;

use rand::distributions::{Distribution};
use std::f64;
use v3::V3;
use num_traits::Num;

struct RandomWalk<'a, Rng> {
    rng: &'a mut Rng,
    diffusivity: f64,
    pos: V3<f64>
}

impl<'a, Rng: rand::Rng> std::iter::Iterator for RandomWalk<'a, Rng> {
    type Item = V3<f64>;
    fn next(&mut self) -> Option<V3<f64>> {
        use rand::distributions::Normal;
        let r = Normal::new(0.0, self.diffusivity).sample(&mut self.rng);
        self.pos = self.pos + OnSphere.sample(&mut self.rng) * r;
        Some(self.pos)
    }
}

struct OnSphere;

impl Distribution<V3<f64>> for OnSphere {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> V3<f64> {
        use std::f64::consts::PI;
        use rand::distributions::Uniform;
        let phi = rng.sample(Uniform::new(0.0, PI));
        let theta = rng.sample(Uniform::new(-PI, PI));
        V3 {
            x: theta.sin() * phi.cos(),
            y: theta.sin() * phi.sin(),
            z: theta.cos()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Box {
    box_size: V3<f64>
}

impl Box {
    fn contains(&self, p: &V3<f64>) -> bool {
        let s = self.box_size;
        p.x >= 0.0 && p.x < s.x
            && p.y >= 0.0 && p.y < s.y
            && p.z >= 0.0 && p.z < s.z
    }
}

struct PointInBox(Box);

impl Distribution<V3<f64>> for PointInBox {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> V3<f64> {
        use rand::distributions::Uniform;
        let PointInBox(b) = self;
        V3 {
            x: rng.sample(Uniform::new(0.0, b.box_size.x)),
            y: rng.sample(Uniform::new(0.0, b.box_size.y)),
            z: rng.sample(Uniform::new(0.0, b.box_size.z)),
        }
    }
}

struct WalkThroughBox {
    sim_box: Box,
    diffusivity: f64
}

impl WalkThroughBox {
    fn sample<R: rand::Rng + Sized>(&self, rng: &mut R) -> Vec<V3<f64>> {
        let b = &self.sim_box;
        let p0 = rng.sample(PointInBox(*b));
        let walk0: Vec<V3<f64>> = { 
            let walk = RandomWalk { rng: rng, diffusivity: self.diffusivity, pos: p0 };
            walk.take_while(|p| b.contains(p)).collect()
        };
        let walk1: Vec<V3<f64>> = { 
            let walk = RandomWalk { rng: rng, diffusivity: self.diffusivity, pos: p0 };
            walk.take_while(|p| b.contains(p)).collect()
        };
        walk0.into_iter().rev().chain(walk1.into_iter()).collect()
    }
}

fn beam_intensity(beam_size: V3<f64>, p: V3<f64>) -> f64 {
    let alpha: f64 =
          beam_size.x.powi(2) / p.x.powi(2)
        + beam_size.y.powi(2) / p.y.powi(2)
        + beam_size.z.powi(2) / p.z.powi(2);
    (-alpha / 2.0).exp()
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

fn write_vec<T>(path: &std::path::Path, v: &Vec<T>) -> std::io::Result<()> where
    T: std::fmt::Display
{
    use std::io::prelude::*;
    let mut file = std::fs::File::create(path)?;
    v.iter().try_for_each(|x| {
        write!(file, "{}\n", x)
    })?;
    Ok(())
}

fn main() {
    use std::vec::Vec;
    use rand::FromEntropy;
    use rayon::prelude::*;
    use num_traits::real::Real;

    let diffusivity = 1.1e-3;
    let beam_size = V3 {x:1.0, y:1.0, z:10.0};
    let sim_box = Box {box_size: beam_size * 50.0};
    let step_t: f64 = 1e-9; // nanoseconds
    let n_steps: u64 = (1e-1 / step_t) as u64;
    let sample_idxs: Vec<_> = (0..2).collect();
    let max_tau: u64 = (100e-3 / step_t) as u64;
    let n_taus: u64 = 1000;

    let tau_ts: Vec<f64> = log_space(step_t as f64, max_tau as f64, n_taus as usize);
    let taus: Vec<usize> =
        tau_ts
        .iter()
        .map(|x| Real::round(*x) as usize)
        .collect();
    println!("taus: {:?}\n", tau_ts);

    let results: Vec<Vec<f64>> = sample_idxs.par_iter().map(move |_i| {
          let mut rng = rand::rngs::SmallRng::from_entropy();
          let walk = WalkThroughBox {
              sim_box: sim_box,
              diffusivity: diffusivity
          };
          let steps: Vec<f64> = 
              walk
              .sample(&mut rng)
              .into_iter()
              .map(|x| beam_intensity(beam_size, x))
              .take(n_steps as usize)
              .collect();
          //write_vec(std::path::Path::new("out.txt"), &steps).unwrap();
          let corrs: Vec<f64> =
              taus
              .par_iter()
              .map(|tau| correlate(max_tau as usize, *tau, &steps))
              .collect();
          corrs
    }).collect();
    println!("hello {:?}", results);

    results.iter().enumerate().try_for_each(|(i, corrs)| {
        let fname = format!("corr-{}", i);
        let path = std::path::Path::new(&fname);
        write_vec(path, corrs)
    }).unwrap();
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
    mean(vec.iter().take(vec.len() - max_tau).zip(vec.iter().skip(tau)).map(|(x,y)| x*y))
}
