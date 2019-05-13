extern crate rand;
extern crate nalgebra as na;
extern crate num_traits;
extern crate rayon;
mod v3;

use rand::distributions::{Distribution};
use std::f64;
use v3::V3;
use num_traits::Num;

struct RandomWalk<Rng> {
    rng: Rng,
    diffusivity: f64,
    pos: V3<f64>
}

impl<Rng: rand::Rng> std::iter::Iterator for RandomWalk<Rng> {
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

fn beam_intensity(beam_size: V3<f64>, p: V3<f64>) -> f64 {
    let alpha: f64 =
          beam_size.x.powi(2) / p.x.powi(2)
        + beam_size.y.powi(2) / p.y.powi(2)
        + beam_size.z.powi(2) / p.z.powi(2);
    (-alpha / 2.0).exp()
}

fn main() {
    use std::vec::Vec;
    use rand::FromEntropy;
    use rayon::prelude::*;
    let beam_size = V3 {x:1.0, y:1.0, z:10.0};
    let xs: Vec<_> = (0..128).collect();
    let results: Vec<f64> = xs.par_iter().map(|_i| {
          let rng = rand::rngs::SmallRng::from_entropy();
          let walk = RandomWalk {
              rng: rng, diffusivity: 1.0, pos: V3::origin()
          };
          let steps: Vec<f64> = walk.map(|x| beam_intensity(beam_size, x)).take(10000000).collect();
          //let steps: Vec<V3<f64>> = walk.map(|x| beam_intensity(beam_size, x)).take(10000000).collect();
          correlate(100, 32, steps)
    }).collect();
    println!("hello {:?}", results);

    //debug!("Hello {}", steps);
}

fn mean<N, T>(iter: T) -> N where
    N: Num + std::convert::From<u32>,
    T: Iterator<Item=N>
{
    use std::convert::From;
    let (accum, count) = iter.fold((From::from(0 as u32), 0), |(accum, count): (N, u32), x| (accum+x, count+1));
    accum / From::from(count)
}

fn correlate(max_tau: usize, tau: usize, vec: Vec<f64>) -> f64 {
    mean(vec.iter().take(vec.len() - max_tau).zip(vec.iter().skip(tau)).map(|(x,y)| x*y))
}
