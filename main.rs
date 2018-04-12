extern crate rand;
extern crate nalgebra as na;

use na::Vector3;
use na::geometry::Point3;
use rand::distributions::{Distribution};
use std::f64;

struct RandomWalk<Rng> {
    rng: Rng,
    diffusivity: f64,
    pos: Point3<f64>
}

impl<Rng: rand::Rng> std::iter::Iterator for RandomWalk<Rng> {
    type Item = Point3<f64>;
    fn next(&mut self) -> Option<Point3<f64>> {
        use rand::distributions::Normal;
        let r = Normal::new(0.0, self.diffusivity).sample(&mut self.rng);
        self.pos += OnSphere.sample(&mut self.rng) * r;
        Some(self.pos)
    }
}

struct OnSphere;

impl Distribution<Vector3<f64>> for OnSphere {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Vector3<f64> {
        use std::f64::consts::PI;
        use rand::distributions::Range;
        let phi = rng.sample(Range::new(0.0, PI));
        let theta = rng.sample(Range::new(-PI, PI));
        Vector3::new(theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos())
    }
}

fn beam_intensity(beam_size: Vector3<f64>, x: Point3<f64>) -> f64 {
    use std::iter::Iterator;
    let alpha: f64 = beam_size.iter().zip(x.iter()).map(|(w,x)| w.powi(2) / x.powi(2)).sum();
    (-alpha / 2.0).exp()
}

fn main() {
    use std::vec::Vec;
    use rand::NewRng;
    let mut rng = rand::SmallRng::new();
    let walk = RandomWalk {
        rng: rng, diffusivity: 1.0, pos: Point3::origin()
    };
    let steps: Vec<Point3<f64>> = walk.take(10000000).collect();
    let s: Vector3<f64> = steps.into_iter().map(|x| x - Point3::origin()).sum();
    println!("hello {:?}", s);
    //debug!("Hello {}", steps);
}
