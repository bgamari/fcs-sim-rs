{ nixpkgs ? (import <nixpkgs> {}) }:

with nixpkgs;
callPackage (import ./fcs-sim.nix) {
  inherit (rustPlatform) buildRustPackage;
}
