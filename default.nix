{ nixpkgs ? (import <nixpkgs> {}) }:
let
  nixpkgsSrc = fetchGit {
    url = "https://github.com/nixos/nixpkgs";
    rev = "7cd2e4ebe8ca91f829b405451586868744270100";
    ref = "release-19.03";
  };
  nixpkgs = import nixpkgsSrc {};
in
with nixpkgs;
callPackage (import ./fcs-sim.nix) {
  inherit (rustPlatform) buildRustPackage;
}
