{ buildRustPackage }:

buildRustPackage rec {
  name = "fcs-sim-${version}";
  version = "0.1.0";

  src = ./.;
  cargoSha256 = "1syqc2ccx1vf7pcjy9yi9bww8s5111vp7d21nffjsgz9yi5qsrxh";
}
