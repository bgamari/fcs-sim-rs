{ buildRustPackage }:

buildRustPackage rec {
  name = "fcs-sim-${version}";
  version = "0.1.0";

  src = ./.;
  cargoSha256 = "1vb22q6vrrbr6dn299aw4xhldz5ws3mixn4yn34y3m7jx2276x3b";
}
