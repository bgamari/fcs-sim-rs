{ buildRustPackage }:

buildRustPackage rec {
  name = "fcs-sim-${version}";
  version = "0.1.0";

  src = ./.;
  cargoSha256 = "1mayyq0vsaamzcyymgnf9jwk5pr73ik46ffw8id0r0f2v4kzpqbs";
}
