{ buildRustPackage }:

buildRustPackage rec {
  name = "fcs-sim-${version}";
  version = "0.1.0";

  src = ./.;
  #depsSha256 = "0q68qyl2h6i0qsz82z840myxlnjay8p1w5z7hfyr8fqp7wgwa9cx";
  cargoSha256 = "1mayyq0vsaamzcyymgnf9jwk5pr73ik46ffw8id0r0f2v4kzpqbs";
}
