# intelnav — development commands

default: check

check:
    cargo check --workspace --all-targets

build:
    cargo build --workspace --release

run *ARGS:
    cargo run -p intelnav-cli --release -- {{ARGS}}

chat:
    cargo run -p intelnav-cli --release -- chat

gateway:
    cargo run -p intelnav-cli --release -- gateway

fmt:
    cargo fmt --all

lint:
    cargo clippy --workspace --all-targets -- -D warnings

test:
    cargo test --workspace

clean:
    cargo clean
