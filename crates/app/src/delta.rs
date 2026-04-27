//! Stream chunks from a chat driver to the renderer.

#[derive(Clone, Debug)]
pub struct ChatMessage {
    pub role:    String,
    pub content: String,
}

#[derive(Debug)]
pub enum Delta {
    Token(String),
    Done,
    Error(String),
}
