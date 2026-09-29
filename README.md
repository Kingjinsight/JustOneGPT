# JustOneGPT

A nano GPT model trained from scratch on *明朝那些事儿*, a Chinese history book about the Ming dynasty.

I built this project as a hands-on way to understand the GPT architecture and training process, following Andrej Karpathy's [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) and drawing inspiration from [nanoGPT](https://github.com/karpathy/nanoGPT).

This is a learning project, and there is still plenty to improve. The current model can continue Chinese text, but its output is often repetitive and incoherent.

![Cover of 明朝那些事儿](assets/明朝那些事.png)

## Example generation

- **Prompt:** `嘉靖帝`
- **Continuation length:** up to 200 additional tokens

**Model output:**

> 嘉靖帝的命令，这两个人是徐阶相比，也算穷困，但徐阶依然以往，他们要考官位是徐阶，也是一个可怕的人。徐阶看了这两个人不是有点的，徐阶的命令：你们想找人，不到。徐阶的想法徐阶的理由，不是不住的，但徐阶明白，也算不能用说，徐阶的目标是有办理，那就算不可能如那么简单了，老老天爷的苦头来了。自从这个角度来就想了，如果你徐阶的意思考成绩单位是不够的，不是这样炼丹台，不了你怎么来干，也也不能出点，自己都不能
